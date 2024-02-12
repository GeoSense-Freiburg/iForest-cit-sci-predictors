"""Script to compute grid stats using a reverse tree query approach."""

import argparse
import gc
import logging
import multiprocessing as mp
import os
import shutil
from pathlib import Path
from typing import Generator, Iterable

import dask.dataframe as dd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from dask.diagnostics.progress import ProgressBar
from dask.distributed import Client
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

from src.conf.parse_params import config
from src.utils.df_utils import read_df, write_df
from src.utils.log_utils import setup_logger, subprocess_logger

setup_logger()
log = logging.getLogger(__name__)


def get_top_species(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Returns a dataframe with the top n species."""
    counts = df.species_id.value_counts()
    top_species = counts.nlargest(n).index
    filtered = df[df.species_id.isin(top_species)].reset_index(drop=True)
    return filtered  # pyright: ignore[reportReturnType]


def sp_idx_points(df: pd.DataFrame) -> list[tuple[int, np.ndarray]]:
    """Returns a list of tuples with species index and xy coordinates."""
    xy = np.c_[df.x, df.y]
    sp_idx = df.species_id
    return list(zip(sp_idx, xy))


def pixel_to_coord(row, col, transform):
    """Returns global coordinates to pixel center using base-0 raster index"""
    x = transform.c + transform.a * col + transform.b * row
    y = transform.f + transform.d * col + transform.e * row
    return x, y


def radial_mask(cell_radius: int) -> np.ndarray:
    """Returns a circular mask of cell radius r in units of pixels."""
    y, x = np.ogrid[-cell_radius : cell_radius + 1, -cell_radius : cell_radius + 1]
    mask = x**2 + y**2 <= cell_radius**2
    return mask


def neighborhood_indices(origin: np.ndarray, cell_radius: int) -> np.ndarray:
    """Returns the indices of a circular neighborhood around a point."""
    mask = radial_mask(cell_radius)
    indices = np.argwhere(mask) - cell_radius
    return indices + origin


def transform_indices(indices: np.ndarray, transform: Affine) -> np.ndarray:
    """Transforms pixel indices to geospatial coordinates."""
    transformed_idx = np.zeros(indices.shape)
    for i, centroid in enumerate(indices):
        transformed_idx[i, 0], transformed_idx[i, 1] = pixel_to_coord(
            centroid[0], centroid[1], transform
        )
    return transformed_idx


def get_kernel_centroids(
    origin: np.ndarray, cell_radius: int, transform: Affine
) -> tuple:
    """Returns the indices and centroid coordinates of a circular neighborhood around a
    point."""
    nbr_idx = neighborhood_indices(origin, cell_radius)
    nbr_centroids = nbr_idx + 0.5
    trans_centroids = transform_indices(nbr_centroids, transform)
    return nbr_idx, trans_centroids


def get_kernel_stats(
    kernel_idx: np.ndarray,
    kernel_centroids: np.ndarray,
    origin: np.ndarray,
    radius: int,
) -> list[tuple]:
    """Returns the indices and distance weights of the points within a circular neighborhood."""
    kernel_tree = KDTree(kernel_centroids)
    query_indices = kernel_tree.query_ball_point(origin, radius, p=2)

    distances = np.array(
        [euclidean(origin, kernel_centroids[i]) for i in query_indices]
    )

    weights = 1 - distances / radius

    return [(kernel_idx[j], weights[i]) for i, j in enumerate(query_indices)]


def chunk_list(lst: list, size: int) -> Generator[list, None, None]:
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def chunk_size(data_length: int, n: int) -> int:
    """Returns the number of chunks of size n that fit in data_length."""
    return data_length // n + (data_length % n > 0)


def get_chunk_stats(
    chunk: list[tuple], raster: rasterio.DatasetReader, radius: int
) -> Generator[tuple, None, None]:
    """Returns the species id, grid cell, and distance for each point in a chunk."""
    cell_radius = radius // int(abs(raster.res[0]))
    for sp_id, point in chunk:
        for grid_cell, weight in get_kernel_stats(
            *get_kernel_centroids(
                np.array(raster.index(point[0], point[1])),
                cell_radius,
                raster.transform,
            ),
            origin=point,
            radius=radius,
        ):
            yield (sp_id, f"{grid_cell[0]} {grid_cell[1]}", weight)


def stats_df(stats: Iterable[tuple]) -> pd.DataFrame:
    """Returns a dataframe with species id, grid cell, and distance."""
    return (
        pd.DataFrame(stats, columns=["species_id", "grid_cell", "weight"])
        .astype(
            {
                "species_id": "category",
                "grid_cell": "string[pyarrow]",
                "weight": np.float32,
            }
        )
        .groupby(["grid_cell", "species_id"], observed=True)
        .aggregate(count=("weight", "count"), mean_weight=("weight", "mean"))
        .astype({"count": "int16", "mean_weight": "float32"})
        .reset_index()
    )


def process_chunk(args: tuple) -> None:
    """Computes the stats for a chunk and writes the output to a parquet file."""
    chunk, i, raster_src, radius, out_dir = args
    sub_log = subprocess_logger(f"sub_proc_{i}")

    with rasterio.open(raster_src) as src:
        chunk_stats = get_chunk_stats(chunk, src, radius)

    sub_log.info("Building dataframe for chunk %s", i)
    df = stats_df(chunk_stats)

    sub_log.info("Writing stats for chunk %s", i)
    out = Path(out_dir, f"{int(abs(src.res[0]))}m_{radius}r_chunk_{i}.parquet")
    write_df(df, out, compression="zstd", compression_level=18)

    del chunk, chunk_stats, df, src
    gc.collect()


def aggregate_chunks(chunks_dir: os.PathLike) -> pd.DataFrame:
    """Aggregates the stats from the chunks and returns a dataframe."""
    ddf = (
        dd.read_parquet(  # pyright: ignore[reportPrivateImportUsage]
            list(Path(chunks_dir).glob("*.parquet")),
            dtype_backend="pyarrow",
        )
        .astype(
            {
                # "grid_cell": "string[pyarrow]",  # This seems to be _increasing_ mem usage??
                "species_id": "category",
                "count": "int16",
                "mean_weight": "float32",
            }
        )
        .repartition(npartitions=4000)  # pyright: ignore[reportAttributeAccessIssue]
    )
    # TODO: Calculate npartitions based on memory_usage of a single chunk dataframe.
    # Then set split_out (below) to half that (TBD; may need to change that ratio)

    # TODO: Change the operation here now that we're doing a first pass groupby when
    # creating the individual chunks
    total_counts = ddf.groupby(["grid_cell", "species_id"])["count"].transform("sum")
    ddf["total_count"] = total_counts

    result = ddf.groupby(["grid_cell", "species_id"]).aggregate

    result = ddf.assign(wt_count=ddf["count"] * ddf["mean_weight"]).astype(
        {
            "grid_cell": "string[pyarrow]",
            "species_id": "category",
            "count": "int16",
            "wt_count": "float32",
        }
    )
    result = result.drop(columns=["mean_weight"])

    return result


def cli_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("resolution", type=int, help="Resolution of the target raster.")
    parser.add_argument("radius", type=int, help="Radius of the circular neighborhood.")
    parser.add_argument("-a", "--agg-only", action="store_true", help="Aggregate only.")
    parser.add_argument("-k", "--keep", action="store_true", help="Keep chunk files.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Control verbosity of output."
    )
    parser.add_argument("-n", "--n_procs", type=int, help="Number of processes to use.")
    return parser.parse_args()


def main(cfg: dict) -> None:
    """Main function to compute grid stats."""
    args = cli_args()

    if args.verbose:
        log.setLevel(logging.INFO)

    if args.n_procs == -1:
        args.n_procs = mp.cpu_count()

    out_dir = Path(cfg["stats"]["out_dir"], f"{args.resolution}m", f"{args.radius}r")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.agg_only:
        log.info("Reading data...")
        idx_xy = (
            read_df(cfg["gbif"]["points"])
            .pipe(get_top_species, n=cfg["stats"]["n_species"])
            .pipe(sp_idx_points)
        )

        chunks = list(
            chunk_list(idx_xy, chunk_size(len(idx_xy), cfg["stats"]["n_chunks"]))
        )

        mp_args = [
            (chunk, i, cfg[f"s2_{args.resolution}m"]["src"], args.radius, out_dir)
            for i, chunk in enumerate(chunks)
        ]

        log.info("Initializing multiprocessing...")
        with mp.Pool(args.n_procs) as p:
            p.map(process_chunk, mp_args)

    log.info("Aggregating chunk stats...")
    client = Client(
        n_workers=args.n_procs,
        threads_per_worker=1,
        memory_limit=f"{200 / args.n_procs:.2f}GiB",
        dashboard_address=":36001",
        scheduler_port=0,
    )
    log.info("%s", client.dashboard_link)

    agg_out = Path(
        cfg["stats"]["out_dir"], f"{args.resolution}m_{args.radius}r_stats.parquet"
    )
    agg_stats = aggregate_chunks(out_dir)

    log.info("Writing aggregated stats...")
    with ProgressBar():
        agg_stats.to_parquet(
            agg_out,
            engine="pyarrow",
            write_options={"compression": "ZSTD", "compression_level": 15},
        )

    if not args.keep:
        log.info("Cleaning up...")
        shutil.rmtree(out_dir.parent)

    log.info("Done. ✅")


if __name__ == "__main__":
    main(config)
