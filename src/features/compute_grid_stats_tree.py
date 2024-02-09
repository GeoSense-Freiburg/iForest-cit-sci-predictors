"""Compute grid stats for GBIF points at 10 m and 20 m resolutions across different query radii"""

import argparse
import itertools
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Generator, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from src.conf.parse_params import config
from src.utils.df_utils import read_gdf, write_df
from src.utils.log_utils import setup_logger, subprocess_logger

setup_logger()
log = logging.getLogger(__name__)


def get_centroids(shape: tuple, transform: Affine) -> Generator[Point, None, None]:
    """Get grid cell centroids from a raster with a given width, height, and transform.
    Centroids are returned in the flattened shape ((height * width), 2)
    """
    for row in range(shape[0]):
        for col in range(shape[1]):
            x_coord, y_coord = tuple(
                transform
                * (
                    col + 0.5,
                    row + 0.5,
                )
            )
            yield Point(x_coord, y_coord)


def get_buffers(
    centroids: Iterable[Point], radius: int
) -> Generator[Polygon, None, None]:
    """Get buffers around each centroid."""
    for centroid in centroids:
        yield centroid.buffer(radius)


def weighted_count(
    matches: gpd.GeoDataFrame | gpd.GeoSeries, point: Point, radius: int | float
) -> float:
    """Compute a weighted count of the number of points within a given distance of a point."""
    weights = (radius - matches.distance(point)) / radius
    return weights.sum()


def iterchunks(iterable: Iterable, size: int, args: list):
    """Yield successive n-sized chunks from an iterable."""
    for first in iterable:
        yield (itertools.chain([first], itertools.islice(iterable, size - 1)), *args)


def process_data(data: list[list]) -> pd.DataFrame:
    """Process the data and return a DataFrame with the correct dtypes."""
    dtypes = {
        "row": np.uint32,
        "col": np.uint32,
        "species_id": np.uint16,
        "count": np.uint32,
        "weighted_count": np.float32,
    }
    df = pd.DataFrame(
        data, columns=["row", "col", "species_id", "count", "weighted_count"]
    ).astype(dtypes)

    return df


def process_query_set(
    i_proc: int,
    raster_src: os.PathLike,
    points: gpd.GeoDataFrame,
    radius: int,
    out: os.PathLike,
) -> None:
    """Process a chunk of centroids and return a list of results."""
    sub_log = subprocess_logger()

    with rasterio.open(raster_src) as src:
        target_shape = (src.meta["height"], src.meta["width"])
        target_transform = src.transform
        resolution = int(abs(src.res[0]))

    query_set = f"{resolution} res, {radius} radius"
    sub_log.info("Computing stats for %s", query_set)

    sindex = points.sindex
    centroids = get_centroids(target_shape, target_transform)
    total = target_shape[0] * target_shape[1]

    data = []
    pbar = tqdm(total=total, desc=query_set, position=i_proc)
    for i, centroid in enumerate(centroids):
        buffer = centroid.buffer(radius)
        precise_matches_idx = sindex.query(buffer, predicate="intersects")
        if len(precise_matches_idx) > 0:
            centroid = buffer.centroid
            row, col = src.index(centroid.x, centroid.y)
            precise_matches = points.iloc[precise_matches_idx]
            for sp_id, obs in precise_matches.groupby("species_id"):
                count = len(obs)
                wt_count = weighted_count(
                    obs,  # pyright: ignore[reportGeneralTypeIssues]
                    centroid,
                    radius,
                )
                data.append([row, col, sp_id, count, wt_count])
        if i % 10_000 == 0:
            pbar.update(10_000)

    sub_log.info("Processing %s", query_set)
    df = process_data(data)
    sub_log.info("Writing %s", query_set)
    write_df(df, out)


def compute_grid_stats(cfg: dict, points: gpd.GeoDataFrame) -> None:
    """Compute grid stats for a raster and a set of points."""

    n_procs = cfg["stats"]["n_procs"]

    if n_procs == -1:
        n_procs = mp.cpu_count()

    args = []
    i = 1
    for res in [10, 20]:
        for radius in cfg["stats"]["radii"]:
            s2_src = cfg[f"s2_{res}m"]["src"]
            filename = f"{res}m_{radius}r.parquet"
            out = Path(cfg["stats"]["out_dir"]) / filename
            out.mkdir(parents=True, exist_ok=True)
            args.append((i, s2_src, points, radius, out))
            i += 1

    with mp.Pool(n_procs) as pool:
        log.info("Init multiprocessing.")
        pool.starmap(process_query_set, args)


def main(cfg: dict):
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("-r", "--resolution", type=int, help="Resolution of the grid.")
    parser.add_argument("-d", "--radius", type=int, help="Radius of the query.")
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.INFO)

    log.info("Reading data...")
    points_ids = read_gdf(cfg["gbif"]["points"])

    compute_grid_stats(cfg, points_ids)

    log.info("Done. ✅")


if __name__ == "__main__":
    main(config)
