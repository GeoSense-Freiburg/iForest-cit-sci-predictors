"""Compute grid cell statistics for each species in the masked GBIF dataset."""

import argparse
import gc
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cupy as cp
import cupyx.scipy.signal as cpx_signal
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rioxarray as riox  # pylint: disable=unused-import
import xarray as xr
from affine import Affine

from src.conf.parse_params import config
from src.utils.df_utils import read_df
from src.utils.log_utils import setup_logger, subprocess_logger
from src.utils.raster_utils import pack_dataset

setup_logger()
log = logging.getLogger(__name__)


def get_cell_counts(points: np.ndarray, src: rasterio.DatasetReader) -> np.ndarray:
    """Get the counts of points in each grid cell."""
    counts = np.zeros((src.height, src.width), dtype=int)
    for point in points:
        row, col = src.index(point[0], point[1])
        counts[row, col] += 1

    return counts


def radial_kernel(
    radius: int, weighted: bool = False, sigma: Optional[float | int] = None
) -> np.ndarray:
    """Create a radial kernel for convolution."""
    sigma = radius * 0.5 if sigma is None else sigma
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    distances_squared = x**2 + y**2

    if weighted:
        kernel = np.exp(-distances_squared / (2 * sigma**2))
        kernel[distances_squared > radius**2] = 0
        return kernel

    kernel = x**2 + y**2 <= radius**2
    return kernel.astype(int)


def get_radial_counts(
    counts: np.ndarray,
    radius: int,
    weighted: bool = False,
    sigma: Optional[float | int] = None,
    gpu_id: Optional[int] = None,
) -> np.ndarray:
    """Get the species counts in a given radius for each grid cell."""
    dtype = np.float32 if weighted else np.uint16
    if gpu_id is None:
        # Split processes across GPUs
        gpu_id = 0 if weighted else 1
    with cp.cuda.Device(gpu_id):
        counts_gpu = cp.asarray(counts, dtype=dtype)
        kernel_gpu = cp.asarray(radial_kernel(radius, weighted, sigma), dtype=dtype)
        radial_counts_gpu = (
            cpx_signal.convolve2d(  # pyright: ignore[reportGeneralTypeIssues]
                counts_gpu, kernel_gpu, mode="same"
            )
        )

    return cp.asnumpy(radial_counts_gpu)


def write_species_stats_as_npz(
    data: list[np.ndarray],
    out: os.PathLike,
) -> None:
    """Write species stats to a compressed .npz file."""
    np.savez_compressed(out, counts=data[0], wt_counts=data[1])


def shrink_arr(arr: np.ndarray) -> np.ndarray:
    """Shrink an array to only include non-zero values."""
    idx = np.where(arr > 0)
    values = arr[idx]
    result = np.stack(idx + (values,), axis=-1)
    return result


@dataclass
class SpeciesSet:
    """Dataclass for species data.

    Attributes:
        id (str): The ID of the species set.
        df (pd.DataFrame): The DataFrame containing the species data.
        resolution (int): The resolution of the grid.
        radius (int): The radius used for computing grid statistics.
    """

    id: str
    df: pd.DataFrame
    resolution: int
    radius: int


@dataclass
class RefRaster:
    """Reference raster data."""

    crs: pyproj.CRS
    transform: Affine


def write_species_stats_as_netcdf(
    sp: SpeciesSet,
    data: list[np.ndarray],
    ref_raster: RefRaster,
    out: os.PathLike,
) -> None:
    """Write the species stats data to a packed NetCDF."""

    data_names = ["counts", "wt_counts"]
    data_arrays = {}
    for stat, stat_name in zip(data, data_names):
        da = (
            xr.DataArray(
                stat,
                dims=["y", "x"],
                attrs={
                    "transform": ref_raster.transform,
                    "crs": ref_raster.crs,
                    "long_name": f"sp{sp.id}_{stat_name}",
                },
            )
            .rio.write_crs(ref_raster.crs)
            .rio.write_transform(ref_raster.transform)
        )
        data_arrays[f"sp{sp.id}_{stat_name}"] = da

    ds = xr.Dataset(data_arrays)
    ds = ds.rio.write_crs(ref_raster.crs).rio.write_transform(ref_raster.transform)

    # Pack the Dataset and add compression to minimize space requirements
    ds = pack_dataset(ds, nodata=False, signed=False)
    encoding = {var: {"zlib": True, "complevel": 9} for var in ds.data_vars}

    ds.to_netcdf(out, engine="h5netcdf", encoding=encoding)
    ds.close()
    del ds, da, data_arrays
    gc.collect()


def write_handler(
    sp: SpeciesSet,
    data: list[np.ndarray],
    ref_raster: RefRaster,
    writer: str,
    out_dir: str | os.PathLike,
):
    """Write the species stats to the desired format."""
    writers = ["npz", "netcdf"]
    if writer not in writers:
        raise ValueError(f"Writer must be one of {writers}")

    out = Path(
        out_dir,
        f"{sp.resolution}m",
        f"{sp.radius}r",
        f"{sp.resolution}m_{sp.radius}r_sp{sp.id}",
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    if writer == "npz":
        out = out.with_suffix(".npz")
        write_species_stats_as_npz(data, out)
    else:
        out = out.with_suffix(".nc")
        write_species_stats_as_netcdf(sp, data, ref_raster, out)


def process_species(
    sp_set: SpeciesSet,
    raster_src: str | os.PathLike,
    out_dir: str | os.PathLike,
    writer: str,
    gpu_id: Optional[int] = None,
) -> None:
    """Process the species data and write the results to disk."""
    sub_log = subprocess_logger(f"sp_{sp_set.id}")

    with rasterio.open(raster_src) as src:
        ref_raster = RefRaster(src.crs, src.transform)
        points = np.c_[sp_set.df.x, sp_set.df.y]
        cell_counts = get_cell_counts(points, src)

    # Define the radius in terms of grid cells
    cell_radius = sp_set.radius // sp_set.resolution

    sub_log.info("Processing counts for species %s", sp_set.id)
    counts = get_radial_counts(cell_counts, cell_radius, gpu_id=gpu_id).astype(
        np.uint16
    )
    sub_log.info("Processing weighted counts for species %s", sp_set.id)
    wt_counts = get_radial_counts(cell_counts, cell_radius, True, gpu_id=gpu_id).astype(
        np.float32
    )

    sub_log.info("Writing data for species %s", sp_set.id)

    data = [counts, wt_counts]

    write_handler(
        sp_set,
        data,
        ref_raster,
        writer=writer,
        out_dir=out_dir,
    )


def top_n_species(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Get the top n species by occurrence count."""
    counts = df.species_id.value_counts()
    top_species = counts.nlargest(n).index
    return df[df.species_id.isin(top_species)].reset_index(drop=True)


def cli() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute grid statistics")
    parser.add_argument("resolution", type=int, help="Resolution of the target raster.")
    parser.add_argument("-g", "--gpu-id", type=int, help="GPU device to use.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Control verbosity of output."
    )
    parser.add_argument("-n", "--n_procs", type=int, help="Number of processes to use.")
    return parser.parse_args()


def main(cfg: dict) -> None:
    """Run the script."""
    args = cli()

    if args.verbose:
        log.setLevel(logging.INFO)

    log.info("Reading data...")
    points = top_n_species(read_df(cfg["gbif"]["points"]), cfg["stats"]["n_species"])

    out_dir = Path(cfg["stats"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for radius in cfg["stats"]["radii"]:
        query_set = f"{args.resolution}m_{radius}r"
        log.info("Preparing %s...", query_set)
        mp_args = [
            (
                SpeciesSet(str(sp_id), sp_df, int(args.resolution), radius),
                cfg[f"s2_{args.resolution}m"]["src"],
                cfg["stats"]["out_dir"],
                cfg["stats"]["writer"],
                args.gpu_id,
            )
            for sp_id, sp_df in points.groupby("species_id")
        ]

        log.info("Init multiprocessing")
        with mp.Pool(args.n_procs) as p:
            p.starmap(process_species, mp_args)

    log.info("Done. ✅✅✅✅✅✅✅✅✅")


if __name__ == "__main__":
    main(config)
if __name__ == "__main__":
    main(config)
