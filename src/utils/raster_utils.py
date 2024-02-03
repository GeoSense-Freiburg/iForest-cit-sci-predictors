"""A collection of utility functions for working with raster data."""

import multiprocessing
import os
from pathlib import Path
from typing import Any, Optional

import rioxarray as riox
import xarray as xr


def da_to_raster(
    da: xr.DataArray,
    out: os.PathLike,
    dtype: Optional[Any] = None,
    num_threads: int = -1,
    **kwargs
) -> None:
    """Write a DataArray to a raster file."""
    dtype = dtype if dtype is not None else da.dtype
    if num_threads == -1:
        num_threads = multiprocessing.cpu_count()

    if Path(out).suffix == ".tif":
        tiff_opts = {
            "driver": "GTiff",
            "dtype": dtype,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "ZSTD",
            "num_threads": num_threads,
        }
        da.rio.to_raster(out, **tiff_opts, **kwargs)
    else:
        da.rio.to_raster(out, dtype=dtype, **kwargs)


def open_rasterio(src: os.PathLike, **kwargs) -> xr.DataArray | xr.Dataset:
    """A handler for rioxarray.open_rasterio that avoids annoying type hinting errors
    that arise from the rarely-used option to pass in a list of filenames and return a
    list of datasets.

    NOTE: This method should not be used if you want to use rioxarray.open_rasterio to
    return a list of datasets. Use the original method instead.
    """
    data = riox.open_rasterio(src, **kwargs)

    if isinstance(data, list):
        raise ValueError("src must be a single raster file with one or many bands.")

    return data
