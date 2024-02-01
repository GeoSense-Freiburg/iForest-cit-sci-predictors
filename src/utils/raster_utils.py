"""A collection of utility functions for working with raster data."""
import multiprocessing
from pathlib import Path
from typing import Any, Optional

import xarray as xr


def da_to_raster(
    da: xr.DataArray,
    out: Path,
    dtype: Optional[Any] = None,
    num_threads: int = -1,
    **kwargs
) -> None:
    """Write a DataArray to a raster file."""
    dtype = dtype if dtype is not None else da.dtype
    if num_threads == -1:
        num_threads = multiprocessing.cpu_count()

    if out.suffix == ".tif":
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
