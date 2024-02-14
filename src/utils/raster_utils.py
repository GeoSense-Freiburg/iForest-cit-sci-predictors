"""A collection of utility functions for working with raster data."""

import multiprocessing
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
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


def pack_dataset(
    ds: xr.Dataset, nodata: bool = True, signed: bool = False
) -> xr.Dataset:
    """
    Pack an xarray Dataset into int16 format.

    Uses https://docs.unidata.ucar.edu/nug/current/best_practices.html as reference.
    """
    with xr.set_options(keep_attrs=True):
        for dv in ds.data_vars:
            if np.issubdtype(ds[dv].dtype, np.floating):
                min_val = ds[dv].min().item()
                max_val = ds[dv].max().item()

                bit_exp = 16
                bit_count = 2**bit_exp

                if nodata:
                    nodata_val = -(2**15)
                    scale_factor = np.float32((max_val - min_val) / (bit_count - 2))

                    if signed:
                        offset = (max_val + min_val) / 2
                    else:
                        offset = min_val - scale_factor

                    ds[dv] = ds[dv].fillna(nodata_val)
                    ds[dv] = ds[dv].rio.write_nodata(nodata_val, encoded=True)
                    ds[dv].attrs["_FillValue"] = nodata_val

                else:
                    scale_factor = (max_val - min_val) / (bit_count - 1)

                    if signed:
                        offset = min_val + 2 ** (bit_exp - 1) * scale_factor
                    else:
                        offset = min_val

                ds[dv] = (ds[dv] - offset) / scale_factor
                ds[dv] = ds[dv].astype(np.int16)

                ds[dv].attrs["scale_factor"] = scale_factor
                ds[dv].attrs["add_offset"] = offset
            else:
                ds[dv].attrs["scale_factor"] = 1.0
                ds[dv].attrs["add_offset"] = 0

    return ds
