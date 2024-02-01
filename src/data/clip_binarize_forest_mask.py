"""Clip the forest mask to the extent of the Sentinel-2 20m raster."""
import logging
from pathlib import Path

import numpy as np
import rioxarray as riox
import xarray as xr

from src.conf.parse_params import config
from src.utils.raster_utils import da_to_raster
from src.utils.setup_logger import setup_logger

setup_logger()
log = logging.getLogger(__name__)


def clip_raster_to_extent(src: Path, ref: Path) -> xr.DataArray:
    """
    Clip a raster to the extent of another raster. Returns the path of the clipped
    raster, where the path is out_dir / [src stem]_clipped.tif.
    """
    src_raster = riox.open_rasterio(src)
    ref_raster = riox.open_rasterio(ref)

    if isinstance(src_raster, list) or isinstance(ref_raster, list):
        raise ValueError("src_raster and ref_raster must be single-band rasters.")

    bounds = ref_raster.rio.transform_bounds(src_raster.rio.crs)

    clipped = src_raster.rio.clip_box(*bounds)

    # Cleanup because xarray (or rioxarray) isn't the best at memory management.
    src_raster.close()
    ref_raster.close()

    return clipped


def binarize_forest_mask(da: xr.DataArray) -> xr.DataArray:
    """
    Overwrite the values of a raster according to a set of threshold conditions.
    """
    da = da.where((da > 0) & (da <= 2), 0)
    da = da.where((da == 0) | (da > 2), 1)

    da = da.rio.write_nodata(0)

    return da


def main(cfg: dict):
    """Run the script."""
    if cfg["forest_mask"]["verbose"]:
        log.setLevel(logging.INFO)

    log.info("Clipping...")
    clipped = clip_raster_to_extent(
        src=Path(cfg["forest_mask"]["src"]),
        ref=Path(cfg["s2_20m"]["src"]),
    )

    log.info("Binarizing...")
    binarized = binarize_forest_mask(clipped)

    log.info("Writing...")
    da_to_raster(binarized, Path(cfg["forest_mask"]["clipped"], dtype="byte"))


if __name__ == "__main__":
    main(config)
