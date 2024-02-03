"""Clip the forest mask to the extent of the Sentinel-2 20m raster."""

import logging

import numpy as np
import xarray as xr

from src.conf.parse_params import config
from src.utils.raster_utils import da_to_raster, open_rasterio
from src.utils.setup_logger import setup_logger

setup_logger()
log = logging.getLogger(__name__)


def rough_clip(
    src_raster: xr.DataArray | xr.Dataset, ref_raster: xr.DataArray | xr.Dataset
) -> xr.DataArray:
    """
    Clip a raster to the extent of another raster. Returns the path of the clipped
    raster, where the path is out_dir / [src stem]_clipped.tif.
    """
    bounds = ref_raster.rio.transform_bounds(src_raster.rio.crs)
    clipped = src_raster.rio.clip_box(*bounds)

    return clipped


def match_raster_to_ref(
    src_raster: xr.DataArray | xr.Dataset, ref_raster: xr.DataArray | xr.Dataset
) -> xr.DataArray:
    """Reproject a raster to a new EPSG code."""

    return src_raster.rio.reproject_match(ref_raster)


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

    log.info("Reading data...")
    forest_mask = open_rasterio(cfg["forest_mask"]["src"])
    s2_10m = open_rasterio(cfg["s2_10m"]["src"])

    log.info("Clipping (rough)...")
    clipped = rough_clip(forest_mask, s2_10m)
    forest_mask.close()

    log.info("Binarizing...")
    binarized = binarize_forest_mask(clipped)
    clipped.close()

    log.info("Matching to Sentinel-2 10m...")
    matched = binarized.rio.reproject_match(s2_10m)
    s2_10m.close()
    binarized.close()

    log.info("Writing...")
    da_to_raster(matched, cfg["forest_mask"]["matched"])


if __name__ == "__main__":
    main(config)
