"""Reproject the clipped forest mask to a new EPSG code."""

import logging
from pathlib import Path

import rioxarray as riox
import xarray as xr

from src.conf.parse_params import config
from src.utils.raster_utils import da_to_raster, open_rasterio
from src.utils.setup_logger import setup_logger

setup_logger()
log = logging.getLogger(__name__)


def reproject_raster_to_epsg(src: Path, dst_epsg: int) -> xr.DataArray:
    """Reproject a raster to a new EPSG code."""
    src_raster = riox.open_rasterio(src)

    if isinstance(src_raster, list):
        raise ValueError("src must be a single raster file.")

    return src_raster.rio.reproject(dst_crs=f"EPSG:{dst_epsg}")


def main(cfg: dict) -> None:
    """Run the script."""
    if cfg["forest_mask"]["verbose"]:
        log.setLevel(logging.INFO)

    log.info("Reprojecting...")
    reprojected = reproject_raster_to_epsg(
        src=Path(cfg["forest_mask"]["clipped"]),
        dst_epsg=cfg["forest_mask"]["dst_epsg"],
    )

    # Clip the raster to the Sentinel-2 extent (again) because the previous clip was
    # done prior to reprojection and so has some errors.
    log.info("Clipping raster...")
    s2_20m = open_rasterio(cfg["s2_20m"]["src"])
    reprojected = reprojected.rio.clip_box(*s2_20m.rio.bounds())

    log.info("Writing...")
    da_to_raster(reprojected, Path(cfg["forest_mask"]["reproj"]))


if __name__ == "__main__":
    main(config)
