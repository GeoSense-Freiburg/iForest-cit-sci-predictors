"""Reproject the clipped forest mask to a new EPSG code."""
import logging
from pathlib import Path

import rioxarray as riox

from src.conf.parse_params import config
from src.utils.setup_logger import setup_logger

setup_logger()
log = logging.getLogger(__name__)


def reproject_raster_to_epsg(src: Path, out: Path, dst_epsg: int) -> None:
    """Reproject a raster to a new EPSG code."""
    src_raster = riox.open_rasterio(src)

    if isinstance(src_raster, list):
        raise ValueError("src must be a single raster file.")

    reproj_raster = src_raster.rio.reproject(dst_crs=f"EPSG:{dst_epsg}")

    reproj_raster.rio.to_raster(out)


def main(cfg: dict) -> None:
    """Run the script."""
    if cfg["forest_mask"]["verbose"]:
        log.setLevel(logging.INFO)

    log.info("Reprojecting...")
    reproject_raster_to_epsg(
        src=Path(cfg["forest_mask"]["clipped"]),
        out=Path(cfg["forest_mask"]["reproj"]),
        dst_epsg=cfg["forest_mask"]["dst_epsg"],
    )


if __name__ == "__main__":
    main(config)
