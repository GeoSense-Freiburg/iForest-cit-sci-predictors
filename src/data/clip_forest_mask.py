"""Clip the forest mask to the extent of the Sentinel-2 20m raster."""
import logging
from pathlib import Path

import rioxarray as riox

from src.conf.parse_params import config

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z",
)


def clip_raster_to_extent(src: Path, ref: Path, out: Path) -> Path:
    """
    Clip a raster to the extent of another raster. Returns the path of the clipped
    raster, where the path is out_dir / [src stem]_clipped.tif.
    """
    src_raster = riox.open_rasterio(src)
    ref_raster = riox.open_rasterio(ref)

    if isinstance(src_raster, list) or isinstance(ref_raster, list):
        raise ValueError("src_raster and ref_raster must be single-band rasters.")

    bounds = ref_raster.rio.transform_bounds(src_raster.rio.crs)

    log.info("Clipping raster...")
    clipped = src_raster.rio.clip_box(*bounds)

    # Cleanup because xarray (or rioxarray) isn't the best at memory management.
    src_raster.close()
    ref_raster.close()

    log.info("Writing raster...")
    clipped.rio.to_raster(out, dtype="uint8", windowed=True)

    return out


def main(cfg: dict):
    """Run the script."""
    clip_raster_to_extent(
        src=Path(cfg["forest_mask"]["src"]),
        ref=Path(cfg["s2_20m"]["src"]),
        out=Path(cfg["forest_mask"]["clipped"]),
    )


if __name__ == "__main__":
    main(config)
