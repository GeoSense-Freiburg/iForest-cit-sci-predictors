"""Mask GBIF data with the forest mask."""

import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from src.conf.parse_params import config
from src.utils.df_utils import read_gdf, write_df
from src.utils.log_utils import setup_logger

setup_logger()
log = logging.getLogger(__name__)


def mask_points_with_raster(
    points_src: os.PathLike, raster_src: os.PathLike
) -> gpd.GeoDataFrame:
    """Mask points with a binary raster."""

    log.info("Loading data...")
    points = read_gdf(points_src)

    with rasterio.open(raster_src) as src:
        points_xy = list(zip(points.geometry.x, points.geometry.y))

        log.info("Extracting raster values at points...")
        raster_values = np.fromiter(
            (samp[0] for samp in src.sample(points_xy, 1, masked=False)), dtype=np.uint8
        )

        log.info("Adding raster values to points dataframe...")
        points["raster_value"] = raster_values

    log.info("Cleaning up...")
    return (  # pyright: ignore
        points.pipe(lambda df_: df_[df_.raster_value == 1])
        .drop(columns=["raster_value"])  # pyright: ignore
        .reset_index(drop=True)
    )


def main(cfg: dict):
    """Run the script."""
    if cfg["gbif"]["verbose"]:
        log.setLevel(logging.INFO)

    masked_gdf = mask_points_with_raster(
        points_src=cfg["gbif"]["clipped"], raster_src=cfg["forest_mask"]["matched"]
    )

    log.info("Writing masked points...")
    write_df(masked_gdf, out=Path(cfg["gbif"]["masked"]))

    log.info("Done.")


if __name__ == "__main__":
    main(config)
