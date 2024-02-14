"""Clip GBIF data by extent of Sentinel-2 rasters."""

import logging
from functools import partial
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from src.conf.parse_params import config
from src.utils.df_utils import chain_log, clip_df_to_bbox, write_df
from src.utils.log_utils import setup_logger
from src.utils.raster_utils import open_rasterio

setup_logger()
log = logging.getLogger(__name__)

chlog = partial(chain_log, name=__name__)


def gbif_set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Set dtypes for GBIF data."""
    log.info("Setting dtypes...")
    dtypes = {
        "species": str,
        "decimallatitude": np.float32,
        "decimallongitude": np.float32,
    }

    return df.astype(dtypes)


def gbif_clip_to_extent(df: pd.DataFrame, bounds: Iterable) -> pd.DataFrame:
    """Clip a GBIF DataFrame to the input bounds."""
    log.info("Clipping to Sentinel-2 extent...")
    return (
        df.rename(columns={"decimallongitude": "x", "decimallatitude": "y"})
        .pipe(clip_df_to_bbox, bounds=bounds)
        .reset_index(drop=True)
    )


def gbif_df_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert a GBIF DataFrame to a GeoDataFrame."""
    log.info("Converting to GeoDataFrame...")
    return (
        df.assign(
            geometry=lambda df_: gpd.points_from_xy(df_["x"], df_["y"], crs="EPSG:4326")
        )
        .drop(columns=["x", "y"])
        .pipe(gpd.GeoDataFrame)
    )


def clip_gbif_to_s2(
    df: pd.DataFrame, s2: xr.DataArray | xr.Dataset
) -> gpd.GeoDataFrame:
    """Clip GBIF data by extent of Sentinel-2 rasters and mask with a shapefile
    containing polygons pertaining to forested areas."""

    return (  # pyright: ignore
        df.dropna()
        .pipe(gbif_set_dtypes)
        .pipe(
            gbif_clip_to_extent,
            bounds=s2.rio.transform_bounds(dst_crs="EPSG:4326"),
        )
        .pipe(gbif_df_to_gdf)
        .reset_index(drop=True)
        .pipe(chlog, msg="Projecting to UTM 32N")
        .to_crs(crs=s2.rio.crs)
        .clip(s2.rio.bounds())  # pyright: ignore[reportOptionalMemberAccess]
        .reset_index(drop=True)
    )


def main(cfg: dict) -> None:
    """Run the script."""
    if cfg["gbif"]["verbose"]:
        log.setLevel(logging.INFO)

    log.info("Reading data...")
    cols = ["species", "decimallatitude", "decimallongitude"]
    df = pd.read_parquet(cfg["gbif"]["src"], columns=cols)
    s2_raster = open_rasterio(Path(cfg["s2_20m"]["src"]))

    gdf = clip_gbif_to_s2(
        df=df,
        s2=s2_raster,
    )

    # Make the output directory if it doesn't exist
    Path(cfg["gbif"]["clipped"]).parent.mkdir(parents=True, exist_ok=True)
    write_df(gdf, Path(cfg["gbif"]["clipped"]))


if __name__ == "__main__":
    main(config)
