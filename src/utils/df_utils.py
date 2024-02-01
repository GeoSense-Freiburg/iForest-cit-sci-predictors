"""Utility functions for working with DataFrames and GeoDataFrames."""
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd


def chain_log(
    df: pd.DataFrame | gpd.GeoDataFrame, msg: str, name: str = __name__
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Simple logging function to log messages chaining methods on a DataFrame."""
    log = logging.getLogger(name)
    log.info(msg)
    return df


def write_gdf(gdf: gpd.GeoDataFrame, out: Path, **kwargs) -> None:
    """Write a GeoDataFrame to file."""
    parquet_exts = [".parquet", ".parq"]
    if out.suffix in parquet_exts:
        gdf.to_parquet(out, **kwargs)
    else:
        gdf.to_file(out, **kwargs)
