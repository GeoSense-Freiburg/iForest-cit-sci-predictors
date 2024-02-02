"""Utility functions for working with DataFrames and GeoDataFrames."""
import logging
import os
from pathlib import Path
from typing import Iterable

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
        gdf.to_file(out, engine="pyogrio", **kwargs)


def read_gdf(src: str | os.PathLike, **kwargs) -> gpd.GeoDataFrame:
    """Read a GeoDataFrame from file."""
    parquet_exts = [".parquet", ".parq"]
    if Path(src).suffix in parquet_exts:
        return gpd.read_parquet(src, **kwargs)

    return gpd.read_file(
        src, engine="pyogrio", **kwargs
    )  # pyright: ignore[reportGeneralTypeIssues]


def clip_df_to_bbox(df: pd.DataFrame, bounds: Iterable[float]) -> pd.DataFrame:
    """Clips a regular DataFrame with x and y columns in EPSG:4326 to the input bounds.
    Input bounds should be an iterable in the form of (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = bounds
    return df[
        (df["x"] >= xmin) & (df["y"] >= ymin) & (df["x"] <= xmax) & (df["y"] <= ymax)
    ]
