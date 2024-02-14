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


def write_df(df: pd.DataFrame | gpd.GeoDataFrame, out: os.PathLike, **kwargs) -> None:
    """Write a DataFrame or GeoDataFrame to file based on file extension."""
    parquet_exts = [".parquet", ".parq"]
    ext = Path(out).suffix
    if ext in parquet_exts:
        df.to_parquet(out, **kwargs)
    elif ext.lower() == ".gpkg":
        df.to_file(
            out, engine="pyogrio", **kwargs
        )  # pyright: ignore[reportGeneralTypeIssues, reportCallIssue]
    else:
        raise ValueError(
            f"Unsupported file format: {Path(out).suffix}. Consider using pandas"
            "directly to write this file."
        )


def read_gdf(src: os.PathLike, **kwargs) -> gpd.GeoDataFrame:
    """Read a GeoDataFrame from file based on file extension."""
    parquet_exts = [".parquet", ".parq"]
    if Path(src).suffix in parquet_exts:
        return gpd.read_parquet(src, **kwargs)

    return gpd.read_file(
        src, engine="pyogrio", **kwargs
    )  # pyright: ignore[reportGeneralTypeIssues, reportReturnType]


def read_df(src: os.PathLike, **kwargs) -> pd.DataFrame:
    """Read a DataFrame from file based on file extension."""
    parquet_exts = [".parquet", ".parq"]
    if Path(src).suffix in parquet_exts:
        return pd.read_parquet(src, **kwargs)

    raise ValueError(
        f"Unsupported file format: {Path(src).suffix}. Consider using pandas directly"
        "to read this file."
    )


def clip_df_to_bbox(df: pd.DataFrame, bounds: Iterable[float]) -> pd.DataFrame:
    """Clips a regular DataFrame with x and y columns in EPSG:4326 to the input bounds.
    Input bounds should be an iterable in the form of (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = bounds
    return df[
        (df["x"] >= xmin) & (df["y"] >= ymin) & (df["x"] <= xmax) & (df["y"] <= ymax)
    ]
