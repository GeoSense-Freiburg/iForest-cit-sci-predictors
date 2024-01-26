"""
Script to setup the forest mask to apply to GBIF data. Process involves polygonizing the
Copernicus Forest Type 2018 .tif files and saving as an interim shapefile.
"""
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Iterable

import click
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes

log = logging.getLogger(__name__)


def polygonize_raster(filename: str | Path) -> gpd.GeoDataFrame:
    """Polygonize a single-band raster image"""
    with rasterio.open(filename) as src:
        # Read the first band
        image = src.read(1)

        # Get the CRS
        crs = src.crs

        # Polygonize the raster
        results = (
            {"properties": {"raster_val": v}, "geometry": s}
            for i, (s, v) in enumerate(shapes(image, transform=src.transform))
        )

    return gpd.GeoDataFrame.from_features(list(results), crs=crs)


def polygonize_forest_type_raster(filename: str | Path) -> gpd.GeoDataFrame:
    """
    Polygonize a single-band raster image from the Copernicus Forest Type 2018 and merge
    forest types into a single mask of multi-polygons.
    """
    gdf = (
        polygonize_raster(filename)  # pyright: ignore[reportGeneralTypeIssues]
        .pipe(lambda df_: df_[df_.raster_val.isin([1, 2])])
        .drop(columns="raster_val")
        .dropna(ignore_index=True)
    )
    log.info("Polygonized %s", filename)
    return gdf


def polygonize_and_merge_forest_type_tiles(
    filenames: Iterable[str | Path], n_procs: int = -1
) -> gpd.GeoDataFrame:
    """Polygonize and merge a set of Copernicus Forest Type 2018 tiles."""
    if n_procs < -1 or n_procs == 0:
        raise ValueError("n_procs must be -1 or a positive integer.")

    n_procs = mp.cpu_count() if n_procs == -1 else n_procs

    with mp.Pool(n_procs) as p:
        gdfs = p.map(polygonize_forest_type_raster, filenames)

    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True).reset_index(drop=True))


@click
def main() -> None:
    ...
