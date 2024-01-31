"""
Script to setup the forest mask to apply to GBIF data. Process involves polygonizing the
Copernicus Forest Type 2018 .tif files and saving as an interim GPKG.
"""
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes

from src.conf.parse_params import config
from src.utils.setup_logger import setup_logger

setup_logger()
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
            for _, (s, v) in enumerate(shapes(image, transform=src.transform))
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
        .reset_index(drop=True)
        .dissolve()
        .reset_index(drop=True)
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

    return (
        gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True).reset_index(drop=True))
        .dissolve()
        .reset_index(drop=True)
    )  # pyright: ignore[reportGeneralTypeIssues]


def write_gdf(gdf: gpd.GeoDataFrame, out: Path, **kwargs) -> None:
    """Write a GeoDataFrame to file."""
    parquet_exts = [".parquet", ".parq"]
    if out.suffix in parquet_exts:
        gdf.to_parquet(out, **kwargs)
    else:
        gdf.to_file(out, **kwargs)


def main(cfg: dict) -> None:
    """Main function."""
    if cfg["forest_mask"]["verbose"]:
        log.setLevel(logging.INFO)

    Path(cfg["forest_mask"]["final"]).parent.mkdir(parents=True, exist_ok=True)

    log.info("Polygonizing raster...")
    forest_type_gdf = polygonize_forest_type_raster(cfg["forest_mask"]["reproj"])

    log.info("Saving mask...")
    write_gdf(forest_type_gdf, Path(cfg["forest_mask"]["final"]), index=False)

    log.info("Done.")


if __name__ == "__main__":
    main(config)
