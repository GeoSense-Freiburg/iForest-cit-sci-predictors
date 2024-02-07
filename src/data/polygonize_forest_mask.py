"""
Script to setup the forest mask to apply to GBIF data. Process involves polygonizing the
Copernicus Forest Type 2018 .tif files and saving as an interim GPKG.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import shapes

from src.conf.parse_params import config
from src.utils.df_utils import chain_log, write_df
from src.utils.setup_logger import setup_logger

setup_logger()
log = logging.getLogger(__name__)


def polygonize_raster(filename: str | Path, n_procs: int = -1) -> gpd.GeoDataFrame:
    """Polygonize a single-band raster image"""

    if n_procs == -1:
        n_procs = mp.cpu_count()

    with rasterio.open(filename) as src:
        image = src.read(1)
        crs = src.crs

        # Split the image into chunks along the first axis
        chunks = np.array_split(image, n_procs)

        # Create a process pool and apply the polygonize_chunk function to each chunk
        with ProcessPoolExecutor(max_workers=n_procs) as executor:
            results = executor.map(
                _polygonize_chunk,
                [
                    (
                        chunk,
                        src.transform * Affine.translation(0, i * chunk.shape[0]),
                    )
                    for i, chunk in enumerate(chunks)
                ],
            )

    # Flatten the list of results and create a GeoDataFrame
    results = [item for sublist in results for item in sublist]

    return gpd.GeoDataFrame.from_features(list(results), crs=crs)


def _polygonize_chunk(args):
    image_chunk, transform = args
    return list(
        {"properties": {"raster_val": v}, "geometry": s}
        for _, (s, v) in enumerate(
            shapes(image_chunk, mask=image_chunk, transform=transform)
        )
    )


def polygonize_forest_type_raster(
    filename: str | Path, n_procs: int = -1
) -> gpd.GeoDataFrame:
    """
    Polygonize a single-band raster image from the Copernicus Forest Type 2018 and merge
    forest types into a single mask of multi-polygons.
    """
    gdf = (
        polygonize_raster(filename, n_procs)  # pyright: ignore[reportGeneralTypeIssues]
        .pipe(chain_log, msg="Selecting only forest geometries")
        .pipe(lambda df_: df_[df_.raster_val.isin([1])])
        .pipe(chain_log, msg="Dropping NAs")
        .drop(columns="raster_val")
        .dropna(ignore_index=True)
    )
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


def main(cfg: dict) -> None:
    """Main function."""
    if cfg["forest_mask"]["verbose"]:
        log.setLevel(logging.INFO)

    Path(cfg["forest_mask"]["final"]).parent.mkdir(parents=True, exist_ok=True)

    log.info("Polygonizing raster...")
    forest_type_gdf = polygonize_forest_type_raster(
        cfg["forest_mask"]["reproj"], n_procs=cfg["forest_mask"]["n_procs"]
    )

    log.info("Saving mask...")
    write_df(forest_type_gdf, Path(cfg["forest_mask"]["final"]), index=False)

    log.info("Done.")


if __name__ == "__main__":
    main(config)
