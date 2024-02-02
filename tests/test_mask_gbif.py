"""Tests for the mask_gbif module."""

import logging

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from shapely.geometry import Point

from src.data.mask_gbif import mask_points_with_raster
from src.utils.setup_logger import setup_logger

setup_logger()
log = logging.getLogger(__name__)


def test_mask_points_with_raster(tmp_path):
    """Test the mask_points_with_raster function."""
    points_src = tmp_path / "test_points.parquet"
    raster_src = tmp_path / "test_raster.tif"
    crs = CRS.from_epsg(4326)

    # Create a sample GeoDataFrame with points
    points = gpd.GeoDataFrame(  # pyright: ignore[reportGeneralTypeIssues]
        {"geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)]}, crs=crs
    )

    # Save the points to a shapefile
    points.to_parquet(points_src)

    # Create a sample raster with values
    raster_data = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
    with rasterio.open(
        raster_src,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="uint8",
        crs=crs,
    ) as dst:
        dst.write(raster_data, 1)

    # Call the function to be tested
    result = mask_points_with_raster(points_src, raster_src)

    # Assert the expected output
    expected_result = gpd.GeoDataFrame(  # pyright: ignore[reportGeneralTypeIssues]
        {
            "geometry": [Point(0, 0), Point(2, 2)],
        },
        crs=crs,
    )

    log.info(expected_result)
    log.info(result)

    assert result.equals(expected_result)
