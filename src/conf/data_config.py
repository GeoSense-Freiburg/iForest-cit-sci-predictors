"""Data setup config script."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())  # load environment variables

ext_raw_data_dir = os.getenv("EXT_RAW_DATA_DIR")
local_raw_data_dir = os.getenv("LOCAL_RAW_DATA_DIR")

if ext_raw_data_dir is None or local_raw_data_dir is None:
    raise ValueError(
        "Please set the environment variables EXT_RAW_DATA_DIR and LOCAL_RAW_DATA_DIR"
    )


@dataclass
class GbifConfig:
    """GBIF data config."""

    src: Path = Path(
        ext_raw_data_dir,
        "gbif",
        "all_tracheophyta_non-cult_2024-01-21",
        "all_tracheophyta_non-cult_2024-01-21.parquet",
    )


@dataclass
class GermanyMaskConfig:
    """Germany mask config."""

    src: Path = Path(ext_raw_data_dir, "masks/regions/germany/germany.geojson")


@dataclass
class ForestMaskConfig:
    """Forest mask config."""

    src: Path = Path(
        ext_raw_data_dir, "masks/forest_type_2018/FTY_2018_010m_de_03035_v010/DATA"
    )
    out: Path = Path(os.getcwd(), "data/interim/forest_mask.gpkg")
    n_procs: int = -1
    verbose: bool = True


@dataclass
class S210mConfig:
    """Sentinel-2 10m config."""

    src: Path = Path(local_raw_data_dir, "sentinel-2/fca_grid_empty_raster_R10m.tif")


@dataclass
class S220mConfig:
    """Sentinel-2 20m config."""

    src: Path = Path(local_raw_data_dir, "sentinel-2/fca_grid_empty_raster_R20m.tif")


@dataclass
class DataConfig:
    """Data setup config."""

    raw_data_dir: Path = Path(ext_raw_data_dir)
    gbif: GbifConfig = GbifConfig()
    de_mask: GermanyMaskConfig = GermanyMaskConfig()
    fst_mask: ForestMaskConfig = ForestMaskConfig()
    s2_10m: S210mConfig = S210mConfig()
    s2_20m: S220mConfig = S220mConfig()


# Export as instantiated config object
data_config = DataConfig()
