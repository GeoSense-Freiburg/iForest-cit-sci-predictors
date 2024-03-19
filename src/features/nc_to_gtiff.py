"""Convert netCDF files to GeoTIFFs."""

import gc
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import rioxarray as riox
import xarray as xr
from dotenv import find_dotenv, load_dotenv
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from src.conf.parse_params import config
from src.utils.log_utils import setup_logger
from src.utils.raster_utils import da_to_raster, open_rasterio

load_dotenv(find_dotenv())
data_root = Path(os.environ["DATA_ROOT"])

setup_logger()
log = logging.getLogger(__name__)

NC_TO_GTIFF = False
ROTATE_TIFFS = True

# Ignore georeference warnings because the netcdf files were saved incorrectly (somehow)
# and don't contain a CRS on load.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def rotate_tifs(radius_dir: Path):
    tif_dir = radius_dir / "tif"
    out_dir = tif_dir / "rot"
    out_dir.mkdir(exist_ok=True, parents=True)

    for rast_file in tqdm(list(tif_dir.glob("*.tif")), desc=radius_dir.stem):
        data = riox.open_rasterio(rast_file)
        # set the data matrices of each data var to their absolute values to remove negatives
        data[1] = data[1].where(data[1] >= 0, -data[1])
        data = data.assign_coords({"y": data.y[::-1]})
        da_to_raster(data, out_dir / rast_file.name, dtype=np.float32)
        data.close()
        del data
        gc.collect()


def nc_to_gtiffs(radius_dir: Path, ref_rast: xr.DataArray | xr.Dataset):
    ref_crs = ref_rast.rio.crs
    ref_transform = ref_rast.rio.transform()

    tif_dir = radius_dir / "tif"
    tif_dir.mkdir(exist_ok=True, parents=True)

    for nc_file in tqdm(list(radius_dir.glob("*.nc")), desc=radius_dir.stem):
        data = riox.open_rasterio(nc_file, mask_and_scale=True, parse_coordinates=False)
        data = data.rio.write_transform(ref_transform)
        data = data.rio.write_crs(ref_crs)

        da_to_raster(
            data.squeeze(),
            tif_dir / f"{nc_file.stem}.tif",
            dtype=np.float32,
        )
        data.close()
        del data
        gc.collect()


def main(cfg: dict) -> None:
    """Convert netCDF files to GeoTIFFs."""
    ref_rast = open_rasterio(cfg["s2_20m"]["src"])

    data_dir = Path(data_root, cfg["stats"]["out_dir"], "20m")
    for radius_dir in data_dir.glob("*"):
        if radius_dir.is_dir():
            if NC_TO_GTIFF:
                nc_to_gtiffs(radius_dir, ref_rast)

            if ROTATE_TIFFS:
                rotate_tifs(radius_dir)


if __name__ == "__main__":
    main(config)
