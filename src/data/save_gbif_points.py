"""Save the species points to a csv file and the species ids to a json file."""

import logging

from src.conf.parse_params import config
from src.utils.df_utils import read_gdf, write_df
from src.utils.log_utils import setup_logger

setup_logger()
log = logging.getLogger(__name__)


def main(cfg: dict):
    """Save the species points to a csv file and the species ids to a json file."""
    if cfg["gbif"]["verbose"]:
        log.setLevel(logging.INFO)

    log.info("Reading data and assigning IDs...")
    gbif = (
        read_gdf(cfg["gbif"]["masked"])
        .sort_values("species", ignore_index=True)
        .assign(species_id=lambda df_: df_.species.astype("category").cat.codes)
    )

    log.info("Writing species IDs...")
    write_df(
        gbif[["species_id", "species"]].drop_duplicates(ignore_index=True),
        cfg["gbif"]["species_ids"],
    )

    log.info("Extracting point coordinates...")
    points = gbif.assign(
        x=lambda df_: df_.geometry.apply(lambda p: p.x),
        y=lambda df_: df_.geometry.apply(lambda p: p.y),
    ).drop(columns=["species", "geometry"])

    log.info("Writing points...")
    write_df(points, cfg["gbif"]["points"])

    log.info("Done. ✅")


if __name__ == "__main__":
    main(config)
