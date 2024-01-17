"""Initiate a GBIF download and save it to disk once it is ready."""
import json
import logging
import time
from pathlib import Path

import click
import requests
from dotenv import find_dotenv, load_dotenv
from pygbif import occurrences as occ
from tqdm import tqdm


class GbifDownloadFailure(Exception):
    """Exception raised when a GBIF download job fails."""

    def __init__(self, message="GBIF download job failed."):
        self.message = message
        super().__init__(self.message)


def init_gbif_download(query: dict) -> str:
    """
    Initiate a GBIF download job and return the job key.
    """
    download_key = occ.download(query)  # type: ignore
    return download_key[0]


def check_download_status(key: str) -> str:
    """Check the status of a GBIF download job."""
    status = occ.download_meta(key)
    return status["status"]


def download_request_to_disk(key: str, output_path: Path) -> None:
    """Download a completed GBIF download job's zipfile and metadata."""
    url = f"https://api.gbif.org/v1/occurrence/download/request/{key}.zip"

    response = requests.get(url, stream=True, timeout=30)
    total_size = int(response.headers.get("content-length", 0))  # size in bytes

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        try:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        progress_bar.update(len(chunk))
                        f.write(chunk)
        # delete partial file on timeout or ctrl-c exception
        except (requests.exceptions.Timeout, KeyboardInterrupt):
            if output_path.exists():
                output_path.unlink()
            raise

    if total_size not in (0, progress_bar.n):
        raise RuntimeError("Could not download file!")

    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(occ.download_meta(key), f)


def set_download_path(path: Path, key: str, overwrite: bool = False) -> Path:
    """
    Returns the output path for a given key.

    Args:
        path (Path): The base path where the output file will be saved. If a directory,
            the file will be named "path/{key}.csv". If the directory doesn't exist,
            it will be created.
        key (str): The key used to generate the output file name.
        overwrite (bool, optional): Whether to overwrite the file if it already exists.
            Defaults to False.

    Returns:
        Path: The output path for the given key.

    Raises:
        FileExistsError: If the file already exists and `overwrite` is set to False.
    """
    if path.is_dir():
        if not path.exists():
            raise FileNotFoundError(f"Directory {path.absolute()} does not exist.")
        return path / f"{key}.zip"

    if not path.parent.exists():
        raise FileNotFoundError(f"Directory {path.parent.absolute()} does not exist.")

    if path.suffix != ".zip":
        raise ValueError("Output path must be a directory or a .zip file")

    if path.exists() and not overwrite:
        raise FileExistsError(
            "File already exists. Set `overwrite=True` to overwrite it."
        )

    return path


@click.command()
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=Path().cwd(),
    show_default=True,
)
@click.option(
    "-w", "--overwrite", "overwrite", type=click.BOOL, default=False, show_default=True
)
def main(output_path: Path, overwrite: bool):
    """Check the status of a GBIF download job and download the CSV file once it's ready."""
    log = logging.getLogger(__name__)

    load_dotenv(find_dotenv())  # Find local .env to expose GBIF credentials

    tracheophyta_taxon_key = "7707728"

    all_tracheophyta = {
        "type": "and",
        "predicates": [
            {
                "type": "equals",
                "key": "TAXON_KEY",
                "value": tracheophyta_taxon_key,
            },
            {"type": "not", "key": "DEGREE_OF_ESTABLISHMENT", "value": "Cultivated"},
        ],
    }

    download_key = init_gbif_download(all_tracheophyta)

    log.info("Download key: %s", download_key)
    log.info("Checking if download job is ready...")
    while True:
        status = check_download_status(download_key)
        if status == "SUCCEEDED":
            output_path = set_download_path(output_path, download_key, overwrite)
            download_request_to_disk(
                download_key,
                output_path,
            )
            break
        if status == "FAILED":
            raise GbifDownloadFailure(f"Download job {download_key} failed.")

        time.sleep(60)  # wait for 60 seconds before checking the status again


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
