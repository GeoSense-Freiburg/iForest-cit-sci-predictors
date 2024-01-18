"""Initiate a GBIF download and save it to disk once it is ready."""
import json
import logging
import time
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from pygbif import occurrences as occ


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
    occ.download_get(key, str(output_path))

    # Also save the metadata alongside the data
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


def check_download_job_and_download_file(
    key: str, output_path: Path, max_hours: int | float = 6
) -> None:
    """
    Checks a pending GBIF download for a given amount of time, downloads the file once
    it is ready.

    Args:
        output_path (Path): The path where the downloaded file will be saved.
        key (str): The key of the GBIF download job to check.
        max_hours (int | float, optional): The maximum number of hours to wait for the download
            to complete. Defaults to 6.

    Raises:
        GbifDownloadFailure: If the download job fails or exceeds the maximum waiting time.

    """
    start_time = time.time()
    while True:
        status = check_download_status(key)
        if status == "SUCCEEDED":
            download_request_to_disk(
                key,
                output_path,
            )
            break

        if status == "FAILED":
            raise GbifDownloadFailure(f"Download job {key} failed.")

        if time.time() - start_time > max_hours * 60 * 60:
            output_path.unlink()
            raise GbifDownloadFailure(
                f"Download job {key} did not complete within" f"{max_hours}. Aborting"
            )
        time.sleep(60)


@click.command()
@click.option(
    "-q",
    "--query",
    "query_file",
    type=click.Path(path_type=Path),
    default=Path("references/gbif/all_tracheophyta.json"),
    show_default=True,
)
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
def main(query_file: Path, output_path: Path, overwrite: bool):
    """Check the status of a GBIF download job and download the CSV file once it's ready."""
    log = logging.getLogger(__name__)
    load_dotenv(find_dotenv())  # Find local .env to expose GBIF credentials

    if query_file.suffix != ".json":
        raise ValueError("Query file must be a .json file.")

    with open(query_file, "r", encoding="utf-8") as f:
        gbif_query = json.load(f)

    download_key = init_gbif_download(gbif_query)

    log.info("Checking if download job is ready...")
    check_download_job_and_download_file(
        download_key, set_download_path(output_path, download_key, overwrite)
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
