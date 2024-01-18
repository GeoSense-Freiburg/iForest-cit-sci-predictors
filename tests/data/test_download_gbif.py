"""Test iforest_cit_sci_feats.data.download_gbif"""
from pathlib import Path

import pytest
from pygbif import occurrences as occ

from iforest_cit_sci_feats.data.download_gbif import (
    GbifDownloadFailure,
    check_download_job_and_download_file,
    check_download_status,
    download_request_to_disk,
    init_gbif_download,
    set_download_path,
)

# Define the module name to be used for mocking
TESTED_MODULE = "iforest_cit_sci_feats.data.download_gbif"


@pytest.fixture(name="gbif_query")
def fixture_gbif_query() -> dict:
    """Sample GBIF query"""
    return {
        "type": "and",
        "predicates": [
            {
                "type": "equals",
                "key": "TAXON_KEY",
                "value": "2653316",  # Ephedra equisetina, only one record
            },
            {"type": "not", "key": "DEGREE_OF_ESTABLISHMENT", "value": "cultivated"},
        ],
    }


@pytest.fixture(name="gbif_download_info")
def fixture_gbif_download_info() -> tuple:
    """Sample GBIF download info"""
    return (
        "0000000-000000000000000",
        {
            "creator": "lusk",
            "notification_address": ["daniel.lusk@geosense.uni-freiburg.de"],
            "sendNotification": True,
            "predicate": {
                "type": "and",
                "predicates": [
                    {"type": "equals", "key": "TAXON_KEY", "value": "2653316"},
                    {
                        "type": "not",
                        "predicate": {
                            "type": "in",
                            "key": "DEGREE_OF_ESTABLISHMENT",
                            "values": ["cultivated"],
                        },
                    },
                ],
            },
            "format": "SIMPLE_CSV",
        },
    )


@pytest.fixture(name="gbif_download_meta_running")
def fixture_gbif_download_meta_running() -> dict:
    """Sample GBIF download meta in a RUNNING state."""
    return {
        "key": "0000000-000000000000000",
        "doi": "10.15468/dl.4fm52d",
        "license": "http://creativecommons.org/licenses/by-nc/4.0/legalcode",
        "request": {
            "predicate": {
                "type": "and",
                "predicates": [
                    {
                        "type": "equals",
                        "key": "TAXON_KEY",
                        "value": "7707728",
                        "matchCase": False,
                    },
                    {
                        "type": "not",
                        "predicate": {
                            "type": "in",
                            "key": "DEGREE_OF_ESTABLISHMENT",
                            "values": ["cultivated"],
                            "matchCase": False,
                        },
                    },
                ],
            },
            "sendNotification": True,
            "format": "SIMPLE_CSV",
            "type": "OCCURRENCE",
            "verbatimExtensions": [],
        },
        "created": "2024-01-17T14:18:26.685+00:00",
        "modified": "2024-01-17T14:31:12.445+00:00",
        "eraseAfter": "2024-07-17T14:18:26.641+00:00",
        "status": "RUNNING",
        "downloadLink": "https://api.gbif.org/v1/occurrence/download/request/"
        "0000000-000000000000000.zip",
        "size": 50765,
        "totalRecords": 1,
        "numberDatasets": 1,
    }


@pytest.fixture(name="gbif_download_meta_success")
def fixture_gbif_download_meta_success(gbif_download_meta_running) -> dict:
    """Sample GBIF download meta in a SUCCEEDED state."""
    return {**gbif_download_meta_running, "status": "SUCCEEDED"}


@pytest.fixture(name="gbif_download_meta_failed")
def fixture_gbif_download_meta_failed(gbif_download_meta_running) -> dict:
    """Sample GBIF download meta in a FAILED state."""
    return {**gbif_download_meta_running, "status": "FAILED"}


@pytest.fixture(name="mock_download_meta")
def fixture_mock_download_meta(monkeypatch, gbif_download_meta_success: dict):
    """Mock pygbif.occ.download_meta to return a successful download meta."""

    def _mock_download_meta(*args, **kwargs):
        return gbif_download_meta_success

    monkeypatch.setattr(occ, "download_meta", _mock_download_meta)


@pytest.fixture(name="mock_download_get")
def fixture_mock_download_get(monkeypatch):
    """Mock pygbif.occ.download_get to write a file to output_path."""

    def _mock_download_get(*args, **kwargs):
        with open(args[1], "wb") as f:
            f.write(b"test")

    monkeypatch.setattr(occ, "download_get", _mock_download_get)


def test_init_gbif_download(monkeypatch, gbif_query, gbif_download_info):
    """Test init_gbif_download"""

    def mock_download(*args, **kwargs):
        return gbif_download_info

    monkeypatch.setattr(occ, "download", mock_download)

    download_key = init_gbif_download(gbif_query)

    assert download_key == "0000000-000000000000000"


def test_check_download_status(
    monkeypatch,
    gbif_download_info,
    gbif_download_meta_running,
    gbif_download_meta_success,
    gbif_download_meta_failed,
):
    """Test check_download_status"""

    def mock_download_meta_running(*args, **kwargs):
        return gbif_download_meta_running

    def mock_download_meta_succeeded(*args, **kwargs):
        return gbif_download_meta_success

    def mock_download_meta_failed(*args, **kwargs):
        return gbif_download_meta_failed

    monkeypatch.setattr(occ, "download_meta", mock_download_meta_running)
    assert check_download_status(gbif_download_info[0]) == "RUNNING"

    monkeypatch.setattr(occ, "download_meta", mock_download_meta_succeeded)
    assert check_download_status(gbif_download_info[0]) == "SUCCEEDED"

    monkeypatch.setattr(occ, "download_meta", mock_download_meta_failed)
    assert check_download_status(gbif_download_info[0]) == "FAILED"


def test_set_download_path(tmp_path, gbif_download_info):
    """Test set_download_path"""
    dl_key = gbif_download_info[0]

    # Test with directory
    assert set_download_path(tmp_path, dl_key) == tmp_path / f"{dl_key}.zip"

    # Test with nonexistent directory
    with pytest.raises(FileNotFoundError):
        set_download_path(Path("definitely/not/a/dir"), dl_key)

    # Test with file w/ nonexistent directory
    with pytest.raises(FileNotFoundError):
        set_download_path(Path("definitely/not/a/dir") / "test.zip", dl_key)

    # Test w/o .zip
    with pytest.raises(ValueError):
        set_download_path(tmp_path / "test", dl_key)

    # Test when file exists but overwrite is false
    with pytest.raises(FileExistsError):
        with open(tmp_path / f"{dl_key}.zip", "w", encoding="utf-8") as f:
            f.write("test")
        set_download_path(tmp_path / f"{dl_key}.zip", dl_key)

    # Test when file exists and overwrite is true
    with open(tmp_path / f"{dl_key}.zip", "w", encoding="utf-8") as f:
        f.write("test")
    assert set_download_path(tmp_path / f"{dl_key}.zip", dl_key, overwrite=True) == (
        tmp_path / f"{dl_key}.zip"
    )


def mock_download_get(
    _key: str, output_path: Path, *args, **kwargs  # pylint: disable=unused-argument
) -> None:
    """Mock pygbif.occurences.download_get to write a throwaway file."""
    with open(output_path, "wb") as f:
        f.write(b"test")


def test_download_request_to_disk(
    tmp_path, mocker, gbif_download_info, gbif_download_meta_success
):
    """Test download_request_to_disk"""
    output_path = tmp_path / "tmp.zip"
    mocker.patch(f"{TESTED_MODULE}.occ.download_get", side_effect=mock_download_get)
    mocker.patch(
        f"{TESTED_MODULE}.occ.download_meta", return_value=gbif_download_meta_success
    )
    download_request_to_disk(gbif_download_info[0], output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.with_suffix(".json").exists()
    assert output_path.with_suffix(".json").stat().st_size > 0


def test_check_download_job_and_download_file(tmp_path, mocker, gbif_download_info):
    """Test check_download_job_and_download_file"""
    output_path = tmp_path / "tmp.zip"
    mocker.patch(f"{TESTED_MODULE}.check_download_status", return_value="FAILED")
    with pytest.raises(GbifDownloadFailure) as excinfo:
        check_download_job_and_download_file(gbif_download_info[0], output_path)

    assert "failed." in str(excinfo.value)
