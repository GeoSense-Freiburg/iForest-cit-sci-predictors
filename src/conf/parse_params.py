"""Parses a DVC params.yaml file to return a config object that can be used in other scripts."""

import os

import yaml
from dotenv import find_dotenv, load_dotenv


def parse_params() -> dict:
    """
    Parse a DVC params.yaml file to return a config object that can be used in other scripts.
    Note that this requires a PROJECT_ROOT variable to be set as an environment variable
    in a local .env file.
    """
    load_dotenv(find_dotenv())
    with open(f"{os.environ['PROJECT_ROOT']}/params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f.read())


config = parse_params()

if __name__ == "__main__":
    print(config)
