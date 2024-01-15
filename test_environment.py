"""Check to make sure the development environment is set up correctly."""
import os
import shutil
import sys

REQUIRED_PYTHON = "python3"


def check_python():
    """Check that the correct version of Python is running."""
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError(f"Unrecognized python interpreter: {REQUIRED_PYTHON}")

    if system_major != required_major:
        raise TypeError(
            f"This project requires Python {required_major}. Found: Python {sys.version}"
        )
    print(">>> Python3 is installed. \u2713")


def check_env():
    """Check that a conda/mamba/micromamba virtual environment is activated."""
    active_env = os.getenv("CONDA_DEFAULT_ENV")

    if active_env is None:
        raise RuntimeError("Could not find an active conda/mamba environment!")

    print(f">>> Environment {active_env} is activated. \u2713")


def check_poetry():
    """Check that poetry is installed."""
    if not shutil.which("poetry"):
        raise RuntimeError(
            "Poetry is not installed. To install use\npipx install poetry==~1.7"
        )

    print(">>> Poetry is installed. \u2713")


def main():
    """Main script"""
    check_python()
    check_env()
    check_poetry()


if __name__ == "__main__":
    main()
