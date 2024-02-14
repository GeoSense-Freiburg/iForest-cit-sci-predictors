iForest Prestudy: Citizen-science Predictor Data
==============================

Citizen science predictors (based on GBIF) matched to Sentinel-2 grids for the iForest prestudy. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile                     <- Makefile with commands like `make data` or `make
    │                                   train`
    ├── README.md                    <- The top-level README for developers using this
    │                                   project.
    ├── data
    │   ├── external                 <- Data from third party sources.
    │   ├── interim                  <- Intermediate data that has been transformed.
    │   ├── processed                <- The final, canonical data sets for modeling.
    │   └── raw                      <- The original, immutable data dump.
    │       └── sentinel-2           <- Sentinel-2 grids at 10m and 20m resolutions
    │                                   (UTM 32N).
    │
    ├── notebooks                    <- Jupyter notebooks. Naming convention is a number
    │                                   (for ordering), the creator's initials, and a
    │                                   short `-` delimited description, e.g.
    │                                   `1-0-jqp-initial-data-exploration`.
    │
    ├── references                   <- Data dictionaries, manuals, and all other
    │                                   explanatory materials.
    │
    ├── reports                      <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                  <- Generated graphics and figures to be used in reporting
    │
    ├── setup.py                     <- makes project pip installable (pip install -e .)
    │                                   so src can be imported
    ├── src                          <- Source code for use in this project.
    │   ├── __init__.py              <- Makes src a Python module
    │   │
    │   ├── data                     <- Scripts to download or generate data
    │   │
    │   ├── features                 <- Scripts to turn raw or interim data into features for modeling
    │   │
    │   └── visualization            <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    ├── tests                        <- Unit tests for use with `pytest`
    │
    ├── pyproject.toml               <- Human-readable project dependencies managed with
    │                                   Poetry
    ├── poetry.lock                  <- File used by Poetry to install dependencies
    ├── conda-linux-64.lock          <- File used by conda-lock to install dependencies for 64-bit Linux systems
    ├── environment.yml              <- File used by conda-lock to specify dependencies
    ├── dvc.yml                      <- DVC pipeline definitions
    ├── params.yml                   <- DVC parameter definitions. IMPORTANT: this project
    │                                   also uses this file as a config file (see src.conf.parse_params)
    ├── dvc.lock                     <- DVC file which tracks changes to data tracked by DVC
    └── .pre-commit-config.yaml      <- pre-commit Git hooks


--------

## Installation

### 1. Clone this project repository
```bash
git clone https://github.com/dluks/iForest-cit-sci-predictors.git

cd iForest-cit-sci-predictors
```

### 2. Install poetry, conda-lock, and DVC.
Use `pipx` to install CLI requirements in isolated environments so to avoid dependency conflicts between the CLI tools and the project-specific requirements.

```bash
pipx install poetry==1.7
pipx install dvc==3.39
pipx install conda-lock==2.5
```

 >[!NOTE]
 > If you don't have root access, `condax` (a similar project) can be installed with:
 > ```bash
 > pip install --user condax
 > ```

### 3. Create a virtual environment
```bash
conda create --name if-cit-sci-feats --file conda-linux-64.lock
conda activate if-cit-sci-feats
```

### 4. Install requirements
```bash
poetry install
```

### 5. Configure pre-commit Git hooks (optional)
```bash
pre-commit install
```
> [!NOTE]
> If you'd prefer not to use `pre-commit` to manage Git hooks simply remove `pre-commit` with:
> ```bash
> pre-commit uninstall  # if you already installed pre-commit hooks
> poetry remove pre-commit
> ```

## Usage

### Raw data 
To get the required raw data, please contact daniel.lusk@geosense.uni-freiburg.de.

### Reproduce full pipeline
This project utilizes DVC to manage data and to ensure reproducibility. If you have the raw data files used for this project, simply run `dvc repro` to reproduce the final products. DVC will automatically check if any stages have already been run and will skip them if so.
```bash
dvc repro
```

### Reproduce specific stages of pipeline
In some cases, only the outputs of a single stage may be desired. This is useful for debugging or to execute the pipeline in piecemeal fashion.
```bash
dvc repro [stage you'd like to reproduce (see dvc.yml)]
```