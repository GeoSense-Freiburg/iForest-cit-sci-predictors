iForest Prestudy: Citizen-science Predictor Data
==============================

Citizen science predictors (based on GBIF) matched to Sentinel-2 grids for the iForest prestudy. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    ├── pyproject.toml    <- Human-readable project dependencies managed with Poetry
    ├── poetry.lock       <- File used by Poetry to install dependencies
    └── .pre-commit-config.yaml <- pre-commit Git hooks


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Installation

### 1. Clone this project repository
```bash
git clone https://github.com/dluks/iForest-cit-sci-predictors.git

cd iForest-cit-sci-predictors
```

### 2. Install poetry and DVC.
Use `pipx` to install CLI requirements in isolated environments so to avoid dependency conflicts between the CLI tools and the project-specific requirements.

```bash
pipx install poetry==1.7
pipx install dvc==3.39
```

 >[!NOTE]
 > If you don't have root access, `condax` (a similar project) can be installed with:
 > ```bash
 > pip install --user condax
 > ```

### 3. Create a virtual environment
```bash
conda create -n if-cit-sci-feats -c conda-forge python=3.10
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
