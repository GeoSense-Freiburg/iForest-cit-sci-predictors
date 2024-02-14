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

## Project description

The goal of this project is to support a prestudy by developing citizen-science-based species observation statistcs as features for use in training Sentinel-2-based species prediction models.

The pipeline takes [all vascular plant observations](https://doi.org/10.15468/dl.ybmj5x) as obtained from the GBIF database, subsets them by the extent of Sentinel-2 grids which in turn cover the extent of Germany, and further subsets the observations by masking out all observations in non-forested areas. Forested areas are determined from the [Copernicus Forest Type 2018 dataset](https://land.copernicus.eu/en/products/high-resolution-layer-forest-type/forest-type-2018#general_info). Next, species counts, as well as species counts weighted by distance, are produced for each grid cell in the reference Sentinel-2 raster at a series of different query radii.

The features are produced as 2-layer raster images that match the CRS, extent, and transform of the reference Sentinel-2 grid for each species:
- Each species' statistics are saved as a netcdf file which contains a 2-layer raster with the layers `counts` and `wt_counts`.
- To conserve space, the `wt_counts` layer is "packed" into `int16` format. For end users of the data, be sure to apply the scale and offset to get the actual weighted counts.
- Each species is indicated by a species ID, mapped in `species_ids.parquet`.


```shell
                                +-------------------------+                  
                                | data/raw/sentinel-2.dvc |                  
                               *+-------------------------+                  
                         ******           *               ******             
                   ******                 *                     *****        
                ***                      *                           *****   
+-------------------+             +-----------+                           ***
| setup_forest_mask |             | clip_gbif |                             *
+-------------------+             +-----------+                             *
                 ***            ***                                         *
                    **        **                                            *
                      **    **                                              *
                   +-----------+                                            *
                   | mask_gbif |                                            *
                   +-----------+                                            *
                          *                                                 *
                          *                                                 *
                          *                                                 *
            +-------------------------+                                   ***
            | save_species_ids_points |                              *****   
            +-------------------------+                         *****        
                                  ****                    ******             
                                      ***            *****                   
                                         **       ***                        
                                +-----------------------+                    
                                | compute_all_radii_20m |                    
                                +-----------------------+    
```
An example DAG for the pipeline described above, in which final species statistics are computed at 20 m resolution for all query radii. This can be easily recreated with `dvc dag`.

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