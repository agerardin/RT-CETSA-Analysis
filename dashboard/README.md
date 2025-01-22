# RT_CETSA Dashboard (v0.5.0-dev0)

Web-based dashboard that manages a full RT_CETSA pipeline.

## Quickstart

### Setup

The dashboard requires python >3.10.
It is recommended to create a dedicated environment for the project.

```conda create -n {env_name} python=3.10``` or ```python -m venv {env_name}```

### Install

Install all dependencies:

```pip install pyproject.toml```

or through [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

```poetry install```

### Run the dashboard

```solara run src/rt_cetsa/dashboard.py```

==========

## Documentation

### Prerequisites


R needs to be available on the system in order to run the statistical analysis.

For osx, some R packages (`r-mess` and `r-hrbrthemes`) are not available on osx, so packages will need to be installed  directly with R.

On linux we can install all dependencies with conda.
On dump of the last working version of all dependencies has been created in `env-linux-dump.yml`
The environmnent can be recreated with this command :

`conda env create -f environment.yml`


### Data

A specific `data` folder can be provided where all uploaded data and generated output will be stored. Other solara options can be passed by environment variables.

```DATA_DIR=path/to/data HOST=host PORT=port solara run src/rt_cetsa/dashboard.py```

