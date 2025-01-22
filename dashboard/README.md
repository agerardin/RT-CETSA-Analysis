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

R and its dependencies can be installed beforehand this way :

`conda env create -f environment.yml`

Unfortunately, it seems that `r-mess` and `r-hrbrthemes` are not available
on conda-forge (at least for osx-arm64).

### Data

A specific `data` folder can be provided where all uploaded data and generated output will be stored.

```DATA_DIR=path/to/data solara run src/rt_cetsa/dashboard.py```

