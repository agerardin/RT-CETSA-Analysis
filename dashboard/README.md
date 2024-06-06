# RT_CETSA Dashboard (v0.3.0-dev0)

Web-based dashboard that manages a full RT_CETSA pipeline.

## Setup

The dashboard requires python >3.10.
It is recommended to create a dedicated environment for the project.

```conda create -n {env_name} python=3.10``` or ```python -m venv {env_name}```

## Install

Make sure you are in the dashboard directory.

`cd dashboard`

Install R:

```bash
sudo apt update
sudo apt install r-base
```

Install all other dependencies:

```pip install pyproject.toml```

or through [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

```poetry install```

## Run the dashboard


Local development :
```solara run src/rt_cetsa/dashboard.py```


To run it on notebookhub, first duplicate the `.sample-env` file as `.env` in the dashboard 
root folder, then define the variables that configure the proxy in `.env`

```solara run src/rt_cetsa/dashboard.py --env-file .env```

or 

```SOLARA_APP=$SOLARA_APP uvicorn --workers 1 --root-path $SOLARA_SERVICE_PREFIX --host 0.0.0.0 --port 8765 solara.server.starlette:app```

## Data

All the data created will be stored in a `data` folder in the current working directory.