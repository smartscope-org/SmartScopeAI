This codebase is a wrapper around the smartscopeAI repository which adds a celery interface for SmartScope to call the algorithms. It can be installed anywhere for as long as both the SmartScope instance and the SmartScopeAI can connect to the same REDIS cache.

# Installation

Requirements:

- SmartScopeAI was tested with cuda 12.6. It is expected to work with other cuda 12.X but it has not been tested.

## Local installation with `uv` in a venv

First, clone the repository with its submodules:

`git clone --recurse-submodules git@github.com:JoQCcoz/SmartScopeAI.git`

`cd SmartScopeAI`

`uv sync`

## Docker

`git clone --recurse-submodules git@github.com:JoQCcoz/SmartScopeAI.git`

`cd SmartScopeAI`

`docker build -t SmartScopeAI:latest .`

# Downloading the AI models

Coming soon.

# Starting the celery worker

Two enviroment variables are needed to connect to the redis cache.

```
##Variables and their default values
REDIS_HOST = "localhost"
REDIS_PORT = "6379"
```

If installed locally, the TEMPLATE_FILES variable also needs to be set. This is the directory to the AI models.

The default values for TEMPLATE_FILES varies for docker and uv:

uv: TEMPLATE_FILES=./template_files
docker: TEMPLATE_FILES=/opt/template_files


## Starting with `uv`


