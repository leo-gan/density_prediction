# Density Prediction API

This project provides a `FastAPI`-based service for density prediction using a `Transformer` model.
Accurate [density](https://en.wikipedia.org/wiki/Density_of_air) estimations of the [Thermosphere](https://en.wikipedia.org/wiki/Thermosphere) are essential for all spacecraft operations in
low earth orbit. Density estimation is a part of the [Space Weather](https://en.wikipedia.org/wiki/Space_weather) prediction process.

## Project Structure

```bash
root/  # Project root
│
├── docs  # Documentation. . Not in Repo!
└── libs/density_prediction  # Main application folder
    ├── Dockerfile
    ├── poetry.lock
    ├── pyproject.toml
    ├── README.md  # This file
    ├──  app  # FastAPI application
    ├──  logs # Log files
    ├──  scripts  # Scripts for building and running the Docker container
    └──  tests  # Unit tests
```

## Requirements

- Python 3.10+
- Poetry 1.8+

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/leo-gan/density_prediction
    cd <repository_directory>
    ```

2. Install Poetry if you haven't already:
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3. Install the required packages:
    ```sh
    poetry install
    ```

4. Activate the virtual environment:
    ```sh
    poetry shell
    ```

## Configuration

Set the `MODEL_PATH` environment variable to point to your model file. You can also modify the default path in `libs/density_prediction/app/config/settings.py`.

## Running the Server

To start the FastAPI server, run:
```sh
bash libs/density_prediction/scripts/start_server.sh
```
To stop the FastAPI server, run:
```sh
bash libs/density_prediction/scripts/stop_server.sh
```