# Density Prediction API Documentation

## Overview

This project provides a `FastAPI`-based service for density prediction using a `Transformer` model. Accurate [density](https://en.wikipedia.org/wiki/Density_of_air) estimations of the [Thermosphere](https://en.wikipedia.org/wiki/Thermosphere) are essential for all spacecraft operations in low earth orbit. Density estimation is a part of the [Space Weather](https://en.wikipedia.org/wiki/Space_weather) prediction process.

## Project Structure

```bash
root/  # Repository root
│
├── data  # Datasets and evaluation results. Not in Repo!
│   ├──  original
│   └── reduced
├── docs  # Documentation. Not in Repo!
├── experiments  # Jupyter notebooks with experiments
└── libs/density_prediction  # Package root
    ├── app  # FastAPI application
    │   ├── config  # Configuration files
    │   │   └── settings.py  # Application settings
    │   ├── main.py  # Entry point for the FastAPI application
    │   └── models  # Model definitions
    ├── logs  # Log files
    ├── scripts  # Scripts for building and running the Docker container
    │   ├── start_server.sh  # Script to start the FastAPI server
    │   └── stop_server.sh  # Script to stop the FastAPI server
    ├── tests  # Unit tests
    ├── poetry.lock  # Poetry lock file
    ├── pyproject.toml  # Poetry configuration file
    └── README.md  # Project README file
```
## Main Parts and Classes

### Dependencies

Dependencies are managed using `Poetry`, as defined in the `pyproject.toml` file. This includes both runtime and development dependencies.  

### FastAPI Application (`main.py`)

The `main.py` file is the entry point for the FastAPI application. It typically includes the following components:  
* **API Endpoints**: Define the routes and their corresponding request handlers.
* **Middleware**: Add middleware for logging, authentication, etc.
* **Event Handlers**: Define startup and shutdown events.

### Configuration (`config/settings.py`)

This file contains configuration settings for the application, such as environment variables, database connections, and other settings.  

### Models (`models`)

This directory contains the model class definitions used in the application. These models can be Pydantic models for request/response validation or ORM models for database interactions.

### Class Relations 

* `Request Handling`: The FastAPI application receives a request at an endpoint, such as `/predict`.
* `Validation`: The request data is validated using Pydantic.
* `Processing`: The validated data is processed to predict the density.
* `Response`: The result is returned as a response to the client.

### Logging

Logs are stored in the `libs/density_prediction/logs` directory. Logging is essential for monitoring and debugging the application.

### Scripts

- `start_server.sh`: Starts the FastAPI server.
- `stop_server.sh`: Stops the FastAPI server.

### REST API definitions

When the `FastAPI` application is running, you can access the `API documentation` at `http://localhost:8000/docs`. 
This page provides an interactive interface for testing the `API endpoints` (`Swagger` interface).

### Tests

The `tests` directory contains unit tests for the application. 
These tests ensure that the different components of the application work 
as expected.

To run the tests, use the following command:
```bash
poetry run pytest
```
### Summary

* **FastAPI**: Main framework for building the API.
* **Pydantic**: Used for data validation.
* **Poetry**: Dependency management.
* **Scripts**: For managing server operations.
* **Logging**: For monitoring and debugging.

This structure ensures a modular and maintainable codebase, with clear separation of concerns and easy management of dependencies.