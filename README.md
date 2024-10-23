# Density Prediction API

This project provides a `FastAPI`-based service for density prediction using a `Transformer` model.
Accurate [density](https://en.wikipedia.org/wiki/Density_of_air) estimations of the [Thermosphere](https://en.wikipedia.org/wiki/Thermosphere) are essential for all spacecraft operations in
low earth orbit. Density estimation is a part of the [Space Weather](https://en.wikipedia.org/wiki/Space_weather) prediction process.

## Project Structure

```bash
root/  # Project root
│
├── data  # Datasets and evaluation results. Not in Repo!
│   ├──  original
│   └── reduced
├── docs  # Documentation. . Not in Repo!
├── experiments  # Jupyter notebooks with experiments
└── libs/density_prediction  # Main application folder
 ```

## Requirements and Setup

See the [package README.md](libs/density_prediction/README.md).

## Experiments

- [Data Analysis](experiments/data_analysis.ipynb)
- [Modeling](experiments/modeling.ipynb)

