# Nixtla Intro Workshop

This repository contains hands-on Jupyter notebooks that demonstrate the basics of time series forecasting using Nixtla's ecosystem, including [StatsForecast](https://nixtlaverse.nixtla.io/statsforecast/), [mlforecast](https://nixtlaverse.nixtla.io/mlforecast/), [NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/), and [TimeGPT](https://www.nixtla.io/docs/introduction/introduction).

## Contents

- [**Introduction_to_Nixtlaverse.ipynb**](https://colab.research.google.com/github/bettercodepaul/nixtla_intro_workshop/blob/main/Introduction_to_Nixtlaverse.ipynb)
  Introduction to classic time series forecasting using StatsForecast. Covers data exploration, decomposition, classic models, and rolling cross-validation.

- [**Introduction_to_MLForecast.ipynb**](https://colab.research.google.com/github/bettercodepaul/nixtla_intro_workshop/blob/main/Introduction_to_MLForecast.ipynb)
  Machine learning approaches for time series forecasting using mlforecast. Includes feature engineering, local vs global models, and validation strategies.

- [**Introduction_to_NeuralForecast_TimeGPT.ipynb**](https://colab.research.google.com/github/bettercodepaul/nixtla_intro_workshop/blob/main/Introduction_to_NeuralForecast_TimeGPT.ipynb)
  Forecasting with NeuralForecast and TimeGPT, including neural network models and foundation models for time series.

- **utilities.py**  
  Utility functions for time series decomposition and plotting, used by the notebooks.

- **retail_sales.parquet**  
  Example dataset: Monthly sales data for various countries.

- **retail_sales_product_level.parquet**  
  Example dataset: Monthly sales data at the product level.

## Installation

This repo uses [uv](https://docs.astral.sh/uv/). You can run `uv sync` to install the environment.