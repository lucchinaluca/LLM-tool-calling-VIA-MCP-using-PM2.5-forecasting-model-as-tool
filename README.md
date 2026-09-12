# Bath Air Quality and PM2.5 Forecasting Project

This repository contains a research and prototype project focused on forecasting air pollution in Bath, UK, using weather, traffic, and pollution data. The work combines exploratory notebooks, preprocessing and imputation pipelines, time-series forecasting models, and a local AI assistant that can query live weather, pollution, and PM2.5 forecasting tools via MCP servers.

## Project goal

The project aims to:

- analyse Bath air quality and pollution patterns
- combine environmental and transport-related signals
- test different imputation and forecasting approaches
- compare classical and deep-learning time-series models
- build a local decision-support assistant for weather and air-quality queries

## Key report

The latest added report is the viva preparation document in `mds/viva_prep.md`.

It summarises the full thesis narrative and key findings for the Bath PM2.5 forecasting project, including:

- the end-to-end data pipeline combining AQ, weather, and traffic sources
- the chronological evaluation setup using January 2025 as the adversarial test period
- model progress from baseline LSTM through Darts Transformer, hybrid Transformer-LSTM, and the optimised final model
- the main result: the best-performing model achieved an R² of 0.6234
- the negative result from the spatio-temporal transfer learning experiment, showing that the spatial prior worsened performance due to limited traffic coverage
- the live MCP deployment using Open-Meteo, OpenAQ, and the saved forecasting model for local advisory generation

This report is especially useful as a concise thesis narrative and as a source of viva-style questions and answers.

## Main components

### 1. Data collection and preparation
The project ingests data from multiple sources, including:

- Bath pollution data
- weather data
- traffic and ANPR data
- maps and sensor metadata

Relevant folders:

- `classified_datasets/` - cleaned and structured input datasets
- `online_data/` - live or online data snapshots
- `processed_data_pkl/` - processed pickle artifacts
- `maps/` - geographic sensor and map outputs

### 2. Forecast modelling notebooks
The notebooks progressively build the forecasting pipeline:

- `1.pollution_data.ipynb` - pollution dataset exploration
- `1.traffic_data.ipynb` - traffic dataset exploration
- `1.weather_data.ipynb` - weather dataset exploration
- `2.graph_imputation.ipynb` - graph-based missing-data handling
- `3.knn_imputation.ipynb` - KNN imputation experiments
- `4.1new_time_series.ipynb` - time-series baseline work
- `4.2darts.ipynb` - Darts library benchmarking
- `4.3darts_optimization.ipynb` - model tuning and optimisation
- `4.4transformer_lstm.ipynb` - Transformer-LSTM forecasting model
- `4.5transformer_lstm_optimization.ipynb` - optimized Transformer-LSTM workflow
- `4.6multi_variate.ipynb` - multivariate forecasting experiments
- `5.spatio_temporal_OptionB.ipynb` - spatio-temporal model experimentation
- `6.permutation_importance_timeseries.ipynb` - feature importance analysis
- `model_comparison_same_days.ipynb` - model comparison for same-day forecasting

### 3. Local MCP forecasting tools
The project includes live data endpoints exposed through the Model Context Protocol (MCP):

- `weather_mcp.py` - returns Bath weather + next 12-hour forecast from Open-Meteo
- `pollution_mcp.py` - returns latest Bath pollution values from OpenAQ
- `model_mcp.py` - runs PM2.5 forecasting using a saved Transformer-LSTM model

These tools are combined in:

- `client.py` - a command-line client that connects to the MCP servers and invokes them through an OpenAI-compatible model interface

This allows a local LLM assistant to answer questions such as:

- What is the current Bath weather?
- What is the current air quality?
- What is the PM2.5 forecast for the next 24 hours?

## Repository structure

```text
.
├── README.md
├── client.py
├── model_mcp.py
├── pollution_mcp.py
├── weather_mcp.py
├── 1.pollution_data.ipynb
├── 1.traffic_data.ipynb
├── 1.weather_data.ipynb
├── 2.graph_imputation.ipynb
├── 3.knn_imputation.ipynb
├── 4.1new_time_series.ipynb
├── 4.2darts.ipynb
├── 4.3darts_optimization.ipynb
├── 4.4transformer_lstm.ipynb
├── 4.5transformer_lstm_optimization.ipynb
├── 4.6multi_variate.ipynb
├── 5.spatio_temporal_OptionB.ipynb
├── 6.permutation_importance_timeseries.ipynb
├── model_comparison_same_days.ipynb
├── classified_datasets/
├── online_data/
├── models_saved/
├── processed_data_pkl/
└── .venv/
```

## Data sources

This project relies on multiple external and local data sources:

- Open-Meteo for weather and air-quality data
- OpenAQ for pollutant measurements
- Bath traffic and ANPR datasets
- local preprocessed datasets in the project folders

## Environment and dependencies

The project is configured for Python and uses libraries such as:

- pandas
- numpy
- scikit-learn
- PyTorch
- requests
- requests-cache
- retry-requests
- openmeteo-requests
- mcp
- openai

A virtual environment is already present in `.venv`, but you can recreate it with:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows PowerShell
pip install pandas numpy scikit-learn torch openmeteo-requests requests-cache retry-requests mcp openai
```

## Running the project

### 1. Start a local LLM backend
If you want to use the AI assistant client, make sure Ollama is installed and running:

```bash
ollama serve
ollama pull qwen2.5:7b
```

### 2. Run the MCP servers
The client expects the weather, pollution, and PM2.5 forecast servers to be available:

```bash
python weather_mcp.py
python pollution_mcp.py
python model_mcp.py
```

These are started through the client automatically in the normal flow, but they can also be launched individually for testing.

### 3. Start the interactive client
```bash
python client.py
```

Then enter a prompt such as:

- What is the current weather in Bath?
- What is the PM2.5 level in Bath?
- Should I go for a run tomorrow based on air quality?

The assistant will call the relevant live tools before answering.

## Common workflow

1. Explore raw data in the notebook series.
2. Clean and preprocess datasets.
3. Test imputation strategies for missing values.
4. Train and compare forecasting models.
5. Save best-performing models to `models_saved/`.
6. Use live MCP endpoints to query real-time conditions and forecast outputs.

## Notes

- Several notebooks are experiment-oriented and are intended for research and model development rather than a single production pipeline.
- The live forecasting model is designed for Bath, UK, and depends on the trained artifact in `models_saved/`.
- The project is a hybrid of data science experimentation and a local AI tooling prototype.

## Suggested next steps

- formalize a single reproducible training pipeline
- add a `requirements.txt` file for one-command setup
- document the exact model training parameters and evaluation metrics
- package the forecasting pipeline into a reusable Python module
- add a clean front-end or notebook for running comparisons and reporting results

## Summary

This project sits at the intersection of environmental data analysis, machine learning, and AI tooling. It provides both a research codebase for Bath air-quality prediction and a live local assistant that can answer weather and pollution questions using real-time tool calls.
