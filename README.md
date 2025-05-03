# Asset Management Tool – Portfolio Simulation and Strategy Comparison

This project provides a Python-based tool for simulating and evaluating investment strategies under various financial models and constraints.  
It is intended for asset managers—such as insurance companies—seeking to build portfolios that match long-term liabilities while accounting for risk and sustainability constraints.

## Features

- Stochastic modeling of asset prices (Black-Scholes, Heston, Jump Diffusion)
- Portfolio strategies:
  - Buy & Hold
  - Equally Weighted Rebalancing
  - Markowitz Optimization
- ESG constraints and allocation rules
- Backtesting using simulated or historical data
- Risk metrics: variance, quantile loss, etc.
- ESG integration (e.g. carbon risk)

## Technical Architecture

The solution is implemented in **Python**, and can be used in two ways:
- Via **Jupyter Notebooks**
- Via a **Streamlit app** (recommended for interactive use)

### Core Components

The tool is structured into two main classes:

#### 1. `Model`

Handles the financial model and data generation.

**Attributes:**
- `model_name`
- `parameters`

**Key Methods:**
- `fit()` – fits the model to historical stock price data
- `generate_logreturns()` – simulates new return paths based on the selected model

#### 2. `Simulation`

Orchestrates the simulation and evaluation of strategies.

**Attributes:**
- `nb_scenarios`
- `model`
- `strategy`
- `parameters`
- `dataESG`
- `constraints`

**Key Methods:**
- `compute_allocation()`
- `generate_scenarios()`
- `generate_evolutions()`
- `compute_metrics()`
- `plot()` – visualizes portfolio performance and metrics

## How to Use

1. **Clone the repository - You need the folders Data and src**

2. **Prepare your environment**
   - Install dependencies (suggested: use a virtualenv or conda environment)
   ```bash
   pip install -r requirements.txt

4. **Launch Streamlit**
   ```bash
   python -m streamlit run app.py
