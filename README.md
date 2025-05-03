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

## Data

> 📈 **Historical Data**:  
> The project uses historical CAC40 stock prices to simulate portfolio dynamics. These data files are stored in the `Data/` folder and used during model calibration and backtesting.

> 🛠️ **Adaptability**:  
> You can replace or extend the dataset by modifying the contents of the `Data/` folder.  
> The code is modular and will work with any stock universe (e.g., S&P500, EuroStoxx50), as long as the formatting is respected (e.g., CSV format with consistent datetime index and price columns).

> 🌿 **ESG Data**:  
> You may include additional ESG metrics (e.g., carbon intensity) by placing them in the same folder and referencing them in the `dataESG` parameter.

## References

- Markowitz, H. (1952). *Portfolio Selection*, The Journal of Finance.
- Markowitz, H. (1959). *Portfolio Selection: Efficient Diversification of Investments*, John Wiley & Sons.
- Dichtl, H. (2020). *Investing in the S&P 500 index: Can anything beat the buy and hold strategy?*, Review of Financial Economics, 38(2), 352–378.
- Roncalli, T., Guenedal, T. L., Lepetit, F., & Sekine, T. (2020). *Measuring and managing carbon risk in investment portfolios*, arXiv:2008.13198.
