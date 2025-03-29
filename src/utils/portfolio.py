import numpy as np
import pandas as pd

def generate_evolution(log_returns, allocation, t_rebalancing=-1):
    """
    Generate the evolution of a portfolio value based on log returns and allocation.
    
    Parameters
    ----------
    log_returns : pd.DataFrame
        DataFrame of log returns
        index : dates
        columns : stock names
    allocation : array-like
        Initial portfolio weights
    t_rebalancing : int, optional
        Rebalancing period in days, by default -1 (Buy and Hold)
        
    Returns
    -------
    pd.DataFrame
        DataFrame of portfolio value evolution
        index : dates
        columns : stock names
    """
    nb_periods = log_returns.shape[0]
    nb_stocks = log_returns.shape[1]
    
    # Initialize evolution DataFrame with zeros
    evolution = log_returns * 0.0
    
    if t_rebalancing == -1:
        # Buy and hold strategy
        evolution = np.exp(np.cumsum(log_returns)) * allocation
    else:
        # Rebalancing strategy
        evolution.iloc[:t_rebalancing, :] = np.exp(np.cumsum(log_returns.iloc[:t_rebalancing, :])) * allocation
        
        for i in range(t_rebalancing, nb_periods, t_rebalancing):
            end_idx = min(i + t_rebalancing, nb_periods)
            evolution.iloc[i:end_idx, :] = np.exp(np.cumsum(log_returns.iloc[i:end_idx, :])) * evolution.iloc[i-1, :].sum() * allocation
            
        # Handle the last period if it's not a complete rebalancing period
        last_period = (nb_periods // t_rebalancing) * t_rebalancing
        if last_period < nb_periods:
            evolution.iloc[last_period:, :] = np.exp(np.cumsum(log_returns.iloc[last_period:, :])) * evolution.iloc[last_period-1, :].sum() * allocation
    
    return evolution

def calculate_portfolio_metrics(returns, weights, cov_matrix, rf=0.02):
    """
    Calculate portfolio metrics such as expected return, volatility, and Sharpe ratio.
    
    Parameters
    ----------
    returns : array-like
        Expected returns for each asset
    weights : array-like
        Portfolio weights
    cov_matrix : array-like
        Covariance matrix of returns
    rf : float, optional
        Risk-free rate, by default 0.02
        
    Returns
    -------
    dict
        Dictionary of portfolio metrics:
        - Expected return
        - Volatility
        - Sharpe ratio
    """
    # Expected portfolio return
    portfolio_return = np.sum(returns * weights)
    
    # Portfolio volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Sharpe ratio
    sharpe_ratio = (portfolio_return - rf) / portfolio_volatility
    
    return {
        "Expected return": portfolio_return,
        "Volatility": portfolio_volatility,
        "Sharpe ratio": sharpe_ratio
    }

def calculate_esg_metrics(weights, esg_data):
    """
    Calculate ESG metrics for a portfolio.
    
    Parameters
    ----------
    weights : array-like
        Portfolio weights
    esg_data : pd.DataFrame
        DataFrame of ESG data
        index : stock names
        columns : ESG metrics
        
    Returns
    -------
    dict
        Dictionary of ESG metrics for the portfolio
    """
    esg_metrics = {}
    
    for column in esg_data.columns:
        esg_metrics[column] = np.sum(weights * esg_data[column])
    
    return esg_metrics
