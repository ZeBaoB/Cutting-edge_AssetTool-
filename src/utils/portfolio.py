import numpy as np
import pandas as pd

def generate_evolution(logReturns, allocation, T_rebalancing=-1, initial_portfolio_value = 1.0, get_portfolio_value = False):
    """
    Generate the evolution of a portfolio value based on log returns and allocation.
    
    Parameters
    ----------
    logReturns : pd.DataFrame
        DataFrame of log returns
        index : dates
        columns : stock names
    allocation : array-like
        Initial portfolio weights
    T_rebalancing : int, optional
        Rebalancing period in days, by default -1 (Buy and Hold)
        
    Returns
    -------
    pd.DataFrame
        DataFrame of portfolio value evolution
        index : dates
        columns : stock names
    """
    nb_periods = logReturns.shape[0]
    nb_stocks = logReturns.shape[1]
    evolution = logReturns * 0.0
    if T_rebalancing == -1:
        # Buy and hold strategy
        evolution = np.exp(np.cumsum(logReturns)) * initial_portfolio_value * allocation
    else:
        # Rebalancing strategy
        evolution.iloc[:T_rebalancing, :] = np.exp(np.cumsum(logReturns.iloc[:T_rebalancing, :]))* initial_portfolio_value * allocation
        
        for i in range(T_rebalancing, nb_periods, T_rebalancing):
            evolution.iloc[i:i+T_rebalancing, :] = np.exp(np.cumsum(logReturns.iloc[i:i+T_rebalancing, :])) * evolution.iloc[i-1, :].sum() * allocation
        last_period = (nb_periods//T_rebalancing)*T_rebalancing
        evolution.iloc[last_period:, :] = np.exp(np.cumsum(logReturns.iloc[last_period:, :])) * evolution.iloc[last_period-1, :].sum() * allocation
    portfolio_value = evolution.iloc[-1,:].sum()
    if get_portfolio_value :
        return evolution, portfolio_value
    else :
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
