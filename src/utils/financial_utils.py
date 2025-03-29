import numpy as np
import pandas as pd

def calibrate_BS_model(data):
    """
    Calibrate the Black-Scholes model parameters based on historical price data.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing historical prices
        columns : stock names
        index : dates
        
    Returns
    -------
    tuple
        (mu_annual, volatility_annual, correlation_matrix)
        - mu_annual : pd.Series, annual returns for each stock
        - volatility_annual : pd.Series, annual volatilities for each stock
        - correlation_matrix : pd.DataFrame, correlation matrix of returns
    """
    # Compute annual average return
    mu_annual = np.log((data.iloc[-1, :]/data.iloc[0,:])) / (data.index[-1] - data.index[0]).days * 365.25
    
    # Compute daily returns adjusted for the expected return
    delta_year = data.index.to_series().diff().dt.days.iloc[1:] / 365.25
    deltat_r = pd.DataFrame(delta_year.values[:, None] * mu_annual.values, 
                           columns=mu_annual.index, index=delta_year.index)
    
    # Compute annual volatility
    nb_days_per_year = data.shape[0] / (data.index[-1] - data.index[0]).days * 365.25
    volatility_annual = np.sqrt((np.log((data/data.shift(1)).iloc[1:,:]) - deltat_r).var() * nb_days_per_year)
    
    # Compute correlation matrix
    correlation_matrix = (np.log((data/data.shift(1)).iloc[1:,:]) - deltat_r).corr()
    
    return mu_annual, volatility_annual, correlation_matrix

def generate_BS_scenarios(parameters, begin_date, end_date, number_of_scenarios):
    """
    Generate scenarios for the Black-Scholes model.
    
    Parameters
    ----------
    parameters : dict
        Dictionary containing model parameters:
        - Returns : pd.Series, annual returns for each stock
        - Volatilities : pd.Series, annual volatilities for each stock
        - Correlation matrix : pd.DataFrame, correlation matrix of returns
        - Stocks : list, stock names
    begin_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    number_of_scenarios : int
        Number of scenarios to generate
        
    Returns
    -------
    dict
        Dictionary of scenarios with keys 'Scenario 1', 'Scenario 2', etc.
    """
    returns = parameters["Returns"]
    volatilities = parameters["Volatilities"]
    correlation_matrix = parameters["Correlation matrix"]
    nb_stocks = len(returns)
    
    # Generate dates excluding weekends
    dates = pd.date_range(start=begin_date, end=end_date, freq='B')
    nb_periods = len(dates)
    
    # Calculate the adjustment factor for volatilities
    delta_t = dates.to_series().diff().dt.days[1:] / 365.25
    delta_t = np.insert(delta_t, 0, 0)
    adjustment_factor = np.sqrt(delta_t)
    
    # Generate scenarios of log-returns
    log_returns = np.random.normal(0, 1, (nb_periods, nb_stocks, number_of_scenarios))
    cholesky_matrix = np.linalg.cholesky(correlation_matrix)
    
    for i in range(number_of_scenarios):
        log_returns[:, :, i] = log_returns[:, :, i] @ cholesky_matrix.T
    
    log_returns = log_returns * (volatilities.values[:, None] * adjustment_factor[:, None, None])
    
    # Add the annual average return
    mean_returns = (returns.values[:, None] / 256.2305133079848) * delta_t[:, None, None]
    log_returns += mean_returns
    
    # Create dictionary of scenarios
    scenarios = {
        f'Scenario {i+1}': pd.DataFrame(
            log_returns[:, :, i], 
            index=dates, 
            columns=volatilities.index
        ) for i in range(number_of_scenarios)
    }
    
    return scenarios

def calibrate_merton_model(returns, dt=1/252):
    """
    Calibrate the Merton jump-diffusion model using maximum likelihood estimation.
    
    Parameters
    ----------
    returns : array-like
        Array of log returns
    dt : float, optional
        Time step in years, by default 1/252 (daily)
        
    Returns
    -------
    tuple
        (mu, sigma, lambda_, mu_J, sigma_J)
        - mu : float, drift parameter
        - sigma : float, volatility parameter
        - lambda_ : float, jump intensity
        - mu_J : float, mean jump size
        - sigma_J : float, jump size volatility
    """
    from scipy.optimize import minimize
    from scipy.stats import norm, poisson
    
    def log_likelihood(params, returns, dt):
        mu, sigma, lambda_, mu_J, sigma_J = params
        
        # Probability of a jump occurring
        N_t = poisson.pmf(1, lambda_ * dt)
        
        # Probability density of returns with and without jump
        normal_part = norm.logpdf(returns, loc=mu * dt, scale=sigma * np.sqrt(dt))
        jump_part = norm.logpdf(returns, loc=mu * dt + mu_J, scale=np.sqrt(sigma**2 * dt + sigma_J**2))
        
        # Combine probabilities (mixture of normal and jumped returns)
        log_likelihood = np.log((1 - N_t) * np.exp(normal_part) + N_t * np.exp(jump_part))
        
        return -np.sum(log_likelihood)  # Minimize negative log-likelihood
    
    # Initial parameter guess
    init_params = [0.1, 0.2, 4.4, -0.1, 0.2]
    
    # Calibrate using maximum likelihood
    result = minimize(
        log_likelihood, 
        init_params, 
        args=(returns, dt), 
        method='L-BFGS-B',
        bounds=[(-1, 1), (0.01, 1), (0, 50), (-1, 1), (0.01, 1)]
    )
    
    return result.x
