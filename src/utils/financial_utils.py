import numpy as np
import pandas as pd
from scipy.optimize import minimize

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

def calibrate_heston_model(data : pd.DataFrame, window : int = 15,  params_init : list = [3., 0.15], bounds : list = [(1e-3, 40.), (1e-3, 1.)]):
    """
    Calibrate the Heston model parameters using the historical data of the stocks.
    The function returns the parameters of the model and the correlation matrix of the Brownian motions.
    
    Parameters:
    - data : pd.DataFrame containing the historical prices
        columns : stock names
        lines : dates
    - window : int, the window size for the rolling variance calculation

    Returns:
    - df_params : pd.DataFrame containing the parameters of the model
        index : parameters names ('V_end', 'S_end', 'mu', 'kappa', 'theta', 'sigma')
        columns : stock names
    - dB_dW correlation matrix : pd.DataFrame containing the correlation matrix of the Brownian motions ((dB_i)+(dW_j) == > 2*nb_stocks)
        names : stock names + '_dB'(for price movement) and stock names + '_dW'(for variance movement)
    """

    log_return_daily = np.log(data / data.shift(1)) / (data.index.to_series().diff().dt.total_seconds().values.reshape(-1, 1)) *3600*24
    log_return_daily.dropna(inplace=True)
    realized_var = log_return_daily.rolling(window).var()
    realized_var.dropna(inplace=True)
    log_return_daily = log_return_daily.loc[realized_var.index]
    mu = log_return_daily.mean(axis=0) * 365.25
    realized_var_annualized = realized_var * 365.25
    d_realized_var_annualized = realized_var_annualized.diff().dropna()
    df_params = pd.DataFrame(columns=d_realized_var_annualized.columns, index=['V_end', 'S_end', 'mu', 'kappa', 'theta', 'sigma'])
    df_params.index.name = 'parameters'
    df_params.columns.name = 'companies'
    df_params.loc['mu'] = mu
    df_params.loc['V_end'] = realized_var_annualized.iloc[-1].values
    df_params.loc['S_end'] = data.iloc[-1].values

    def heston_log_likelihood(params : list, variance : np.ndarray) -> float:
        kappa, theta = params
        dt = 1/365.25  # pas de temps (1 jour)
        d_variance = variance[1:] - variance[:-1]
        sigma = d_variance.std()*(365.25**0.5)
        # - log_likelihood
        objective_value = (d_variance - kappa * (theta - variance[:-1])*dt)**2 / sigma**2 / variance[:-1] / dt
        return np.sum(objective_value)

    for column in df_params.columns:
        var_col = realized_var_annualized[column].values
        # Optimize the parameters using a minimization algorithm
        resultat = minimize(heston_log_likelihood, params_init, args=(var_col,), bounds=bounds, method='L-BFGS-B')
        df_params.loc['kappa', column] = resultat.x[0]
        df_params.loc['theta', column] = resultat.x[1]
        d_var_col = var_col[1:] - var_col[:-1]
        sigma = d_var_col.std()*(365.25**0.5)
        df_params.loc['sigma', column] = sigma
        if resultat.x[0] < 0.01 or resultat.x[1] < 0.01:
            print(f"Warning for {column} with params {resultat.x}. The result value is {resultat.fun}. and the sigma is {sigma}.")
            continue
    
    dB = (log_return_daily-(mu - realized_var_annualized/2)/365.25) / realized_var_annualized**0.5 * (365.25**0.5)
    dB.columns = [f'{col}_dB' for col in dB.columns]
    variance = realized_var_annualized.values
    dW = (d_realized_var_annualized - df_params.loc['kappa'].values * (df_params.loc['theta'].values * np.ones(d_realized_var_annualized.shape) - variance[:-1]) * (1/365.25)) / df_params.loc['sigma'].values / variance[:-1]**0.5 / (1/365.25)**0.5
    dW.columns = [f'{col}_dW' for col in dW.columns]
    dB_dW = pd.concat([dB, dW], axis=1)
    dB_dW.dropna(inplace=True)

    return df_params, dB_dW.corr()

def generate_Heston_scenarios(df_params : pd.DataFrame, dBdW : pd.DataFrame, beginDate : str, endDate : str, number_of_scenarios : int, V_init : float = None) -> dict:

    """
    Generate Heston-Monte Carlo scenarios.
    The function returns the log-returns and the variance paths for each scenario.

    Parameters:
    - df_params : pd.DataFrame containing the parameters of the model
        index : parameters names ('V_end', 'S_end', 'mu', 'kappa', 'theta', 'sigma')
        columns : stock names
    - dBdW : pd.DataFrame containing the correlation matrix of the Brownian motions ((dB_i)+(dW_j) == > 2*nb_stocks)
        names : stock names + '_dB'(for price movement) and stock names + '_dW'(for variance movement)
    - beginDate : str, the start date of the simulation (format 'YYYY-MM-DD')
    - endDate : str, the end date of the simulation (format 'YYYY-MM-DD')
    - number_of_scenarios : int, the number of scenarios to generate
    - V_init : float, the initial value of the variance (optional, default is None, which means the last value of the variance in df_params)

    Returns:
    - log_return_paths : dict, the log-return paths for each scenario
        keys : scenario names ('Scenario 1', 'Scenario 2', ...)
        values : pd.DataFrame containing the log-returns
            index : dates
            columns : stock names
    - variance_paths : dict, the variance paths for each scenario
        keys : scenario names ('Scenario 1', 'Scenario 2', ...)
        values : pd.DataFrame containing the variances
            index : dates
            columns : stock names
    """
    # Get the parameters
    mu = df_params.loc['mu']
    kappa = df_params.loc['kappa']
    theta = df_params.loc['theta']
    sigma = df_params.loc['sigma']
    nb_stocks = len(mu)
    if V_init is None:
        V_init = df_params.loc['V_end']
    # generate dates excluding Saturdays and Sundays
    dates = pd.date_range(start=beginDate, end=endDate, freq='B')
    nb_periods = len(dates)
    # Calculate the adjustment factor for volatilities
    delta_t = dates.to_series().diff().dt.days[1:] / 365.25
    delta_t = np.insert(delta_t, 0, 0)
    # Generate scenarios of dB and dW ==> 2*nb_stocks brownian motions per scenarios
    choelesky_matrix = np.linalg.cholesky(dBdW.values)
    lReturns_Variance = np.random.normal(0, 1, (nb_periods, nb_stocks*2, number_of_scenarios))
    for i in range(number_of_scenarios):
        lReturns_Variance[:, :, i] = np.dot(lReturns_Variance[:, :, i], choelesky_matrix)
    
    # Variance paths
    variance_paths = {}
    for i in range(number_of_scenarios):
        variance_path = np.zeros((nb_periods, nb_stocks))
        variance_path[0, :] = V_init.values
        for j in range(1, nb_periods):
            variance_path[j, :] = np.maximum(variance_path[j-1, :] + kappa.values * (theta.values - variance_path[j-1, :]) * delta_t[j] + sigma.values * np.sqrt(variance_path[j-1, :]) * lReturns_Variance[j, nb_stocks:, i], 0) * delta_t[j]**0.5
        variance_paths[f"Scenario {i+1}"] = pd.DataFrame(variance_path, index=dates, columns=df_params.columns)
    
    # Log return paths
    log_return_paths = {}
    for i in range(number_of_scenarios):
        log_return_path = np.zeros((nb_periods, nb_stocks))
        log_return_path[0, :] = 0.0
        for j in range(1, nb_periods):
            log_return_path[j, :] = (mu.values - variance_paths[f"Scenario {i+1}"].iloc[j-1, :] / 2) * delta_t[j] + np.sqrt(variance_paths[f"Scenario {i+1}"].iloc[j-1, :]) * lReturns_Variance[j, :nb_stocks, i] * delta_t[j]**0.5
        log_return_paths[f"Scenario {i+1}"] = pd.DataFrame(log_return_path, index=dates, columns=df_params.columns)

    
    return log_return_paths, variance_paths

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
