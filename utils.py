import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def generate_evolution(logReturn, allocation, T_rebalancement=-1, initial_portfolio_value = 1.0, get_portfolio_value = False):
    """
    initially, sum(allocation) can be different of 1
    """
    
    nb_periods = logReturn.shape[0]
    nb_stocks = logReturn.shape[1]

    evolution = logReturn * 0.0
    
    if T_rebalancement == -1:
        # Buy and hold strategy
        evolution = np.exp(np.cumsum(logReturn)) * initial_portfolio_value * allocation
    else:
        # Rebalancing strategy
        evolution.iloc[:T_rebalancement, :] = np.exp(np.cumsum(logReturn.iloc[:T_rebalancement, :]))* initial_portfolio_value * allocation
        
        for i in range(T_rebalancement, nb_periods, T_rebalancement):
            evolution.iloc[i:i+T_rebalancement, :] = np.exp(np.cumsum(logReturn.iloc[i:i+T_rebalancement, :])) * evolution.iloc[i-1, :].sum() * allocation
        last_period = (nb_periods//T_rebalancement)*T_rebalancement
        evolution.iloc[last_period:, :] = np.exp(np.cumsum(logReturn.iloc[last_period:, :])) * evolution.iloc[last_period-1, :].sum() * allocation
        
    portfolio_value = evolution.iloc[-1,:].sum()
    if get_portfolio_value :
        return evolution, portfolio_value
    else :
        return evolution

def plot_evolutions_full(Dict_evolutions, model, constraints, strategy, parameters, alpha = 0.95, figsize=(14, 6)):
    
    plt.figure(figsize=figsize)
    # Plot individual scenarios in light blue
    list_terminal_values = []
    for evolution_name, evolution_data in Dict_evolutions.items():
        evolution_data.sum(axis=1).plot(color='lightBlue', alpha=0.3, linewidth=1, label='_nolegend_')
        list_terminal_values.append(evolution_data.sum(axis=1).iloc[-1])

    # Calculate and plot mean trajectory in red
    mean_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in Dict_evolutions.items()}).mean(axis=1)
    mean_evolution.plot(color='Red', linewidth=2, label='Mean trajectory')

    # Calculate and plot median trajectory in dark blue
    median_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in Dict_evolutions.items()}).median(axis=1)
    median_evolution.plot(color='blue', linewidth=2, label='Median trajectory')

    # Calculate and plot first quantile in black dotted line
    VAR_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in Dict_evolutions.items()}).quantile(1-alpha, axis=1)
    VAR_evolution.plot(color='Black', linestyle=':', linewidth=2, label='First quantile')

    # Add terminal value annotations
    terminal_value = mean_evolution.iloc[-1]
    terminal_median = median_evolution.iloc[-1]
    terminal_VAR = VAR_evolution.iloc[-1]

    txt_model = f"Model: {model}"
    txt_constraints = "Constraints:"
    for i, constraint in enumerate(constraints["List"]):
        txt_constraints += f"\n{constraint} : {constraints['Value'][i]}"
    # If Rebalancing period doesn't exist in parameters, set it to -1 (Buy and Hold)
    Rebalancing_period = parameters.get("Rebalancing period", -1)
    txt_strategie = f"Strategie: {f'{strategy} (T={Rebalancing_period})' if Rebalancing_period != -1 else 'Buy and Hold'}"

    plt.annotate(txt_model + '\n\n' + txt_strategie + '\n\n' + txt_constraints, 
                xy=(mean_evolution.index[-1], terminal_value + 0.3),
                xytext=(10, 0), textcoords='offset points',
                bbox=dict(facecolor='White', edgecolor='Black', alpha=0.7))

    plt.annotate(f'Mean terminal value: {terminal_value:.3f}', 
                xy=(mean_evolution.index[-1], terminal_value),
                xytext=(10, 0), textcoords='offset points',
                bbox=dict(facecolor='Gray', edgecolor='red', alpha=0.7))

    plt.annotate(f'Median terminal value: {terminal_median:.3f}', 
                xy=(median_evolution.index[-1], terminal_median),
                xytext=(10, 0), textcoords='offset points',
                bbox=dict(facecolor='Gray', edgecolor='blue', alpha=0.7))

    plt.annotate(f'Value at risk {alpha*100}%: {terminal_VAR:.3f}', 
                xy=(mean_evolution.index[-1], terminal_VAR),
                xytext=(10, 0), textcoords='offset points',
                bbox=dict(facecolor='Gray', edgecolor='black', alpha=0.7))
    
    y_lim_min = min(0.9, np.percentile(list_terminal_values, 1)) #Quantile 0.01 or 0.9
    y_lim_max = np.percentile(list_terminal_values, 95) #Quantile 0.95
    plt.ylabel('Portfolio value')
    plt.xlabel('Date')
    plt.ylim(y_lim_min, y_lim_max)
    plt.title("Evolutions of portfolio")
    plt.grid()
    plt.legend()
    plt.show()


def calibrate_BS_model(data):
    # Compute the parameters of the Black-Scholes model

    # Calcul du vecteur de rendement moyen annuel
    mu_annuel = np.log((data.iloc[-1, :]/data.iloc[0,:])) / (data.index[-1] - data.index[0]).days * 365.25
    delta_année = data.index.to_series().diff().dt.days.iloc[1:] / 365.25
    deltat_r = pd.DataFrame(delta_année.values[:, None] * mu_annuel.values, columns=mu_annuel.index, index=delta_année.index)
    nb_jour_pan = data.shape[0] / (data.index[-1] - data.index[0]).days * 365.25
    Volatilite_annuel = np.sqrt((np.log((data/data.shift(1)).iloc[1:,:]) - deltat_r).var() * nb_jour_pan)
    mat_correlation = (np.log((data/data.shift(1)).iloc[1:,:]) - deltat_r).corr()

    return mu_annuel, Volatilite_annuel, mat_correlation

def generate_BS_scenarios(parameters, beginDate, endDate, number_of_scenarios):

    # Generate the scenarios for the Black-Scholes model
    returns = parameters["Returns"]
    volatilities = parameters["Volatilities"]
    Correlation_matrix = parameters["Correlation matrix"]
    nb_stocks = len(returns)
    # generate dates excluding Saturdays and Sundays
    dates = pd.date_range(start=beginDate, end=endDate, freq='B')
    nb_periods = len(dates)
    # Calculate the adjustment factor for volatilities
    delta_t = dates.to_series().diff().dt.days[1:] / 365.25
    delta_t = np.insert(delta_t, 0, 0)
    adjustment_factor = np.sqrt(delta_t)
    # Generate scenarios of log-returns
    log_returns = np.random.normal(0, 1, (nb_periods, nb_stocks, number_of_scenarios))
    cholesky_matrix = np.linalg.cholesky(Correlation_matrix)
    for i in range(number_of_scenarios):
        log_returns[:, :, i] = log_returns[:, :, i] @ cholesky_matrix.T
    log_returns = log_returns * (volatilities.values[:, None] * adjustment_factor[:, None, None])
    # Add the annual average return
    mean_returns = (returns.values[:, None] / 256.2305133079848) * delta_t[:, None, None]
    log_returns += mean_returns
    scenarios = {f'Scenario {i+1}': pd.DataFrame(log_returns[:, :, i], index=dates, columns=volatilities.index) for i in range(number_of_scenarios)}
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
