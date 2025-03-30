import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    first = True
    for evolution_name, evolution_data in Dict_evolutions.items():
        if first:
            evolution_data.sum(axis=1).plot(color='lightBlue', alpha=0.2, linewidth=1, label="Simulated scenarios")
            val_y_max = evolution_data.sum(axis=1).iloc[0] * 3
            first = False
        else:
            evolution_data.sum(axis=1).plot(color='lightBlue', alpha=0.2, linewidth=1, label='_nolegend_')

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

    plt.ylabel('Portfolio value')
    plt.xlabel('Date')
    plt.ylim(0, val_y_max)
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