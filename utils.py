import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_evolution(logReturn, allocation, T_rebalancement=-1):
    nb_periods = logReturn.shape[0]
    nb_stocks = logReturn.shape[1]

    evolution = logReturn * 0.0
    
    if T_rebalancement == -1:
        # Buy and hold strategy
        evolution = np.exp(np.cumsum(logReturn)) * allocation
    else:
        # Rebalancing strategy
        evolution.iloc[:T_rebalancement, :] = np.exp(np.cumsum(logReturn.iloc[:T_rebalancement, :])) * allocation
        
        for i in range(T_rebalancement, nb_periods, T_rebalancement):
            evolution.iloc[i:i+T_rebalancement, :] = np.exp(np.cumsum(logReturn.iloc[i:i+T_rebalancement, :])) * evolution.iloc[i-1, :].sum() * allocation
        last_period = (nb_periods//T_rebalancement)*T_rebalancement
        evolution.iloc[last_period:, :] = np.exp(np.cumsum(logReturn.iloc[last_period:, :])) * evolution.iloc[last_period-1, :].sum() * allocation
    
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