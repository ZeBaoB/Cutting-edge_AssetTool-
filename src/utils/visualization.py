import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_evolutions_full(dict_evolutions, model, constraints, strategy, parameters, alpha=0.95, figsize=(14, 6)):
    """
    Plot the evolution of portfolio value for multiple scenarios.
    
    Parameters
    ----------
    dict_evolutions : dict
        Dictionary of portfolio value evolutions
    model : str
        Model name
    constraints : dict
        Dictionary of constraints
    strategy : str
        Investment strategy
    parameters : dict
        Simulation parameters
    alpha : float, optional
        Confidence level for VaR, by default 0.95
    figsize : tuple, optional
        Figure size, by default (14, 6)
    """
    plt.figure(figsize=figsize)
    
    # Plot individual scenarios in light blue
    first = True
    for evolution_name, evolution_data in dict_evolutions.items():
        if first:
            evolution_data.sum(axis=1).plot(color='lightBlue', alpha=0.2, linewidth=1, label="Simulated scenarios")
            val_y_max = evolution_data.sum(axis=1).iloc[0] * 3
            first = False
        else:
            evolution_data.sum(axis=1).plot(color='lightBlue', alpha=0.2, linewidth=1, label='_nolegend_')

    # Calculate and plot mean trajectory in red
    mean_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in dict_evolutions.items()}).mean(axis=1)
    mean_evolution.plot(color='Red', linewidth=2, label='Mean trajectory')

    # Calculate and plot median trajectory in dark blue
    median_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in dict_evolutions.items()}).median(axis=1)
    median_evolution.plot(color='blue', linewidth=2, label='Median trajectory')

    # Calculate and plot first quantile in black dotted line
    var_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in dict_evolutions.items()}).quantile(1-alpha, axis=1)
    var_evolution.plot(color='Black', linestyle=':', linewidth=2, label=f'VaR ({alpha*100}%)')

    # Add terminal value annotations
    terminal_value = mean_evolution.iloc[-1]
    terminal_median = median_evolution.iloc[-1]
    terminal_var = var_evolution.iloc[-1]

    # Create text for model and constraints
    txt_model = f"Model: {model}"
    txt_constraints = "Constraints:"
    for i, constraint in enumerate(constraints["List"]):
        txt_constraints += f"\n{constraint}: {constraints['Value'][i]}"
        
    # Get rebalancing period (default to -1 for Buy and Hold)
    rebalancing_period = parameters.get("Rebalancing period", -1)
    txt_strategy = f"Strategy: {strategy} (T={rebalancing_period})" if rebalancing_period != -1 else "Strategy: Buy and Hold"

    # Add annotations to the plot
    plt.annotate(txt_model + '\n\n' + txt_strategy + '\n\n' + txt_constraints, 
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

    plt.annotate(f'Value at risk {alpha*100}%: {terminal_var:.3f}', 
                xy=(mean_evolution.index[-1], terminal_var),
                xytext=(10, 0), textcoords='offset points',
                bbox=dict(facecolor='Gray', edgecolor='black', alpha=0.7))

    plt.ylabel('Portfolio value')
    plt.xlabel('Date')
    plt.ylim(0, val_y_max)
    plt.title("Evolution of portfolio value")
    plt.grid()
    plt.legend()
    plt.show()

def plot_returns_distribution(returns, model_params=None, bins=30, figsize=(12, 6)):
    """
    Plot the distribution of returns with fitted model.
    
    Parameters
    ----------
    returns : array-like
        Array of returns
    model_params : tuple, optional
        Model parameters (mu, sigma, lambda_, mu_J, sigma_J), by default None
    bins : int, optional
        Number of bins for histogram, by default 30
    figsize : tuple, optional
        Figure size, by default (12, 6)
    """
    from scipy.stats import norm
    
    plt.figure(figsize=figsize)
    
    # Plot histogram of returns
    n, bins, patches = plt.hist(returns, bins=bins, density=True, alpha=0.6, color='b', label='Historical returns')
    
    # If model parameters are provided, plot fitted distribution
    if model_params is not None:
        mu, sigma, lambda_, mu_J, sigma_J = model_params
        dt = 1/252  # Daily time step
        
        # Generate x values for plotting
        x = np.linspace(min(returns), max(returns), 1000)
        
        # Calculate PDF for Merton model
        pdf = (1 - lambda_ * dt) * norm.pdf(x, mu * dt, sigma * np.sqrt(dt)) + \
              (lambda_ * dt) * norm.pdf(x, mu * dt + mu_J, np.sqrt(sigma**2 * dt + sigma_J**2))
        
        plt.plot(x, pdf, 'r-', linewidth=2, label='Fitted model')
        
        # Add model parameters to plot
        plt.text(0.02, 0.95, f'μ = {mu:.4f}\nσ = {sigma:.4f}\nλ = {lambda_:.4f}\nμ_J = {mu_J:.4f}\nσ_J = {sigma_J:.4f}',
                transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.title('Distribution of Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_comparison(evolutions_dict, labels=None, alpha=0.95, figsize=(16, 8)):
    """
    Plot a comparison of different portfolio strategies.
    
    Parameters
    ----------
    evolutions_dict : dict
        Dictionary of dictionaries of portfolio value evolutions
        {strategy_name: {scenario_name: evolution_data}}
    labels : dict, optional
        Dictionary of labels for each strategy, by default None
    alpha : float, optional
        Confidence level for VaR, by default 0.95
    figsize : tuple, optional
        Figure size, by default (16, 8)
    """
    plt.figure(figsize=figsize)
    
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot mean trajectories for each strategy
    for i, (strategy_name, evolutions) in enumerate(evolutions_dict.items()):
        color = colors[i % len(colors)]
        
        # Calculate mean trajectory
        mean_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in evolutions.items()}).mean(axis=1)
        
        # Plot mean trajectory
        label = labels[strategy_name] if labels and strategy_name in labels else strategy_name
        mean_evolution.plot(color=color, linewidth=2, label=f'Mean - {label}')
        
        # Calculate and plot VaR
        var_evolution = pd.DataFrame({name: data.sum(axis=1) for name, data in evolutions.items()}).quantile(1-alpha, axis=1)
        var_evolution.plot(color=color, linestyle=':', linewidth=1, label=f'VaR ({alpha*100}%) - {label}')
    
    plt.ylabel('Portfolio value')
    plt.xlabel('Date')
    plt.title("Comparison of portfolio strategies")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
