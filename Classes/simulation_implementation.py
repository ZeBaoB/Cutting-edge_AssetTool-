import numpy as np
import pandas as pd
import cvxpy as cp
from Classes.simulation_interface import Simulation
from utils import generate_evolution, plot_evolutions_full

# Méthodes de la classe Simulation --------------------------------------------

def init(self, nb_scenarios, model, strategy, parameters, rf = 0.02):
    if not isinstance(nb_scenarios, int):
        raise ValueError("nb_scenarios must be an integer")
    if model not in ["BS", "On verra"]:
        raise ValueError("model must be either 'BS' or 'On verra'")
    if strategy not in ["Buy and hold", "Rebalancing"]:
        raise ValueError("strategie must be either 'Buy and hold' or 'Rebalancing")
    self.nb_scenarios = nb_scenarios
    self.model = model
    self.strategy = strategy
    self.parameters = parameters
    self.rf = rf

def compute_allocation(self):
    # Define objective function based on criteria
    returns = self.parameters["Returns"]
    volatilities = self.parameters["Volatilities"]
    Correlation_matrix = self.parameters["Correlation matrix"]
    covMatrix = np.diag(volatilities) @ Correlation_matrix @ np.diag(volatilities)
    rf = self.rf
    dataESG = self.dataESG

    # Number of assets in portfolio
    nb_stocks = Correlation_matrix.shape[0]
    # Define portfolio weights as optimization variables
    w = cp.Variable(nb_stocks)
    # Define optimization Markowitz problem
    portfolio_return = cp.sum(w@returns)
    portfolio_variance = cp.quad_form(w, covMatrix)
    constraints = [cp.sum(w) == 1, w >= 0]
    
    if "Minimal return" in self.contraints["List"]:
        constraints.append( portfolio_return >= self.contraints["Value"][self.contraints["List"].index("Minimal return")])
        objective = cp.Minimize(portfolio_variance)
    if "Maximal volatility" in self.contraints["List"]:
        constraints.append(portfolio_variance <= (self.contraints["Value"][self.contraints["List"].index("Maximal volatility")]) ** 2)
        objective = cp.Maximize(portfolio_return)
    else:
        objective = cp.Maximize(portfolio_return)

    # Define optimization ESG problem
    for i, criteria in enumerate(self.dataESG.columns):
        if "Maximal " + criteria in self.contraints["List"]:
            posCriteria = self.contraints["List"].index("Maximal " + criteria)
            constraints.append(cp.sum(cp.multiply(w, dataESG[criteria])) <= self.contraints["Value"][posCriteria])
        elif "Minimal " + criteria in self.contraints["List"]:
            posCriteria = self.contraints["List"].index("Minimal " + criteria)
            constraints.append(cp.sum(cp.multiply(w, dataESG[criteria])) >= self.contraints["Value"][posCriteria])

    # Define optimization problem
    prob = cp.Problem(objective, constraints)
    # Solve optimization problem
    prob.solve()
    # Get optimal portfolio weights
    self.parameters["Allocation"] = w.value

def generate_scenarios(self):
    nb_scenarios = self.nb_scenarios
    nb_stocks = len(self.parameters["Returns"])
    # Génération des dates excluant les samedis et dimanches
    dates = pd.date_range(start=self.parameters["Begin date"], end=self.parameters["End date"], freq='B')
    nb_periods = len(dates)
    # Calcul du facteur d'ajustement pour les volatilities
    delta_t = dates.to_series().diff().dt.days[1:] / 365.25
    delta_t = np.insert(delta_t, 0, 0)
    adjustment_factor = np.sqrt(delta_t)
    
    if self.model == "BS":
        volatilities = self.parameters["Volatilities"]
        returns = self.parameters["Returns"]
        # Génération de scénarios de log-rendements
        log_returns = np.random.normal(0, 1, (nb_periods, nb_stocks, nb_scenarios))
        cholesky_matrix = np.linalg.cholesky(self.parameters["Correlation matrix"])
        for i in range(self.nb_scenarios):
            log_returns[:, :, i] = log_returns[:, :, i] @ cholesky_matrix.T
        log_returns = log_returns * (volatilities.values[:, None] * adjustment_factor[:, None, None])
        # Ajout du rendement moyen annuel
        mean_returns = (returns.values[:, None] / 256.2305133079848) * delta_t[:, None, None]
        log_returns += mean_returns
        self.scenarios = {f'Scenario {i+1}': pd.DataFrame(log_returns[:, :, i], index=dates, columns=volatilities.index) for i in range(nb_scenarios)}


def generate_evolutions(self):
    nb_periods = self.scenarios["Scenario 1"].shape[1]
    nb_stocks = self.scenarios["Scenario 1"].shape[0]
    T_rebalancement = self.parameters["Rebalancing period"] if self.strategy == "Rebalancing" else -1
    allocation = self.parameters["Allocation"]
    self.evolutions = {f'Evolution {i+1}' : generate_evolution(self.scenarios[f"Scenario {i+1}"], allocation, T_rebalancement) for i in range(self.nb_scenarios)}

def compute_metrics(self, alpha=0.95):
    evolutions = self.evolutions
    rf = self.rf
    risk_metrics = {}
    risk_metrics["Mean terminal value"] = round(float(np.mean([evolution.iloc[-1].sum() for evolution in evolutions.values()])), 4)
    risk_metrics["Median terminal value"] = round(float(np.median([evolution.iloc[-1].sum() for evolution in evolutions.values()])), 4)
    risk_metrics["Volatility of terminal value"] = round(float(np.std([evolution.iloc[-1].sum() for evolution in evolutions.values()])), 4)
    risk_metrics[f"VaR({alpha*100}%)"] = round(float(np.quantile([evolution.iloc[-1].sum() for evolution in evolutions.values()], 1-alpha)), 4)
    risk_metrics[f"ES({alpha*100}%)"] = round(float(np.mean([value for value in [evolution.iloc[-1].sum() for evolution in evolutions.values()] if value < risk_metrics[f"VaR({alpha*100}%)"]])), 4)

    self.metrics = risk_metrics

def plot(self, type_plot="full", alpha=0.95, figsize=(14, 6)):
    if type_plot not in ["full", "summary"]:
        raise ValueError("type_plot must be either 'full' or 'summary'")
    if type_plot == "full":
        plot_evolutions_full(self.evolutions, self.model, self.contraints, self.strategy, self.parameters, alpha=alpha, figsize=figsize)


# Attach methods to Simulation class
Simulation.__init__ = init
Simulation.compute_allocation = compute_allocation
Simulation.generate_scenarios = generate_scenarios
Simulation.generate_evolutions = generate_evolutions
Simulation.compute_metrics = compute_metrics
Simulation.plot = plot