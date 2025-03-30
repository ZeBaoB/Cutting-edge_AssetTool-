import numpy as np
import pandas as pd
import cvxpy as cp
from Classes.simulation_interface import Simulation
from utils import generate_evolution, plot_evolutions_full

from Classes.marketmodel_interface import MarketModel

class SimulationImpl(Simulation):

    def __init__(self, nb_scenarios, model, strategy, parameters, rf = 0.02):
        """
        nb_scenarios : int
        model : MarketModel
        strategy : str ("Buy and hold" or "Rebalancing")
        parameters : dict
            - Begin date : str (format : "YYYY-MM-DD")
            - End date : str (format : "YYYY-MM-DD")
            - Rebalancing period : int (only for "Rebalancing" strategy)
        rf : float (risk-free rate)
        """
        if not isinstance(nb_scenarios, int):
            raise ValueError("nb_scenarios must be an integer")
        if not isinstance(model, MarketModel):
            raise ValueError("model must be an instance of MarketModel")
        if strategy not in ["Buy and hold", "Rebalancing"]:
            raise ValueError("strategie must be either 'Buy and hold' or 'Rebalancing")
        self.nb_scenarios = nb_scenarios
        self.model = model
        self.strategy = strategy
        self.parameters = parameters
        self.rf = rf

    def compute_allocation(self):
        if self.model.model_name == "BS":
            # Define objective function based on criteria
            returns = self.model.parameters["Returns"]
            volatilities = self.model.parameters["Volatilities"]
            Correlation_matrix = self.model.parameters["Correlation matrix"]
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
            
            if "Maximal allocation" in self.contraints["List"]:
                constraints.append( w <= self.contraints["Value"][self.contraints["List"].index("Maximal allocation")])
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
        self.scenarios = self.model.generate_logreturns(self.parameters["Begin date"], self.parameters["End date"], self.nb_scenarios)

    def generate_evolutions(self, T_allocation = 0):
        nb_periods = self.scenarios["Scenario 1"].shape[1]
        nb_stocks = self.scenarios["Scenario 1"].shape[0]
        T_rebalancement = self.parameters["Rebalancing period"] if self.strategy == "Rebalancing" else -1
        allocation = self.parameters["Allocation"]
        if T_allocation <= 0:
            self.evolutions = {f'Evolution {i+1}' : generate_evolution(self.scenarios[f"Scenario {i+1}"], allocation, T_rebalancement) for i in range(self.nb_scenarios)}
        else:
            self.evolutions = {}
            for i in range(self.nb_scenarios):
                porfolio_value = 1.0
                scenario = self.scenarios[f"Scenario {i+1}"]
                evolution = []
                current_allocation = allocation.copy()
                intervals = range(0, scenario.shape[0], T_allocation)
                for start in intervals:
                    end = min(start + T_allocation, scenario.shape[0])
                    # Recompute allocation at the start of the intervals
                    if start != 0:
                        model_used = self.model
                        model_used.fit(np.exp(scenario.iloc[start-T_allocation:end-T_allocation, :].cumsum(axis=0)))
                        self.set_model(model_used)
                        self.compute_allocation()
                        current_allocation = self.parameters["Allocation"]
                    # Generate evolution for the interval
                    new_evol, porfolio_value = generate_evolution(scenario.iloc[start:end, :], current_allocation, T_rebalancement = T_rebalancement, initial_portfolio_value = porfolio_value, get_portfolio_value = True)
                    evolution.append(new_evol)
                self.evolutions[f'Evolution {i+1}'] = pd.concat(evolution, axis=0)

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
            plot_evolutions_full(self.evolutions, self.model.model_name, self.contraints, self.strategy, self.parameters, alpha=alpha, figsize=figsize)

# Attach methods to Simulation class
#Simulation.__init__ = init
#Simulation.compute_allocation = compute_allocation
#Simulation.generate_scenarios = generate_scenarios
#Simulation.generate_evolutions = generate_evolutions
#Simulation.compute_metrics = compute_metrics
#Simulation.plot = plot