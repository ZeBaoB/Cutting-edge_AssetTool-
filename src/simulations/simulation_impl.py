import numpy as np
import pandas as pd
import cvxpy as cp
from .simulation import Simulation
from src.models.market_model import MarketModel
from src.utils.visualization import plot_evolutions_full
from src.utils.portfolio import generate_evolution

class SimulationImpl(Simulation):
    """
    Implementation of the Simulation abstract class.
    
    This class provides concrete implementations of the abstract methods
    defined in the Simulation class.
    """

    def __init__(self, nb_scenarios, model, strategy, parameters, rf=0.02):
        """
        Initialize a simulation implementation.
        
        Parameters
        ----------
        nb_scenarios : int
            Number of scenarios to simulate
        model : MarketModel
            Market model to use for simulations
        strategy : str
            Investment strategy ("Buy and hold" or "Rebalancing")
        parameters : dict
            Simulation parameters including:
            - Begin date : str (format : "YYYY-MM-DD")
            - End date : str (format : "YYYY-MM-DD")
            - Rebalancing period : int (only for "Rebalancing" strategy)
        rf : float, optional
            Risk-free rate, by default 0.02
        
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        if not isinstance(nb_scenarios, int):
            raise ValueError("nb_scenarios must be an integer")
        if not isinstance(model, MarketModel):
            raise ValueError("model must be an instance of MarketModel")
        if strategy not in ["Buy and hold", "Rebalancing"]:
            raise ValueError("strategy must be either 'Buy and hold' or 'Rebalancing'")
            
        super().__init__(nb_scenarios, model, strategy, parameters, rf)

    def compute_allocation(self):
        """
        Calculate optimal portfolio allocation based on constraints and ESG data
        
        Returns
        -------
        array-like
            Optimal portfolio weights
        """
        if self.model.model_name == "BS":
            # Define objective function based on criteria
            returns = self.model.parameters["Returns"]
            volatilities = self.model.parameters["Volatilities"]
            correlation_matrix = self.model.parameters["Correlation matrix"]
            cov_matrix = np.diag(volatilities) @ correlation_matrix @ np.diag(volatilities)
            rf = self.rf
            dataESG = self.dataESG

            # Number of assets in portfolio
            nb_stocks = correlation_matrix.shape[0]
            
            # Define portfolio weights as optimization variables
            w = cp.Variable(nb_stocks)
            
            # Define optimization Markowitz problem
            portfolio_return = cp.sum(w @ returns)
            portfolio_variance = cp.quad_form(w, cov_matrix)
            constraints = [cp.sum(w) == 1, w >= 0]
            
            # Add constraints based on user specifications
            if "Maximal allocation" in self.constraints["List"]:
                constraints.append(w <= self.constraints["Value"][self.constraints["List"].index("Maximal allocation")])
                
            if "Minimal return" in self.constraints["List"]:
                constraints.append(portfolio_return >= self.constraints["Value"][self.constraints["List"].index("Minimal return")])
                objective = cp.Minimize(portfolio_variance)
                
            if "Maximal volatility" in self.constraints["List"]:
                constraints.append(portfolio_variance <= (self.constraints["Value"][self.constraints["List"].index("Maximal volatility")]) ** 2)
                objective = cp.Maximize(portfolio_return)
            else:
                objective = cp.Maximize(portfolio_return)

            # Define optimization ESG problem
            for i, criteria in enumerate(self.dataESG.columns):
                if "Maximal " + criteria in self.constraints["List"]:
                    pos_criteria = self.constraints["List"].index("Maximal " + criteria)
                    constraints.append(cp.sum(cp.multiply(w, dataESG[criteria])) <= self.constraints["Value"][pos_criteria])
                elif "Minimal " + criteria in self.constraints["List"]:
                    pos_criteria = self.constraints["List"].index("Minimal " + criteria)
                    constraints.append(cp.sum(cp.multiply(w, dataESG[criteria])) >= self.constraints["Value"][pos_criteria])

            # Define optimization problem
            prob = cp.Problem(objective, constraints)
            
            # Solve optimization problem
            prob.solve()
            
            # Get optimal portfolio weights
            self.parameters["Allocation"] = w.value
            
            return w.value

    def generate_scenarios(self):
        """
        Generate scenarios of log-returns based on the model and the parameters
        
        Returns
        -------
        dict
            Dictionary of scenarios
        """
        
        self.scenarios, self.scenarios_var = self.model.generate_logreturns(self.parameters["Begin date"], self.parameters["End date"], self.nb_scenarios)

    def generate_evolutions(self, T_allocation=0):
        """
        Generate the evolution of the portfolio value for each scenario
        
        Returns
        -------
        dict
            Dictionary of portfolio value evolutions
        """
        # Get rebalancing period (default to -1 for Buy and Hold)
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
                    # Recompute allocation at the start of the intervals, BS model approach
                    if start != 0:
                        model_used = MarketModel(model_name="BS")
                        model_used.fit(np.exp(scenario.iloc[start-T_allocation:end-T_allocation, :].cumsum(axis=0)))
                        self.set_model_allocation(model_used)
                        self.compute_allocation()
                        current_allocation = self.parameters["Allocation"]
                    # Generate evolution for the interval
                    new_evol, porfolio_value = generate_evolution(scenario.iloc[start:end, :], current_allocation, T_rebalancement = T_rebalancement, initial_portfolio_value = porfolio_value, get_portfolio_value = True)
                    evolution.append(new_evol)
                self.evolutions[f'Evolution {i+1}'] = pd.concat(evolution, axis=0)

    def compute_metrics(self, alpha_var=0.99, alpha_ES=0.975):
        """
        Compute the risk metrics of the simulation
        
        Parameters
        ----------
        alpha_var : float, optional
            Confidence level for VaR, by default 0.95
        alpha_ES : float, optional
            Confidence level for ES, by default 0.95
            
        Returns
        -------
        dict
            Dictionary of risk metrics
        """
        evolutions = self.evolutions
        rf = self.rf
        
        # Extract terminal values for all scenarios
        terminal_values = [evolution.iloc[-1].sum() for evolution in evolutions.values()]
        
        # Calculate risk metrics
        risk_metrics = {}
        risk_metrics["Mean terminal value"] = round(float(np.mean(terminal_values)), 4)
        risk_metrics["Median terminal value"] = round(float(np.median(terminal_values)), 4)
        risk_metrics["Volatility of terminal value"] = round(float(np.std(terminal_values)), 4)
        risk_metrics[f"VaR({alpha_var*100}%)"] = round(float(np.quantile(terminal_values, 1-alpha_var)), 4)
        
        # Calculate Expected Shortfall (ES)
        var_threshold = float(np.quantile(terminal_values, 1-alpha_ES))
        below_var_values = [value for value in terminal_values if value < var_threshold]
        if below_var_values:
            risk_metrics[f"ES({alpha_ES*100}%)"] = round(float(np.mean(below_var_values)), 4)
        else:
            risk_metrics[f"ES({alpha_ES*100}%)"] = var_threshold

        self.metrics = risk_metrics
        return risk_metrics

    def plot(self, type_plot="full", alpha=0.95, figsize=(14, 6)):
        """
        Plot the evolution of the portfolio value for each scenario
        
        Parameters
        ----------
        type_plot : str, optional
            Type of plot ("full" or "summary"), by default "full"
        alpha : float, optional
            Confidence level for VaR, by default 0.95
        figsize : tuple, optional
            Figure size, by default (14, 6)
        """
        if type_plot not in ["full", "summary"]:
            raise ValueError("type_plot must be either 'full' or 'summary'")
            
        if type_plot == "full":
            plot_evolutions_full(
                self.evolutions, 
                self.model.model_name, 
                self.constraints, 
                self.strategy, 
                self.parameters, 
                alpha=alpha, 
                figsize=figsize
            )
