import warnings
from abc import ABC, abstractmethod

class Simulation(ABC):
    """
    Abstract base class for portfolio simulations.
    
    This class defines the interface for all simulation implementations.
    """

    def __new__(cls, *args, **kwargs):
        if cls is Simulation:
            from .simulation_impl import SimulationImpl
            instance = super().__new__(SimulationImpl)
            return instance
        return super().__new__(cls)

    def __init__(self, nb_scenarios, model, strategy, parameters, rf=0.02):
        """
        Initialize a simulation.
        
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
        """
        self.nb_scenarios = nb_scenarios
        self.model = model
        self.strategy = strategy
        self.parameters = parameters
        self.rf = rf

    def set_nb_scenarios(self, nb_scenarios):
        """
        Set the number of scenarios
        
        Parameters
        ----------
        nb_scenarios : int
            Number of scenarios to simulate
        """
        self.nb_scenarios = nb_scenarios
    
    def set_model(self, model):
        """
        Set the model used for the simulation
        
        Parameters
        ----------
        model : MarketModel
            Market model to use for simulations
        """
        self.model = model
    
    def set_strategy(self, strategy):
        """
        Set the strategy used for the simulation
        
        Parameters
        ----------
        strategy : str
            Investment strategy ("Buy and hold" or "Rebalancing")
        """
        self.strategy = strategy

    def set_parameters(self, parameters):
        """
        Set the parameters of the simulation after checking their coherence with the strategy
        
        Parameters
        ----------
        parameters : dict
            Simulation parameters
        
        Raises
        ------
        ValueError
            If parameters are not coherent with the strategy
        """
        # Check coherence of the parameters and strategy
        if self.strategy == "Rebalancing":
            if "Rebalancing period" not in parameters:
                raise ValueError("Rebalancing strategy requires a rebalancing period in the parameters dictionary.\nExample : {'Rebalancing period' : 30}")
            elif parameters["Rebalancing period"] < 0:
                raise ValueError("Rebalancing period must be strictly positive")
        elif self.strategy == "Buy and hold":
            if "Rebalancing period" in parameters and parameters["Rebalancing period"] > 0:
                self.strategy = "Rebalancing"
                warnings.warn("Strategy changed to 'Rebalancing'. Positive rebalancing period is not needed for Buy and hold strategy.\n"
                "To avoid this warning, please use set_strategy() method to change the strategy before updating parameters.")
        self.parameters = parameters

    def set_dataESG(self, dataESG):
        """
        Set ESG data for each stock
        
        Parameters
        ----------
        dataESG : pd.DataFrame
            DataFrame representing the ESG data for each stock
            index : Stock names
            columns : ESG metrics (e.g., Sustainability risk, Carbon risk, etc.)
        """
        self.dataESG = dataESG

    def set_constraints(self, constraints):
        """
        Set constraints for the optimization problem
        
        Parameters
        ----------
        constraints : dict
            Dictionary containing the types and values of constraints
            {
                "List" : ["Maximal volatility", "Minimal return", "Sharpe", "Minimal durability", ...],
                "Value" : [0, 0.1, ...]
            }
        """
        self.constraints = constraints

    def set_allocation(self, allocation):
        """
        Set portfolio allocation
        
        Parameters
        ----------
        allocation : array-like
            Portfolio weights
        """
        self.parameters["Allocation"] = allocation
        
    def set_scenarios(self, scenarios):
        """
        Set scenarios of log-returns
        
        Parameters
        ----------
        scenarios : dict
            Dictionary of scenarios
        """
        self.scenarios = scenarios
        
    def set_evolutions(self, evolutions):
        """
        Set portfolio value evolutions
        
        Parameters
        ----------
        evolutions : dict
            Dictionary of portfolio value evolutions
        """
        self.evolutions = evolutions

    @abstractmethod
    def compute_allocation(self):
        """
        Calculate optimal portfolio allocation based on constraints and ESG data
        
        Returns
        -------
        array-like
            Optimal portfolio weights
        """
        pass

    @abstractmethod
    def generate_scenarios(self):
        """
        Generate scenarios of log-returns based on the model and the parameters
        
        Returns
        -------
        dict
            Dictionary of scenarios
        """
        pass
    
    @abstractmethod
    def generate_evolutions(self):
        """
        Generate the evolution of the portfolio value for each scenario
        
        Returns
        -------
        dict
            Dictionary of portfolio value evolutions
        """
        pass

    @abstractmethod
    def compute_metrics(self, alpha=0.95):
        """
        Compute the risk metrics of the simulation
        
        Parameters
        ----------
        alpha : float, optional
            Confidence level for VaR and ES, by default 0.95
            
        Returns
        -------
        dict
            Dictionary of risk metrics
        """
        pass

    @abstractmethod
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
        pass
