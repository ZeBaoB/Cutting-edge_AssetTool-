import warnings
from abc import ABC, abstractmethod

class Simulation(ABC):

    def __new__(cls, *args, **kwargs):
        if cls is Simulation:
            from .simulation_implementation import SimulationImpl
            instance = super().__new__(SimulationImpl)
            #instance.__init__(*args, **kwargs)
            return instance
        return super().__new__(cls)

    def set_nb_scenarios(self, nb_scenarios):
        """
        Set the number of scenarios
        """
        self.nb_scenarios = nb_scenarios
    
    def set_model(self, model):
        """
        Set the model used for the simulation
        """
        self.model = model
    
    def set_model_allocation(self, model_allocation):
        """
        Set the model used for the allocation
        """
        self.model_allocation = model_allocation
    
    def set_strategy(self, strategy):
        """
        Set the strategy used for the simulation
        """
        self.strategy = strategy

    def set_parameters(self, parameters):
        """
        Set the parameters of the simulation after checking their coherence with the strategy
        """
        #Check coherence of the parameters and strategy
        if self.strategy == "Rebalancing":
            if "Rebalancing period" not in parameters:
                raise ValueError("Rebalancing strategy requires a rebalancing period in the parameters dictionnary.\nExample : {'Rebalancing period' : 30}")
            elif parameters["Rebalancing period"] < 0:
                raise ValueError("Rebalancing period must be strickly positive")
        elif self.strategy == "Buy and hold":
            if "Rebalancing period" in parameters and parameters["Rebalancing period"] > 0:
                self.strategy = "Rebalancing"
                warnings.warn("Strategy changed to 'Rebalancing'. Positive rebalancing period is not needed for Buy and hold strategy.\n"
                "To avoid this warning, please use set_strategy() method to change the strategy before updating parameters.")
        self.parameters = parameters

    def set_dataESG(self, dataESG):
        """
        pandas.series representing the ESG data for each stock
        lines :
            - Stock 1 name
            - Stock 2 name
            - ...
        Columns example: 
            - Sustainability risk
            - Exposure risk
            - Score management
            - ...
        """
        self.dataESG = dataESG

    def set_constraints(self, constraints):
        """
        dictionnary containing the types and the values of contraints of the optimization problem

        {
            "List" : example ["Maximal volatilities" ou "Minimal return" ou "Sharpe" , "Minimal durability" ...]
            "Value : example [0, 0.1, ...]
        }
        """
        self.constraints = constraints

    def set_allocation(self, allocation):
        self.parameters["Allocation"] = allocation
    def set_scenarios(self, scenarios):
        self.scenarios = scenarios
    def set_evolutions(self, evolutions):
        self.evolutions = evolutions

    @abstractmethod
    def compute_allocation(self):
        """
        Calculate optimal portfolio allocation based on constraints and ESG data
        """
        pass

    @abstractmethod
    def generate_scenarios(self):
        """
        Generate scenarios of log-returns based on the model and the parameters
        """
        pass
    
    @abstractmethod
    def generate_evolutions(self):
        """
        Generate the evolution of the portfolio value for each scenario
        """
        pass

    @abstractmethod
    def compute_metrics(self):
        """
        Compute the metrics of the simulation
        """
        pass

    @abstractmethod
    def plot(self, type_plot="full", alpha=0.95, figsize=(14, 6)):
        """
        Plot the evolution of the portfolio value for each scenario
        """
        pass

__all__ = ["Simulation"]