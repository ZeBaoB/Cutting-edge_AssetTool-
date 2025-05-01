from abc import ABC, abstractmethod

class MarketModel(ABC):
    def __new__(cls, *args, **kwargs):
        if cls is MarketModel:
            # Import here to avoid circular import
            from .market_model_impl import MarketModelImpl
            # Create instance of concrete class
            instance = super().__new__(MarketModelImpl)
            return instance
        return super().__new__(cls)
    
    def __init__(self, model_name, parameters=None):
        self.model_name = model_name
        self.parameters = parameters
    
    def set_parameters(self, parameters):
        """
        Set the parameters for the model
        
        Parameters
        ----------
        parameters : dict
            Dictionary of parameters for the model
        """
        self.parameters = parameters
    
    @abstractmethod
    def fit(self, data):
        """
        Calibrate the model based on the data (historical prices)
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the historical prices
            columns : stock names
            lines : dates
        """
        pass

    @abstractmethod
    def generate_logreturns(self, begin_date, end_date, number_of_scenarios):
        """
        Generate scenarios of log-returns
        
        Parameters
        ----------
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
        pass
