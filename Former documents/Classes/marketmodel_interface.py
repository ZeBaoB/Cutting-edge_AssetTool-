from abc import ABC, abstractmethod

class MarketModel(ABC):
    def __new__(cls, *args, **kwargs):
        if cls is MarketModel:
            # Import here to avoid circular import
            from .marketmodel_implementation import MarketModelImpl
            # Create instance of concrete class
            instance = super().__new__(MarketModelImpl)
            #instance.__init__(*args, **kwargs)
            return instance
        return super().__new__(cls)
    
    def _init_(self, model_name, parameters=None):
        self.model_name = model_name
        self.parameters = parameters
    
    @abstractmethod
    def fit(self, data):
        """
        Calibrate the model based on the data (historical prices)
        data : pd.DataFrame containing the historical prices
            columns : stock names
            lines : dates
        """
        pass

    @abstractmethod
    def generate_logreturns(self, beginDate, endDate, number_of_scenarios):
        """
        Generate scenarios of log-returns
        """
        pass


__all__ = ["MarketModel"]
