import numpy as np
import pandas as pd
from .market_model import MarketModel
from src.utils.financial_utils import calibrate_BS_model, generate_BS_scenarios, calibrate_heston_model, generate_Heston_scenarios

class MarketModelImpl(MarketModel):
    def __init__(self, model_name, parameters=None):
        super().__init__(model_name, parameters)

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
        parameters = {
            "Stocks": data.columns
        }
        if self.model_name == "BS":
            parameters["Returns"], parameters["Volatilities"], parameters["Correlation matrix"] = calibrate_BS_model(data)
        elif self.model_name == "Heston":
            parameters["Parameters Heston"], parameters["dB_dW correlation"]= calibrate_heston_model(data)
        elif self.model_name == "Merton":
            # Placeholder for Merton model calibration
            pass
        else:
            # Default or placeholder for other models
            pass
            
        self.parameters = parameters

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
        Var_scenarios = None
        # Generate the log returns
        if self.model_name == "BS":
            scenarios = generate_BS_scenarios(self.parameters, begin_date, end_date, number_of_scenarios)
        elif self.model_name == "Heston":
            scenarios, Var_scenarios = generate_Heston_scenarios(self.parameters['Parameters Heston'], self.parameters['dB_dW correlation'], begin_date, end_date, number_of_scenarios)
        elif self.model_name == "Merton":
            # Placeholder for Merton model
            scenarios = None
        else:
            # Default or placeholder for other models
            scenarios = None
            
        return scenarios, Var_scenarios
