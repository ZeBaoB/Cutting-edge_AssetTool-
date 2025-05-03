import numpy as np
import pandas as pd
from Classes.marketmodel_interface import MarketModel
from utils import calibrate_BS_model, generate_BS_scenarios, calibrate_heston_model, generate_Heston_scenarios

class MarketModelImpl(MarketModel):
    def __init__(self, model_name, parameters=None):
        # Initialize the parent abstract class attributes
        self.model_name = model_name
        self.parameters = parameters

    def fit(self, data):
        parameters = {
            "Stocks": data.columns
        }
        if self.model_name == "BS":
            parameters["Returns"], parameters["Volatilities"], parameters["Correlation matrix"] = calibrate_BS_model(data)

        elif self.model_name == "Heston":
            parameters["Parameters Heston"], parameters["dB_dW correlation"]= calibrate_heston_model(data)

        elif self.model_name == "On verra":
             # Placeholder for other model
            pass
        self.parameters = parameters

    def generate_logreturns(self, beginDate, endDate, number_of_scenarios):
        # Generate the log returns
        if self.model_name == "BS":
            scenarios = generate_BS_scenarios(self.parameters, beginDate, endDate, number_of_scenarios)
        elif self.model_name == "Heston":
            scenarios, Var_scenarios = generate_Heston_scenarios(self.parameters['Parameters Heston'], self.parameters['dB_dW correlation'], beginDate, endDate, number_of_scenarios)
        elif self.model_name == "On verra":
            # Placeholder
            scenarios = None
        return scenarios
