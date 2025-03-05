from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from omegaconf import DictConfig


class BaseModel(ABC):
    """
    A generic base class for supervised methods.
    """

    def __init__(self, hpt: Optional[DictConfig] = None):
        """
        Initialize the model parameters.
        """

        self.hyperparameters_loaded: bool = False
        self.fitted_model = None
        self.verbose: bool = True
        self.hpt_config = hpt

    def load_hyperparameters(self, hyperparameters_dict: dict) -> None:
        """
        Load the hyperparameters for the model.

        Parameters:
        - hyperparameters_dict: A dictionary of hyperparameters.
        """
        self.hyperparameters = hyperparameters_dict
        self.hyperparameters_loaded = True

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        seed: int,
    ):
        """
        Fit the model to the training data.

        Parameters:
        - X: Features from the training data.
        - y: Target variable from the training data.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, return_probs: bool):
        """
        Predict the target variable for given data
        """
        pass
