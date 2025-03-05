import logging
from typing import Optional

import numpy as np
import pandas as pd
from baselines.BaseModel import BaseModel
from omegaconf import DictConfig
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)


class CARTModel(BaseModel):
    def __init__(
        self, task_type: str = "classification", hpt: Optional[DictConfig] = None
    ):
        super().__init__(hpt=hpt)
        assert task_type in [
            "classification",
            "regression",
        ], f"[CART] Invalid task type: {task_type}. Choose either 'classification' or 'regression'."
        self.task_type = task_type

    def fit(self, X: pd.DataFrame, y: np.ndarray, seed: int, **kwargs) -> None:
        if self.hyperparameters_loaded:

            if self.task_type == "classification":
                model = DecisionTreeClassifier(
                    **self.hyperparameters, random_state=seed
                )
            elif self.task_type == "regression":
                model = DecisionTreeRegressor(**self.hyperparameters, random_state=seed)
            else:
                raise ValueError(f"[CART] Invalid task type: {self.task_type}")
            self.fitted_model = model.fit(X, y)
            if self.verbose:
                logger.info(f"[CART] Model fitted successfully...")
        else:
            raise ValueError(
                "[CART] Hyperparameters not loaded. Call load_hyperparameters method first."
            )

    def predict(self, X: pd.DataFrame, return_probs: bool = False) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("[CART] Model not fitted yet. Call fit method first.")

        if return_probs:
            predictions = self.fitted_model.predict_proba(X)[:, 1]
        else:
            predictions = self.fitted_model.predict(X)
        assert isinstance(
            predictions, np.ndarray
        ), f"Expected numpy array but got {type(predictions)}"
        return predictions

    @property
    def name(self):
        return f"CART_{self.task_type}"
