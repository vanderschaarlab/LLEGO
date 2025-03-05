from typing import Optional

import numpy as np
import pandas as pd
from bonsai.base.c45tree import C45Tree
from omegaconf import DictConfig

from baselines.BaseModel import BaseModel


class C45Model(BaseModel):
    def __init__(self, hpt: DictConfig):
        super().__init__(hpt=hpt)

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, seed: Optional[int] = None, **kwargs
    ) -> None:
        if self.hyperparameters_loaded:
            model = C45Tree(**self.hyperparameters)
            # check if X and y are pandas array, if so convert to numpy
            if hasattr(X, "values"):
                X = X.values
            model.fit(X, y)
            self.fitted_model = model
            if self.verbose:
                print(f"[C45] Model fitted successfully...")
        else:
            raise ValueError(
                "[C45] Hyperparameters not loaded. Call load_hyperparameters method first."
            )

    def predict(self, X: pd.DataFrame, return_probs: bool = False) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("[C45] Model not fitted yet. Call fit method first.")
        if hasattr(X, "values"):
            X = X.values
        predictions = self.fitted_model.predict(X, output_type="blah")
        if not return_probs:
            # Threshold
            predictions[predictions < 0.5] = 0
            predictions[predictions >= 0.5] = 1

        assert isinstance(
            predictions, np.ndarray
        ), f"Expected numpy array but got {type(predictions)}"
        return predictions

    @property
    def name(self):
        return "C45"
