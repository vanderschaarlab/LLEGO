import copy
import logging

import numpy as np
import pandas as pd
from gosdt import GOSDT
from omegaconf import DictConfig

from baselines.BaseModel import BaseModel

logger = logging.getLogger(__name__)


class GOSDTModel(BaseModel):
    def __init__(self, hpt: DictConfig):
        super().__init__(hpt=hpt)

    def fit(self, X: pd.DataFrame, y: np.ndarray, seed: int = 0, **kwargs) -> None:

        if self.hyperparameters_loaded:

            model = GOSDT(configuration=self.hyperparameters)

            # If y is a numpy array, then convert it to a dataframe with one column
            y_np = pd.DataFrame(y)

            model.fit(X, y_np)
            self.fitted_model = copy.deepcopy(model)
            if self.verbose:
                logger.info(f"[GOSDT] Model fitted successfully...")
        else:
            raise ValueError(
                "[GOSDT] Hyperparameters not loaded. Call load_hyperparameters method first."
            )

    def predict(self, X: pd.DataFrame, return_probs: bool = False) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("[GOSDT] Model not fitted yet. Call fit method first.")

        if return_probs:
            # get the labels  of the predictiosn
            hard_labels = self.fitted_model.predict(X)
            # confidence
            confidence = self.fitted_model.tree.confidence(X)

            opposite = 1 - confidence

            predictions = hard_labels * confidence + (1 - hard_labels) * opposite

        else:
            predictions = self.fitted_model.predict(X)
        assert isinstance(
            predictions, np.ndarray
        ), f"Expected numpy array but got {type(predictions)}"
        return predictions

    @property
    def name(self):
        return "GOSDT"
