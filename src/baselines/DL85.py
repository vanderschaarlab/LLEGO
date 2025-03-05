import numpy as np
import pandas as pd
from baselines.BaseModel import BaseModel
from omegaconf import DictConfig
from pydl85 import DL85Classifier


class DL85Model(BaseModel):
    def __init__(self, hpt: DictConfig):
        super().__init__(hpt=hpt)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        seed: int,
        **kwargs,
    ) -> None:

        if self.hyperparameters_loaded:
            model = DL85Classifier(**self.hyperparameters)
            # convert y to int
            y = y.astype(int)
            sample_weight = (
                None if "sample_weight" not in kwargs else kwargs["sample_weight"]
            )

            print(f"[DL85] Using following sample_weight: {sample_weight}")
            self.fitted_model = model.fit(X, y, sample_weight=sample_weight)

            if self.verbose:
                print(f"[DL85] Model fitted successfully...")
        else:
            raise ValueError(
                "[DL85] Hyperparameters not loaded. Call load_hyperparameters method first."
            )

    def predict(self, X: pd.DataFrame, return_probs: bool = False) -> np.ndarray:
        if self.fitted_model is None:
            raise ValueError("[DL85] Model not fitted yet. Call fit method first.")

        if return_probs:

            predictions = self.fitted_model.predict_proba(X)[:, 1]
        else:
            predictions = self.fitted_model.predict(X)
        predictions = np.array(predictions)
        return predictions

    @property
    def name(self):
        return "DL85"
