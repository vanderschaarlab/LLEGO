import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_pred, y_true, task_type="classification"):
    """
    Calculate the accuracy, precision, recall of the model.
    Arguments:
        y_pred : np.array : Predicted values [should be probabilities]
        y_true : np.array : True values [should be binary]
    """
    assert len(y_true) == len(y_pred)
    assert len(y_true.shape) == len(y_pred.shape) == 1

    if task_type == "regression":
        result_dict = {}
        result_dict["mse"] = mean_squared_error(y_true, y_pred)
        result_dict["mae"] = mean_absolute_error(y_true, y_pred)
    elif task_type == "classification":

        # y_true is binary
        assert set(y_true) == {0, 1}
        # y_pred is probabilities

        assert np.all(y_pred <= 1)
        assert np.all(y_pred >= 0) and np.all(y_pred <= 1)

        y_pred_label = (y_pred > 0.5).astype(int)

        result_dict = {}
        result_dict["accuracy"] = accuracy_score(y_true, y_pred_label)
        result_dict["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred_label)
        result_dict["precision"] = precision_score(y_true, y_pred_label)
        result_dict["recall"] = recall_score(y_true, y_pred_label)
        result_dict["roc_auc"] = roc_auc_score(y_true, y_pred)
    else:
        raise ValueError(f"Task type {task_type} not supported.")

    return result_dict
