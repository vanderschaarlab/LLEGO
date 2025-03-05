import logging
import pickle

import optuna
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error

logger = logging.getLogger(__name__)
METRICS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "mse": mean_squared_error,
}


def tune_models(
    metric_name,
    model,
    model_name,
    cfg_hpt,
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials,
    path_save=None,
    suffix=None,
    kwargs=None,
):

    # Load hyperparameter search space
    model.verbose = False

    metric_fn = METRICS[metric_name]

    def objective(trial):
        suggested_params = {}
        for param, param_space in cfg_hpt["tunable_parameters"].items():
            if param_space["type"] == "int":
                suggested_params[param] = trial.suggest_int(
                    param, param_space["min"], param_space["max"]
                )
            elif param_space["type"] == "float":
                # check if log is in the keys
                useLog = False
                if "log" in param_space.keys():
                    useLog = param_space["log"]
                suggested_params[param] = trial.suggest_float(
                    param, param_space["min"], param_space["max"], log=useLog
                )
            else:
                raise ValueError(f"Unsupported type: {param_space['type']}")

        suggested_params.update(cfg_hpt["fixed_parameters"])

        model.load_hyperparameters(suggested_params)
        try:
            model.fit(X_train, y_train, seed=69, **kwargs)
            if metric_name != "roc_auc":
                y_pred = model.predict(X_val, return_probs=False)
            else:
                y_pred = model.predict(X_val, return_probs=True)[:, 1]

            metric = metric_fn(y_val, y_pred)

            dic_save = {
                "model": model,
                "hyperparameters": suggested_params,
                "metric": metric,
                "X_train": X_train,
                "y_train": y_train,
            }

            if path_save is not None:
                try:
                    with open(
                        f"{path_save}/{model_name}_{trial.number}_{suffix}.pkl", "wb"
                    ) as f:
                        pickle.dump(dic_save, f)
                except:
                    print(f"Error saving trial {trial.number}")
        except Exception as e:
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {e}")
            metric = 0.0

        return metric

    if not "tunable_parameters" in cfg_hpt.keys():
        return dict(cfg_hpt["fixed_parameters"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials)

    tuned_hyperparameters = study.best_params
    best_accuracy = study.best_value

    tuned_hyperparameters.update(cfg_hpt["fixed_parameters"])

    logger.info(
        f"[HPT] Complete, best {metric_name}: {best_accuracy:.4f} with hyperparameters: {tuned_hyperparameters}"
    )

    return tuned_hyperparameters
