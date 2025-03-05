import json
import logging as logger
import os
import warnings

import pandas as pd

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from utils.data_utils import (
    binarize_dataset,
    compute_sample_weights,
    get_data,
    preprocess_data,
)
from utils.eval_utils import evaluate_model
from utils.tune_models import tune_models

from llego.custom.parsing_to_dict import parse_cart_to_dict


@hydra.main(config_path="../configs", config_name="baselines")
def main(cfg: DictConfig) -> None:
    # Instanciate model
    model = instantiate(cfg.baseline)
    model_name = model.name

    dataset_name = cfg.dataset.dataset_name
    depth = cfg.max_depth
    metric_name = cfg.metric_name
    train_val_test_split = list(cfg.train_val_test_split)
    seed = cfg.seed

    cfg_hpt = cfg.baseline.hpt
    n_trials_hpt = cfg.n_trials_hpt
    exp_name = cfg.exp_name
    include_task_semantics = cfg.include_task_semantics

    preprocessing_fn = (
        preprocess_data
        if model_name not in ["DL85"]
        else binarize_dataset  # DL85 requires binarized data
    )

    dedicated_exp_dir = f"{exp_name}/{depth}/{dataset_name}/{model_name}"
    if not os.path.exists(dedicated_exp_dir):
        os.makedirs(dedicated_exp_dir)
    artifacts_dir = f"{dedicated_exp_dir}/artifacts"
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    dataset_details = cfg.dataset

    # Load data for hyperparameter tuning
    X, y, meta_data = get_data(
        dataset_name,
        dataset_details=dataset_details,
        include_task_semantics=include_task_semantics,
    )

    data, meta_data = preprocessing_fn(
        X, y, meta_data, train_val_test_split=train_val_test_split, seed=seed
    )

    X_train, X_val, X_test, y_train, y_val, y_test = (
        data["X_train"],
        data["X_val"],
        data["X_test"],
        data["y_train"],
        data["y_val"],
        data["y_test"],
    )

    logger.info(
        f"Running experiment with dataset: {dataset_name}, model: {model_name}, seed: {seed}"
    )

    kwargs = {}

    if cfg.use_sample_weights:
        # compute sample weights based on the training data
        sample_weights, class_weights = compute_sample_weights(y_train)
        kwargs = {"sample_weight": sample_weights}

    # Hyperparameter tuning
    if n_trials_hpt > 0:
        tuned_hyperparameters = tune_models(
            metric_name,
            model,
            model_name,
            cfg_hpt,
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=n_trials_hpt,
            path_save=artifacts_dir,
            suffix=f"seed_{seed}",
            kwargs=kwargs,
        )

    model.load_hyperparameters(tuned_hyperparameters)

    # Model training
    model.fit(X_train, y_train, seed=seed, **kwargs)
    # Get predictions

    return_probs = True if meta_data["task_type"] == "classification" else False
    y_pred = model.predict(X_train, return_probs=return_probs)

    # Get train and test set results
    train_set_results = evaluate_model(
        y_pred, y_train, task_type=meta_data["task_type"]
    )

    y_pred = model.predict(X_test, return_probs=return_probs)
    test_set_results = evaluate_model(y_pred, y_test, task_type=meta_data["task_type"])

    results = {
        "train": train_set_results,
        "test": test_set_results,
        "hyperparameters": tuned_hyperparameters,
    }

    if model_name == "CART":
        # Save the tree representation for CART
        parsed_tree, native_representation = parse_cart_to_dict(
            cart_model=model.fitted_model,
            feature_names=meta_data["attribute_names"],
            precision=4,
            task_type=meta_data["task_type"],
        )
        results["parsed_tree"] = parsed_tree
        results["native_representation"] = native_representation

    # Save results
    with open(f"{dedicated_exp_dir}/results_seed_{seed}.json", "w") as f:
        json.dump(results, f, indent=4, default=int)


if __name__ == "__main__":
    main()
