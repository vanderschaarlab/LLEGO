import logging
import os
import pickle
from datetime import datetime

import hydra
import numpy as np
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from baselines.gatree_algorithm import GATreeAlgorithm
from utils.data_utils import get_data, preprocess_data
from utils.wandb import maybe_initialize_wandb

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="gatree")
def main(cfg: DictConfig) -> None:

    dataset_name = cfg.dataset.dataset_name
    model_name = "GATREE"
    train_val_test_split = list(cfg.train_val_test_split)
    seed = cfg.seed
    exp_name = cfg.exp_name
    include_task_semantics = cfg.include_task_semantics
    max_depth = cfg.max_depth
    depth = cfg.max_depth
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    preprocessing_fn = preprocess_data

    dedicated_exp_dir = f"{exp_name}/{depth}/{dataset_name}/{model_name}"
    if not os.path.exists(dedicated_exp_dir):
        os.makedirs(dedicated_exp_dir)
    artifacts_dir = f"{dedicated_exp_dir}/artifacts"
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    dataset_details = cfg.dataset
    now = datetime.now()
    formatted_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    dedicated_exp_name = (
        f"{exp_name}_{dataset_name}_{max_depth}_{seed}_gatree_{formatted_time}"
    )

    maybe_initialize_wandb(cfg, dedicated_exp_name, dedicated_exp_dir)

    # Load data for hyperparameter tuning
    X, y, meta_data = get_data(
        dataset_name,
        dataset_details=dataset_details,
        include_task_semantics=include_task_semantics,
    )
    data, meta_data = preprocessing_fn(
        X, y, meta_data, train_val_test_split=train_val_test_split, seed=seed
    )

    X_train, _, _, y_train, _, _ = (
        data["X_train"],
        data["X_val"],
        data["X_test"],
        data["y_train"],
        data["y_val"],
        data["y_test"],
    )

    # Instanciate the operators

    # initialization operator
    initialization_operator = instantiate(cfg.pop_init)
    initialization_operator = initialization_operator(data=data, meta_data=meta_data)

    # fitness evaluator
    fitness_evaluator = instantiate(cfg.fitness_eval)
    fitness_evaluator = fitness_evaluator(data=data, task_type=meta_data["task_type"])

    # metrics logger
    metrics_logger = instantiate(cfg.metrics_logger)

    # hall of fame
    hall_of_fame = instantiate(cfg.hof)

    ############################################
    feature_list = meta_data["attribute_names"]

    att_indexes = range(len(feature_list))
    att_values = {
        i: [
            (min_val + max_val) / 2
            for min_val, max_val in zip(
                sorted(X_train.iloc[:, i].unique())[:-1],
                sorted(X_train.iloc[:, i].unique())[1:],
            )
        ]
        for i in range(X_train.shape[1])
    }
    att_values[-1] = sorted(np.unique(y_train))
    class_count = len(att_values[-1])
    ############################################

    gatree_algorithm = GATreeAlgorithm(
        n_iterations=cfg.gatree.n_iterations,
        pop_size=cfg.gatree.pop_size,
        mutation_probability=cfg.gatree.mutation_probability,
        pop_initializer=initialization_operator,
        fitness_evaluator=fitness_evaluator,
        metrics_logger=metrics_logger,
        hall_of_fame=hall_of_fame,
        elite_size=cfg.gatree.elite_size,
        fitness_metric=cfg.metric_key,
        lower_is_better=cfg.fitness_eval.lower_is_better,
        selection_tournament_size=cfg.gatree.selection_tournament_size,
        max_depth=max_depth,
    )

    # run the algorithm
    population_across_iterations, hof = gatree_algorithm.run(
        att_indexes=att_indexes,
        att_values=att_values,
        class_count=class_count,
        feature_list=feature_list,
        task=meta_data["task_type"],
        seed=seed,
    )

    # save the final population
    with open(
        f"{dedicated_exp_dir}/gatree_search_populations_seed_{seed}.pkl", "wb"
    ) as f:
        pickle.dump(population_across_iterations, f)
    # save the hall of fame
    with open(f"{dedicated_exp_dir}/gatree_hof_seed_{seed}.pkl", "wb") as f:
        pickle.dump(hof, f)
    # save the config
    with open(f"{dedicated_exp_dir}/gatree_config_{seed}.yaml", "w") as f:
        yaml.dump(config_dict, f)

    logger.info("Experiment completed successfully...")
    logger.info(f"Population and hall of fame saved to {dedicated_exp_dir}...")


if __name__ == "__main__":
    main()
