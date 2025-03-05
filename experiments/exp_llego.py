import os
import pickle
import warnings

import hydra
import wandb
import yaml

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
import logging
from datetime import datetime

import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from llego.custom.llm_output_parser import LLMOutputParser
from llego.llego_algorithm import Algorithm
from utils.data_utils import get_data, preprocess_data
from utils.wandb import maybe_initialize_wandb

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="llego")
def main(cfg):
    # OmegaConf.resolve(cfg)

    exp_name = cfg.exp_name
    dataset_name = cfg.dataset.dataset_name
    max_depth = cfg.max_depth
    seed = cfg.seed
    include_task_semantics = cfg.include_task_semantics

    now = datetime.now()
    formatted_time = now.strftime("%Y_%m_%d_%H_%M_%S")

    dedicated_exp_dir = f"{exp_name}/{dataset_name}/{max_depth}/{seed}/llego"
    dedicated_exp_name = (
        f"{exp_name}_{dataset_name}_{max_depth}_{seed}_llego_{formatted_time}"
    )
    if not os.path.exists(dedicated_exp_dir):
        os.makedirs(dedicated_exp_dir)

    logging.info(f"Running experiment: {dedicated_exp_name}...")

    maybe_initialize_wandb(cfg, dedicated_exp_name, dedicated_exp_dir)

    # load data
    train_val_test_split = cfg.train_val_test_split
    dataset_details = cfg.dataset
    X, y, meta_data = get_data(
        dataset_name,
        dataset_details=dataset_details,
        include_task_semantics=include_task_semantics,
    )
    data, meta_data = preprocess_data(
        X, y, meta_data, train_val_test_split=train_val_test_split, seed=seed
    )

    # initialization operator
    initialization_operator = instantiate(cfg.pop_init)
    initialization_operator = initialization_operator(data=data, meta_data=meta_data)

    # fitness evaluator
    fitness_evaluator = instantiate(cfg.fitness_eval)
    fitness_evaluator = fitness_evaluator(data=data, task_type=meta_data["task_type"])

    # prompt
    mut_prompt_template = cfg.prompts.mutation
    mut_prompt_prefix = mut_prompt_template.format(**meta_data)

    crossover_prompt_template = cfg.prompts.crossover
    crossover_prompt_prefix = crossover_prompt_template.format(**meta_data)
    content = cfg.prompts.content

    # mutation llm api
    mut_llm_api = instantiate(cfg.llm_api)
    mut_llm_api = mut_llm_api(
        system_message=content,
        with_logprobs=True,
        **cfg.endpoint.mut_llm,
    )

    # mutation output parser
    mut_llm_output_parser = LLMOutputParser(
        max_depth=max_depth, tree_metadata=meta_data, with_logprobs=True
    )

    # mutation operator
    mutation_operator = instantiate(cfg.mutation)
    mutation_operator = mutation_operator(
        llm_api=mut_llm_api,
        llm_output_parser=mut_llm_output_parser,
        prompt_prefix=mut_prompt_prefix,
        seed=seed,
    )

    # crossover llm api
    crossover_llm_api = instantiate(cfg.llm_api)
    crossover_llm_api = crossover_llm_api(
        system_message=content,
        with_logprobs=False,
        **cfg.endpoint.xo_llm,
    )

    # crossover output parser
    crossover_llm_output_parser = LLMOutputParser(
        max_depth=max_depth, tree_metadata=meta_data, with_logprobs=False
    )
    # crossover operator
    crossover_operator = instantiate(cfg.crossover)
    crossover_operator = crossover_operator(
        llm_api=crossover_llm_api,
        llm_output_parser=crossover_llm_output_parser,
        prompt_prefix=crossover_prompt_prefix,
        seed=seed,
    )

    selection_operator = instantiate(cfg.selection)

    # metrics logger
    metrics_logger = instantiate(cfg.metrics_logger)

    # hall of fame
    hall_of_fame = instantiate(cfg.hof)

    # algorithm
    algorithm = Algorithm(
        **cfg.llego,
        pop_initializer=initialization_operator,
        pop_selector=selection_operator,
        crossover_operator=crossover_operator,
        mutation_operator=mutation_operator,
        fitness_evaluator=fitness_evaluator,
        metrics_logger=metrics_logger,
        hall_of_fame=hall_of_fame,
    )

    # run the algorithm
    population_across_iterations, hof = algorithm.run(seed=seed)

    # save the final population
    with open(f"{dedicated_exp_dir}/llego_search_populations.pkl", "wb") as f:
        pickle.dump(population_across_iterations, f)
    # save the hall of fame
    with open(f"{dedicated_exp_dir}/llego_hof.pkl", "wb") as f:
        pickle.dump(hof, f)
    # save the config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(f"{dedicated_exp_dir}/llego_config.yaml", "w") as f:
        yaml.dump(config_dict, f)

    logger.info("Experiment completed successfully...")
    logger.info(f"Population and hall of fame saved to {dedicated_exp_dir}...")


if __name__ == "__main__":
    main()
