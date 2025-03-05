import json
import logging
import os

logger = logging.getLogger(__name__)


def save_results(results_dict, model_name, seed, experiment_name=None):
    assert isinstance(results_dict, dict)

    save_file_path = f"experiments/results/{experiment_name}/{model_name}_{seed}.json"

    logger.info(f"Saving results to {save_file_path}...")

    # make the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    with open(save_file_path, "w") as f:
        json.dump(results_dict, f, indent=4, default=int)
