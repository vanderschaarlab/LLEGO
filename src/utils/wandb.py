from typing import Union

import wandb
from omegaconf import DictConfig

from utils.extraction import flatten_config


def maybe_initialize_wandb(
    cfg: DictConfig, dedicated_exp_name: str, dedicated_exp_dir: str
) -> Union[str, None]:
    """Initialize wandb if necessary."""
    cfg_flat = flatten_config(cfg)
    if cfg.log_wandb:
        wandb.init(
            project="LLEGO",
            config=cfg_flat,
            name=dedicated_exp_name,
            dir=dedicated_exp_dir,
        )
        assert wandb.run is not None
        run_id = wandb.run.id
        assert isinstance(run_id, str)
        return run_id
    else:
        return None
