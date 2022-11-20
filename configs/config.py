import os
from typing import Text, Union

from omegaconf import OmegaConf, DictConfig, ListConfig


def get_config(config_file: Text = 'base') -> Union[DictConfig, ListConfig]:
    if not config_file.endswith(".yaml") or not config_file.endswith(".yml"):
        config_file += ".yaml"
    root_configs_dir = os.path.abspath(os.path.join(__file__, ".."))
    job_cfg = OmegaConf.load(os.path.join(root_configs_dir, config_file))
    return job_cfg


cfg = get_config()
