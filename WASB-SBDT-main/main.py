import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import os

from src.runners import select_runner
from src.utils import mkdir_if_missing

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_name='root', config_path='configs')
def main(
        cfg: DictConfig
):
    if cfg['output_dir'] is None:
        cfg['output_dir'] = HydraConfig.get().run.dir
    mkdir_if_missing(cfg['output_dir'])
    
    runner = select_runner(cfg)
    runner.run()

if __name__ == "__main__":
    main()

