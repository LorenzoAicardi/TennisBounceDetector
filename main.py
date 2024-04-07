import logging

import hydra
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from runners import select_runner
from utils import mkdir_if_missing

from omegaconf import DictConfig 

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_name='root', config_path='configs')
def main(
        cfg: DictConfig
):
    # print(OmegaConf.to_yaml(cfg))
    # print(cfg)
    if cfg['output_dir'] is None:
        cfg['output_dir'] = HydraConfig.get().run.dir
    mkdir_if_missing(cfg['output_dir'])
    
    runner = select_runner(cfg)
    runner.run()

if __name__ == "__main__":
    main()