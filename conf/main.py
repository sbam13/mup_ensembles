from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig, MISSING
from hydra.core.config_store import ConfigStore

import hydra


from config_structs import conf_register

conf_register()

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # print(cfg)


if __name__ == '__main__':
    main()