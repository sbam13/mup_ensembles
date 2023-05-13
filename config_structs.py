from dataclasses import dataclass, field
from omegaconf import MISSING
# from hydra.core.config_store import ConfigStore

@dataclass
class WarmupCosineDecayParameters:
    warmup_epochs: float = 0.5
    init_lr: float = 2e-3
    min_lr: float = 6e-5


@dataclass
class TrainingParams:
    eta_0: float = 1e-2
    # momentum: float = 0.9
    # weight_decay: float = 1e-5 # TODO: not implemented; replace with batch_size
    minibatch_size: int = 1024 # changed
    microbatch_size: int = 64
    num_workers: int = 24
    epochs: int = 50
    full_batch_gradient: bool = False
    ensemble_subsets: int = 1 # number of subsets of the ensemble to be run synchronously (increase from 1 if out-of-memory during training); divides ensemble_size
    use_checkpoint: bool = False
    ckpt_dir: str = ''
    model_ckpt_dir: str = ''
    use_warmup_cosine_decay: bool = False
    wcd_params: WarmupCosineDecayParameters = field(default_factory=WarmupCosineDecayParameters)


@dataclass
class DataParams:
    P: int = 2 ** 20
    # k: int = 0 # no target fn
    # on_device: bool = True
    # random_subset: bool = True
    data_seed: int = MISSING
    root_dir: str = 'data-dir'
    val_P: int = 1024


@dataclass
class ModelParams:
    BASE_N: int = 64
    N: int = 128
    # widths: list[int] = [100] # TODO: added for ensembling within GPU
    # repeat: int = 32 # replicated
    ensemble_size: int = 1
    dtype: str = 'bfloat16'


@dataclass
class TaskConfig:
    training_params: TrainingParams = field(default_factory=TrainingParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    # repeat: int = 32
    # parallelize: bool = True
    seed: int = MISSING


@dataclass
class TaskListConfig:
    task_list: list[TaskConfig] = field(default_factory=list)
    data_params: DataParams = field(default_factory=DataParams)

@dataclass
class Setting:
  dataset: str = 'imagenet'
  model: str = 'resnet18'

@dataclass
class Config:
    setting: Setting
    hyperparams: TaskListConfig
    base_dir: str = ''


# def conf_register() -> None:
#     cs = ConfigStore.instance()
#     cs.store(group='hyperparams', name='base-tasks', node=TaskListConfig) # replace node
#     cs.store(group='setting', name='base-cifar10-resnet', node=Setting)
#     cs.store(name='base-config', node=Config)