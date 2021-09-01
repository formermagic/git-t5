import os
from dataclasses import dataclass

import hydra
from git_t5.core import (
    T5DataModule,
    T5DataModuleConfig,
    T5ModelForPreTraining,
    T5ModelForPreTrainingConfig,
    T5Trainer,
    T5TrainerConfig,
    WandbLogger,
    WandbLoggerConfig,
)
from git_t5.core.configs import (
    HFDatasetConfig,
    LocalDatasetConfig,
    MultitaskDatasetConfig,
)
from git_t5.core.optimizers import (
    AdafactorConfig,
    AdagradConfig,
    AdamConfig,
    AdamWConfig,
    OptimizerConfig,
)
from git_t5.core.schedulers import (
    ConstantSchedulerConfig,
    InverseSquareRootSchedulerConfig,
    LinearSchedulerConfig,
    PolynomialSchedulerConfig,
)
from hydra.core.config_store import ConfigStore

from .config import DefaultConfig, register_base_configs


@dataclass
class Config(DefaultConfig):
    data: T5DataModuleConfig = T5DataModuleConfig()
    model: T5ModelForPreTrainingConfig = T5ModelForPreTrainingConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    trainer: T5TrainerConfig = T5TrainerConfig()
    logger: WandbLoggerConfig = WandbLoggerConfig()


def register_optimizers(cs: ConfigStore) -> None:
    cs.store(
        group="optimizer",
        name="base_adam",
        node=AdamConfig,
    )
    cs.store(
        group="optimizer",
        name="base_adamw",
        node=AdamWConfig,
    )
    cs.store(
        group="optimizer",
        name="base_adafactor",
        node=AdafactorConfig,
    )
    cs.store(
        group="optimizer",
        name="base_adagrad",
        node=AdagradConfig,
    )


def register_schedulers(cs: ConfigStore) -> None:
    cs.store(
        group="optimizer/scheduler",
        name="base_polynomial",
        node=PolynomialSchedulerConfig,
    )
    cs.store(
        group="optimizer/scheduler",
        name="base_inverse_square_root",
        node=InverseSquareRootSchedulerConfig,
    )
    cs.store(
        group="optimizer/scheduler",
        name="base_linear",
        node=LinearSchedulerConfig,
    )
    cs.store(
        group="optimizer/scheduler",
        name="base_constant",
        node=ConstantSchedulerConfig,
    )


def register_datasets(cs: ConfigStore) -> None:
    cs.store(
        group="dataset",
        name="base_huggingface_dataset",
        node=HFDatasetConfig,
    )
    cs.store(
        group="dataset",
        name="base_local_dataset",
        node=LocalDatasetConfig,
    )
    cs.store(
        group="dataset",
        name="base_multitask_dataset",
        node=MultitaskDatasetConfig,
    )


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="default", node=Config)
    cs.store(
        group="data",
        name="base_data",
        node=T5DataModuleConfig,
    )
    cs.store(
        group="model",
        name="base_model",
        node=T5ModelForPreTrainingConfig,
    )
    cs.store(
        group="trainer",
        name="base_trainer",
        node=T5TrainerConfig,
    )
    cs.store(
        group="logger",
        name="base_logger",
        node=WandbLoggerConfig,
    )

    register_optimizers(cs)
    register_schedulers(cs)
    register_datasets(cs)


@hydra.main(config_path="conf", config_name="config_model")
def hydra_entry(cfg: Config) -> None:
    # disable rust iterators multithreading
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger = WandbLogger(cfg.logger)
    model = T5ModelForPreTraining.from_config(cfg)
    data_module = T5DataModule.from_config(cfg)

    trainer = T5Trainer(
        config=cfg,
        model=model,
        data_module=data_module,
        logger=logger,
    )

    trainer.fit()


def main() -> None:
    register_base_configs()
    register_configs()
    hydra_entry()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
