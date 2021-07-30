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
from hydra.core.config_store import ConfigStore

from .config import DefaultConfig, register_base_configs


@dataclass
class Config(DefaultConfig):
    data: T5DataModuleConfig = T5DataModuleConfig()
    model: T5ModelForPreTrainingConfig = T5ModelForPreTrainingConfig()
    trainer: T5TrainerConfig = T5TrainerConfig()
    logger: WandbLoggerConfig = WandbLoggerConfig()


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


@hydra.main(config_path="../../conf", config_name="config_model")
def hydra_entry(cfg: Config) -> None:
    logger = WandbLogger(cfg.logger)
    model = T5ModelForPreTraining(cfg.model)
    data_module = T5DataModule(cfg.data)

    trainer = T5Trainer(
        cfg.trainer,
        model,
        data_module,
        logger,
    )

    trainer.fit()


def main() -> None:
    register_base_configs()
    register_configs()
    hydra_entry()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
