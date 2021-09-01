from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import hydra
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    _target_: str = "git_t5.DatasetConfig"
    column_name: Optional[str] = None


@dataclass
class LocalDatasetConfig(DatasetConfig):
    _target_: str = "git_t5.LocalDatasetConfig"
    dataset_path: str = MISSING


@dataclass
class HFDatasetConfig(DatasetConfig):
    _target_: str = "git_t5.HFDatasetConfig"
    dataset_name: str = MISSING
    dataset_config: Optional[str] = None


@dataclass
class MultitaskDatasetConfig(DatasetConfig):
    _target_: str = "git_t5.MultitaskDatasetConfig"
    tasks: Dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> None:
        """Resolves dynamic task configs into instantiated instances."""
        for task_name, task_config in self.tasks.items():
            self.tasks[task_name] = hydra.utils.instantiate(task_config)
