from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class TrainingConfig:
    output_dir: str = MISSING
    checkpoint_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    overwrite_cache: bool = False
    push_to_hub: bool = False
    push_to_hub_model_id: Optional[str] = None
    push_to_hub_organization: Optional[str] = None
    push_to_hub_token: Optional[str] = None
    seed: int = 42
