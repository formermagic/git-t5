from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import traverse_util
from flax.training.common_utils import onehot
from flax.training.train_state import TrainState
from transformers import (
    CONFIG_MAPPING,
    AutoTokenizer,
    FlaxT5ForConditionalGeneration,
    PreTrainedTokenizerBase,
    T5Config,
)

if TYPE_CHECKING:
    from git_t5.cli.train_model import Config
    from git_t5.core.trainer import T5Trainer
else:
    Config = Any
    T5Trainer = Any

MAX_VALUE = (2 << 30) - 1


def decay_mask_fn(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    We use Optax's "masking" functionality to not apply weight decay
    to bias and LayerNorm scale parameters. decay_mask_fn returns a
    mask boolean with the same structure as the parameters.
    The mask is True for parameters that should be decayed.
    """

    flat_params = traverse_util.flatten_dict(params)
    flat_mask = {
        path: (
            path[-1] != "bias"  # type: ignore
            and path[-2:]
            not in [("layer_norm", "scale"), ("final_layer_norm", "scale")]
        )
        for path in flat_params
    }

    return traverse_util.unflatten_dict(flat_mask)


@dataclass
class ModelConfig:
    pass


@dataclass
class T5ModelForPreTrainingConfig(ModelConfig):
    model_path: Optional[str] = None
    model_type: Optional[str] = None
    config_name: Optional[str] = None
    dtype: str = "float32"


@dataclass
class T5ModelForPreTraining:
    config: Config
    model: FlaxT5ForConditionalGeneration
    tokenizer: PreTrainedTokenizerBase
    trainer: Optional[T5Trainer] = None

    @classmethod
    def from_config(cls, config: Config) -> "T5ModelForPreTraining":
        tokenizer = cls.load_tokenizer(config)
        model = cls.load_model(config, tokenizer)
        return T5ModelForPreTraining(config, model, tokenizer)

    @partial(
        jax.pmap,
        axis_name="batch",
        static_broadcasted_argnums=(0,),
        donate_argnums=(1,),
    )
    def train_step(
        self,
        state: TrainState,
        dropout_rng: jnp.ndarray,
        batch: Dict[str, Any],
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray], jnp.ndarray]:
        def loss_fn(params: Dict[str, Any]) -> float:
            labels = batch.pop("labels")
            outputs = state.apply_fn(
                **batch,
                params=params,
                dropout_rng=dropout_rng,
                train=True,
            )

            # compute loss
            logits = outputs[0]
            labels = onehot(labels, logits.shape[-1])
            loss = optax.softmax_cross_entropy(logits, labels).mean()

            return loss

        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)  # type: ignore

        # compute loss & gradients
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        # compute perplexity
        perplexity = jnp.exp(loss).clip(0, MAX_VALUE)
        # apply computed gradients
        grads = jax.lax.pmean(grads, "batch")
        state = state.apply_gradients(grads=grads)
        # prepare train metrics
        metrics = {"loss": loss, "perplexity": perplexity}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return state, metrics, new_dropout_rng  # type: ignore

    @partial(
        jax.pmap,
        axis_name="batch",
        static_broadcasted_argnums=(0,),
        donate_argnums=(1,),
    )
    def valid_step(
        self,
        state: TrainState,
        batch: Dict[str, Any],
    ) -> Dict[str, jnp.ndarray]:
        labels = batch.pop("labels")
        outputs = state.apply_fn(**batch, params=state.params, train=False)

        # compute loss
        logits = outputs[0]
        labels_onehot = onehot(labels, logits.shape[-1])
        loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
        # compute perplexity
        perplexity = jnp.exp(loss).mean().clip(0, MAX_VALUE)
        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels).mean()
        # prepare validation metrics
        metrics = {"loss": loss, "perplexity": perplexity, "accuracy": accuracy}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return metrics

    @classmethod
    def load_model(
        cls,
        config: Config,
        tokenizer: PreTrainedTokenizerBase,
    ) -> FlaxT5ForConditionalGeneration:
        if config.model.model_path is not None:
            model = FlaxT5ForConditionalGeneration.from_pretrained(
                config.model.model_path,
                config=config,
                seed=config.training.seed,
                dtype=getattr(jnp, config.model.dtype),
            )
        else:
            model = FlaxT5ForConditionalGeneration(
                cls.load_config(config, tokenizer),
                seed=config.training.seed,
                dtype=getattr(jnp, config.model.dtype),
            )

        return model

    @classmethod
    def load_config(
        cls,
        config: Config,
        tokenizer: PreTrainedTokenizerBase,
    ) -> T5Config:
        vocab_size = len(tokenizer)  # type: ignore
        if config.model.config_name is not None:
            model_config = T5Config.from_pretrained(
                config.model.config_name,
                cache_dir=config.training.cache_dir,
                vocab_size=vocab_size,
            )
        elif config.model.model_path is not None:
            model_config = T5Config.from_pretrained(
                config.model.model_path,
                cache_dir=config.training.cache_dir,
                vocab_size=vocab_size,
            )
        else:
            model_config = CONFIG_MAPPING[config.model_type]()  # type: ignore

        # make sure the type matches the config
        assert isinstance(model_config, T5Config)
        return model_config

    @classmethod
    def load_tokenizer(cls, config: Config) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer.tokenizer_path,
            cache_dir=config.training.cache_dir,
            use_fast=config.tokenizer.use_fast,
        )

        return tokenizer
