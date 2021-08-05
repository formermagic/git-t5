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
    from .trainer import T5Trainer
else:
    T5Trainer = Any


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
    tokenizer_path: Optional[str] = None
    use_fast_tokenizer: bool = True
    cache_dir: Optional[str] = None
    dtype: str = "float32"
    seed: int = 42


@dataclass
class T5ModelForPreTraining:
    config: T5ModelForPreTrainingConfig
    model: FlaxT5ForConditionalGeneration
    tokenizer: PreTrainedTokenizerBase
    trainer: Optional[T5Trainer] = None

    @classmethod
    def from_config(
        cls, config: T5ModelForPreTrainingConfig
    ) -> "T5ModelForPreTraining":
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

        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        # compute loss & gradients
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        # compute perplexity
        perplexity = jnp.exp(loss)
        # apply computed gradients
        grads = jax.lax.pmean(grads, "batch")
        state = state.apply_gradients(grads=grads)
        # prepare train metrics
        metrics = {"loss": loss, "perplexity": perplexity}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return state, metrics, new_dropout_rng

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
        perplexity = jnp.exp(loss).mean()
        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels).mean()
        # prepare validation metrics
        metrics = {"loss": loss, "perplexity": perplexity, "accuracy": accuracy}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return metrics

    @classmethod
    def load_model(
        cls,
        config: T5ModelForPreTrainingConfig,
        tokenizer: PreTrainedTokenizerBase,
    ) -> FlaxT5ForConditionalGeneration:
        if config.model_path is not None:
            model = FlaxT5ForConditionalGeneration.from_pretrained(
                config.model_path,
                config=config,
                seed=config.seed,
                dtype=getattr(jnp, config.dtype),
            )
        else:
            model = FlaxT5ForConditionalGeneration(
                cls.load_config(config, tokenizer),
                seed=config.seed,
                dtype=getattr(jnp, config.dtype),
            )

        return model

    @classmethod
    def load_config(
        cls,
        config: T5ModelForPreTrainingConfig,
        tokenizer: PreTrainedTokenizerBase,
    ) -> T5Config:
        vocab_size = len(tokenizer)  # type: ignore
        if config.config_name is not None:
            model_config = T5Config.from_pretrained(
                config.config_name,
                cache_dir=config.cache_dir,
                vocab_size=vocab_size,
            )
        elif config.model_path is not None:
            model_config = T5Config.from_pretrained(
                config.model_path,
                cache_dir=config.cache_dir,
                vocab_size=vocab_size,
            )
        else:
            model_config = CONFIG_MAPPING[config.model_type]()  # type: ignore

        # make sure the type matches the config
        assert isinstance(model_config, T5Config)
        return model_config

    @classmethod
    def load_tokenizer(
        cls, config: T5ModelForPreTrainingConfig
    ) -> PreTrainedTokenizerBase:
        if config.tokenizer_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_path,
                cache_dir=config.cache_dir,
                use_fast=config.use_fast_tokenizer,
            )
        elif config.model_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                cache_dir=config.cache_dir,
                use_fast=config.use_fast_tokenizer,
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
                "You can do it from another script, save it, and load it from here, using `tokenizer_path`."
            )

        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        return tokenizer
