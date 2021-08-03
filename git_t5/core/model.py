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
    SchedulerType,
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
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adafactor: bool = False
    scheduler_type: SchedulerType = SchedulerType.LINEAR
    warmup_steps: int = 0
    dtype: str = "float32"
    seed: int = 42


class T5ModelForPreTraining:
    model: FlaxT5ForConditionalGeneration
    tokenizer: PreTrainedTokenizerBase
    trainer: Optional[T5Trainer] = None

    def __init__(self, config: T5ModelForPreTrainingConfig) -> None:
        self.config = config

    def setup(self) -> None:
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

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

    def configure_optimizers(
        self, trainer: T5Trainer
    ) -> Tuple[optax.GradientTransformation, optax.Schedule]:
        # create learning rate schedule
        dataset = trainer.data_module.dataset
        total_steps = len(dataset["train"])
        batch_size = trainer.data_module.config.train_batch_size * jax.device_count()
        num_epochs = trainer.config.max_epochs
        num_train_steps = (total_steps // batch_size) * num_epochs

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=self.config.learning_rate,
            transition_steps=self.config.warmup_steps,
        )
        decay_fn = optax.linear_schedule(
            init_value=self.config.learning_rate,
            end_value=0,
            transition_steps=num_train_steps - self.config.warmup_steps,
        )
        scheduler_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[self.config.warmup_steps],
        )

        if self.config.adafactor:
            # we use the default parameters here to initialize adafactor
            optimizer = optax.adafactor(
                learning_rate=scheduler_fn,
            )
        else:
            optimizer = optax.adamw(
                learning_rate=scheduler_fn,
                b1=self.config.adam_beta1,
                b2=self.config.adam_beta2,
                weight_decay=self.config.weight_decay,
                mask=decay_mask_fn,
            )

        return optimizer, scheduler_fn

    def load_model(self) -> FlaxT5ForConditionalGeneration:
        config = self.load_config()
        if self.config.model_path is not None:
            model = FlaxT5ForConditionalGeneration.from_pretrained(
                self.config.model_path,
                config=config,
                seed=self.config.seed,
                dtype=getattr(jnp, self.config.dtype),
            )
        else:
            model = FlaxT5ForConditionalGeneration(
                config,
                seed=self.config.seed,
                dtype=getattr(jnp, self.config.dtype),
            )

        return model

    def load_config(self) -> T5Config:
        vocab_size = len(self.tokenizer)  # type: ignore
        if self.config.config_name is not None:
            config = T5Config.from_pretrained(
                self.config.config_name,
                cache_dir=self.config.cache_dir,
                vocab_size=vocab_size,
            )
        elif self.config.model_path is not None:
            config = T5Config.from_pretrained(
                self.config.model_path,
                cache_dir=self.config.cache_dir,
                vocab_size=vocab_size,
            )
        else:
            config = CONFIG_MAPPING[self.config.model_type]()  # type: ignore

        # make sure the type matches the config
        assert isinstance(config, T5Config)
        return config

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        if self.config.tokenizer_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_path,
                cache_dir=self.config.cache_dir,
                use_fast=self.config.use_fast_tokenizer,
            )
        elif self.config.model_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                cache_dir=self.config.cache_dir,
                use_fast=self.config.use_fast_tokenizer,
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
                "You can do it from another script, save it, and load it from here, using `tokenizer_path`."
            )

        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        return tokenizer
