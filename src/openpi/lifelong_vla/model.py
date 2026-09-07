"""pi0 model branch with dual-timescale LoRA and stochastic feature replay."""

from __future__ import annotations

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.lifelong_vla import dual_lora
from openpi.lifelong_vla import gemma
from openpi.lifelong_vla.config import LifelongPi0Config
from openpi.models import model as _model
from openpi.models import pi0 as base_pi0
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at


class LifelongPi0(base_pi0.Pi0):
    """An opt-in pi0 variant implementing the LifelongVLA training interface."""

    def __init__(self, config: LifelongPi0Config, rngs: nnx.Rngs):
        _model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05

        gate_width = gemma.get_config(config.paligemma_variant).width
        adapter = dual_lora.DualLoRAConfig(
            short_rank=config.short_rank,
            long_rank=config.long_rank,
            alpha=config.lora_alpha,
            gate_context_dim=gate_width,
            gate_bias=config.gate_bias,
        )
        paligemma_config = gemma.get_config(config.paligemma_variant, adapter)
        action_expert_config = gemma.get_config(config.action_expert_variant, adapter)
        llm = nnx_bridge.ToNNX(
            gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.deterministic = True

    @staticmethod
    def _gate_context(prefix_tokens: at.Array, prefix_mask: at.Array) -> at.Array:
        weights = prefix_mask.astype(jnp.float32)[..., None]
        denominator = jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        pooled = jnp.sum(prefix_tokens.astype(jnp.float32) * weights, axis=1) / denominator
        return jax.lax.stop_gradient(pooled)

    @staticmethod
    def sample_diffusion(rng: at.KeyArrayLike, actions: _model.Actions) -> tuple[at.Array, at.Array]:
        noise_rng, time_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1.0, actions.shape[:-2]) * 0.999 + 0.001
        return noise, time

    def _predict_from_prefix(
        self,
        prefix_tokens: at.Array,
        prefix_mask: at.Array,
        state: at.Array,
        actions: at.Array,
        noise: at.Array,
        time: at.Array,
        *,
        gradient_mode: dual_lora.GradientMode,
    ) -> tuple[at.Array, at.Array]:
        time_expanded = time[..., None, None]
        noised_actions = time_expanded * noise + (1.0 - time_expanded) * actions
        target = noise - actions
        suffix_obs = _model.Observation(images={}, image_masks={}, state=state)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(suffix_obs, noised_actions, time)

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        prefix_ar_mask = jnp.zeros((prefix_tokens.shape[1],), dtype=jnp.bool_)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attention_mask = base_pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        gate_context = self._gate_context(prefix_tokens, prefix_mask)
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attention_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            gate_context=gate_context,
            gradient_mode=gradient_mode,
        )
        prediction = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return prediction, target

    def current_loss_and_features(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = True,
    ) -> tuple[at.Array, dict[str, at.Array]]:
        preprocess_rng, diffusion_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        prefix_tokens, prefix_mask, _ = self.embed_prefix(observation)
        noise, time = self.sample_diffusion(diffusion_rng, actions)
        prediction, target = self._predict_from_prefix(
            prefix_tokens, prefix_mask, observation.state, actions, noise, time, gradient_mode="short"
        )
        loss = jnp.mean(jnp.square(prediction - target), axis=-1)
        features = {
            # NumPy serializes bfloat16 as an opaque two-byte dtype. Float16 keeps
            # the same memory footprint and round-trips through the NPZ sidecar.
            "prefix_tokens": jax.lax.stop_gradient(prefix_tokens).astype(jnp.float16),
            "prefix_mask": jax.lax.stop_gradient(prefix_mask),
            "state": jax.lax.stop_gradient(observation.state).astype(jnp.float32),
            "actions": jax.lax.stop_gradient(actions).astype(jnp.float32),
        }
        return loss, features

    def replay_prediction(
        self,
        replay: dict[str, at.Array],
        noise: at.Array,
        time: at.Array,
        *,
        gradient_mode: dual_lora.GradientMode = "long",
    ) -> tuple[at.Array, at.Array]:
        return self._predict_from_prefix(
            replay["prefix_tokens"],
            replay["prefix_mask"],
            replay["state"],
            replay["actions"],
            noise,
            time,
            gradient_mode=gradient_mode,
        )

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Array:
        preprocess_rng, diffusion_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        prefix_tokens, prefix_mask, _ = self.embed_prefix(observation)
        noise, time = self.sample_diffusion(diffusion_rng, actions)
        prediction, target = self._predict_from_prefix(
            prefix_tokens, prefix_mask, observation.state, actions, noise, time, gradient_mode="both"
        )
        return jnp.mean(jnp.square(prediction - target), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        step_size = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attention_mask = base_pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        gate_context = self._gate_context(prefix_tokens, prefix_mask)
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attention_mask,
            positions=positions,
            gate_context=gate_context,
            gradient_mode="both",
        )

        def step(carry):
            actions_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, actions_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attention_mask = base_pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attention_mask = jnp.concatenate([prefix_to_suffix_mask, suffix_attention_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attention_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                gate_context=gate_context,
                gradient_mode="both",
            )
            velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return actions_t + step_size * velocity, time + step_size

        def condition(carry):
            _, time = carry
            return time >= -step_size / 2

        actions_0, _ = jax.lax.while_loop(condition, step, (noise, 1.0))
        return actions_0
