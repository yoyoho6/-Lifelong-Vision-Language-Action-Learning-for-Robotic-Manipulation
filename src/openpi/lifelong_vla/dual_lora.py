"""Dual-timescale LoRA layers with a sample-conditioned shared gate.

The implementation mirrors ``openpi.models.lora`` but keeps two low-rank
residuals.  Parameter names deliberately contain ``lora`` so OpenPI's released
checkpoint loader initializes them when loading a checkpoint without adapters.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at

GradientMode = Literal["short", "long", "both"]


@dataclasses.dataclass(frozen=True)
class DualLoRAConfig:
    short_rank: int = 16
    long_rank: int = 16
    alpha: float = 16.0
    gate_context_dim: int = 2048
    gate_bias: float = 0.0
    rslora: bool = False
    axes: tuple[int, int] = (-2, -1)
    label: str = "L"

    def __post_init__(self) -> None:
        if self.short_rank <= 0 or self.long_rank <= 0:
            raise ValueError("Both LoRA ranks must be positive.")
        if self.gate_context_dim <= 0:
            raise ValueError("gate_context_dim must be positive.")

    def scale(self, rank: int) -> float:
        return self.alpha / math.sqrt(rank) if self.rslora else self.alpha / rank


def _select_gradient(x: at.Array, mode: GradientMode, pathway: Literal["short", "long"]) -> at.Array:
    if mode not in ("short", "long", "both"):
        raise ValueError(f"Unknown gradient mode: {mode!r}")
    return x if mode in (pathway, "both") else jax.lax.stop_gradient(x)


def _gate(
    context: at.Array,
    kernel: at.Array,
    bias: at.Array,
    output_ndim: int,
    batch_axis: int = 0,
) -> at.Array:
    if context.ndim != 2:
        raise ValueError(f"Gate context must have shape [batch, dim], got {context.shape}.")
    alpha = jax.nn.sigmoid(jnp.einsum("bd,d->b", context.astype(jnp.float32), kernel) + bias)
    shape = [1] * output_ndim
    shape[batch_axis] = alpha.shape[0]
    return alpha.reshape(tuple(shape))


class Einsum(nn.Module):
    """Einsum with short/long LoRA residuals and one sample-level gate."""

    shape: tuple[int, ...]
    config: DualLoRAConfig
    init_fn: nn.initializers.Initializer = nn.initializers.zeros

    def setup(self) -> None:
        self.w = self.param("w", self.init_fn, self.shape)
        self._init_path("short", self.config.short_rank)
        self._init_path("long", self.config.long_rank)
        self.lora_gate_kernel = self.param(
            "lora_gate_kernel", nn.initializers.zeros_init(), (self.config.gate_context_dim,)
        )
        self.lora_gate_bias = self.param("lora_gate_bias", nn.initializers.constant(self.config.gate_bias), ())

    def _init_path(self, name: str, rank: int) -> None:
        shape_a, shape_b = list(self.shape), list(self.shape)
        shape_a[self.config.axes[1]] = rank
        shape_b[self.config.axes[0]] = rank
        setattr(
            self,
            f"lora_{name}_a",
            self.param(f"lora_{name}_a", nn.initializers.normal(stddev=0.01), tuple(shape_a)),
        )
        setattr(
            self,
            f"lora_{name}_b",
            self.param(f"lora_{name}_b", nn.initializers.zeros_init(), tuple(shape_b)),
        )

    @nn.compact
    def __call__(self, equation: str, x: at.Array, gate_context: at.Array, gradient_mode: GradientMode = "both"):
        dtype = x.dtype
        result = jnp.einsum(equation, x, self.w.astype(dtype))
        short = self._path(equation, x, "short", self.config.short_rank, gradient_mode)
        long = self._path(equation, x, "long", self.config.long_rank, gradient_mode)
        output_labels = equation.rsplit("->", maxsplit=1)[1]
        alpha = _gate(
            gate_context,
            self.lora_gate_kernel,
            self.lora_gate_bias,
            result.ndim,
            batch_axis=output_labels.index("B"),
        ).astype(dtype)
        return result + (1.0 - alpha) * short + alpha * long

    def _path(self, equation: str, x: at.Array, name: str, rank: int, mode: GradientMode) -> at.Array:
        equation_a, equation_b = self._make_lora_equations(equation)
        a = _select_gradient(getattr(self, f"lora_{name}_a"), mode, name)
        b = _select_gradient(getattr(self, f"lora_{name}_b"), mode, name)
        value = jnp.einsum(equation_a, x, a.astype(x.dtype))
        value = jnp.einsum(equation_b, value, b.astype(x.dtype))
        return value * self.config.scale(rank)

    def _make_lora_equations(self, equation: str) -> tuple[str, str]:
        if self.config.label in equation:
            raise ValueError(f"LoRA label already appears in equation: {equation}")
        match = re.fullmatch("(.*),(.*)->(.*)", equation)
        if match is None:
            raise ValueError(f"Unsupported einsum equation: {equation}")
        lhs, rhs, out = match.groups()
        a_label, b_label = (rhs[index] for index in self.config.axes)
        label = self.config.label
        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        return f"{lhs},{a_rhs}->{a_out}", f"{a_out},{rhs.replace(a_label, label)}->{out}"


class FeedForward(nn.Module):
    """Gemma feed-forward layer with dual-timescale LoRA."""

    features: int
    hidden_dim: int
    config: DualLoRAConfig

    def setup(self) -> None:
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear", nn.initializers.lecun_normal(in_axis=-2, out_axis=-1), (self.hidden_dim, self.features)
        )
        self._init_path("short", self.config.short_rank)
        self._init_path("long", self.config.long_rank)
        self.lora_gate_kernel = self.param(
            "lora_gate_kernel", nn.initializers.zeros_init(), (self.config.gate_context_dim,)
        )
        self.lora_gate_bias = self.param("lora_gate_bias", nn.initializers.constant(self.config.gate_bias), ())

    def _init_path(self, name: str, rank: int) -> None:
        normal = nn.initializers.normal(stddev=0.01)
        zero = nn.initializers.zeros_init()
        setattr(self, f"gating_lora_{name}_a", self.param(f"gating_lora_{name}_a", normal, (2, self.features, rank)))
        setattr(self, f"gating_lora_{name}_b", self.param(f"gating_lora_{name}_b", zero, (2, rank, self.hidden_dim)))
        setattr(self, f"linear_lora_{name}_a", self.param(f"linear_lora_{name}_a", normal, (self.hidden_dim, rank)))
        setattr(self, f"linear_lora_{name}_b", self.param(f"linear_lora_{name}_b", zero, (rank, self.features)))

    @nn.compact
    def __call__(self, x: at.Array, gate_context: at.Array, gradient_mode: GradientMode = "both") -> at.Array:
        dtype = x.dtype
        alpha = _gate(gate_context, self.lora_gate_kernel, self.lora_gate_bias, x.ndim).astype(dtype)
        gate = self._linear(x, self.w_gating[0], "gating", 0, alpha, gradient_mode)
        value = self._linear(x, self.w_gating[1], "gating", 1, alpha, gradient_mode)
        hidden = nn.gelu(gate) * value
        return self._linear(hidden, self.w_linear, "linear", None, alpha, gradient_mode)

    def _linear(self, x, weight, kind: str, branch: int | None, alpha, mode: GradientMode):
        base = jnp.dot(x, weight.astype(x.dtype))
        short_a = getattr(self, f"{kind}_lora_short_a")
        short_b = getattr(self, f"{kind}_lora_short_b")
        long_a = getattr(self, f"{kind}_lora_long_a")
        long_b = getattr(self, f"{kind}_lora_long_b")
        if branch is not None:
            short_a, short_b, long_a, long_b = (value[branch] for value in (short_a, short_b, long_a, long_b))
        short = jnp.dot(
            jnp.dot(x, _select_gradient(short_a, mode, "short").astype(x.dtype)),
            _select_gradient(short_b, mode, "short").astype(x.dtype),
        ) * self.config.scale(self.config.short_rank)
        long = jnp.dot(
            jnp.dot(x, _select_gradient(long_a, mode, "long").astype(x.dtype)),
            _select_gradient(long_b, mode, "long").astype(x.dtype),
        ) * self.config.scale(self.config.long_rank)
        return base + (1.0 - alpha) * short + alpha * long
