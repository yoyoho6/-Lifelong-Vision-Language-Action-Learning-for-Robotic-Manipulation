"""Model configuration for the opt-in LifelongVLA branch."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import pi0_config
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.lifelong_vla.model import LifelongPi0


@dataclasses.dataclass(frozen=True)
class LifelongPi0Config(pi0_config.Pi0Config):
    """pi0 with dual LoRA in both the VLM and action expert."""

    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    short_rank: int = 16
    long_rank: int = 16
    lora_alpha: float = 16.0
    gate_bias: float = 0.0

    @override
    def create(self, rng: at.KeyArrayLike) -> LifelongPi0:
        from openpi.lifelong_vla.model import LifelongPi0

        return LifelongPi0(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze the entire backbone; only dual LoRA factors and gates train."""
        return nnx.Not(nnx_utils.PathRegex(".*lora_(short|long|gate).*"))

    @property
    def adapter_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx_utils.PathRegex(".*lora_(short|long|gate).*"))
