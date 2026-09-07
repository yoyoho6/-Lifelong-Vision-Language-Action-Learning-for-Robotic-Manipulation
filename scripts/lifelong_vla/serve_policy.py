"""Serve a checkpoint produced by the LifelongVLA training branch."""

from __future__ import annotations

import dataclasses
import logging
import socket

import tyro

from openpi.lifelong_vla.config import LifelongPi0Config
from openpi.models import pi0_config
from openpi.policies import policy as _policy
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


@dataclasses.dataclass(frozen=True)
class Args:
    base_config: str
    checkpoint_dir: str
    short_rank: int = 16
    long_rank: int = 16
    lora_alpha: float = 16.0
    gate_bias: float = 0.0
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False


def main(args: Args) -> None:
    base = training_config.get_config(args.base_config)
    if not isinstance(base.model, pi0_config.Pi0Config):
        raise TypeError("The base configuration must use Pi0Config.")
    model = LifelongPi0Config(
        dtype=base.model.dtype,
        paligemma_variant=str(base.model.paligemma_variant).removesuffix("_lora"),
        action_expert_variant=str(base.model.action_expert_variant).removesuffix("_lora"),
        action_dim=base.model.action_dim,
        action_horizon=base.model.action_horizon,
        max_token_len=base.model.max_token_len,
        pi05=base.model.pi05,
        discrete_state_input=base.model.discrete_state_input,
        short_rank=args.short_rank,
        long_rank=args.long_rank,
        lora_alpha=args.lora_alpha,
        gate_bias=args.gate_bias,
    )
    config = dataclasses.replace(base, model=model)
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir, default_prompt=args.default_prompt)
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    logging.info("Serving LifelongVLA on %s:%d", hostname, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
