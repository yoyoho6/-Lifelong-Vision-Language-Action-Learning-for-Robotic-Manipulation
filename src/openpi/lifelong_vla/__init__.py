"""LifelongVLA research branch for continual pi0 adaptation.

This package is intentionally isolated from the default OpenPI model and trainer.
Importing :mod:`openpi` does not enable any of these components.
"""

from openpi.lifelong_vla.config import LifelongPi0Config
from openpi.lifelong_vla.replay_buffer import FeatureReplayBuffer

__all__ = ["FeatureReplayBuffer", "LifelongPi0Config"]
