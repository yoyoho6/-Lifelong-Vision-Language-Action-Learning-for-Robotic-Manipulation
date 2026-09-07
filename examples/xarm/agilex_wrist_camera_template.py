"""Template adapter for a wrist camera vendor SDK.

Fill this file with the actual SDK calls from your wrist camera vendor.
Then run:

PYTHONPATH=packages/openpi-client/src python examples/xarm/main.py \
  --wrist-camera-backend sdk \
  --wrist-camera-factory examples.xarm.agilex_wrist_camera_template:create_camera
"""

from __future__ import annotations

import numpy as np


class AgilexWristCamera:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        raise NotImplementedError(
            "Replace AgilexWristCamera with the actual wrist camera SDK initialization code."
        )

    def read(self) -> np.ndarray:
        raise NotImplementedError(
            "Return a single HWC uint8 RGB image using the wrist camera SDK."
        )

    def close(self) -> None:
        pass


def create_camera(**kwargs) -> AgilexWristCamera:
    return AgilexWristCamera(**kwargs)
