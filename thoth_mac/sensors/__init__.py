"""macOS-specific sensor implementations.

Importing this package auto-registers each sensor with the
thoth_core SensorRegistry.
"""

from . import camera, microphone, imu, csi

__all__ = ["camera", "microphone", "imu", "csi"]
