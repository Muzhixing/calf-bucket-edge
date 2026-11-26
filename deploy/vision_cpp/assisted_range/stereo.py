"""Compatibility import for the packaged range implementation."""
import sys
from bucket_robot.perception.assisted_range import stereo
sys.modules[__name__] = stereo
