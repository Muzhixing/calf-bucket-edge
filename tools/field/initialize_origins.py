"""Compatibility imports; startup business logic lives in the installed package."""

from bucket_robot.runtime.preparation import (
    StartupService as StartupService,
    initialize as initialize,
    read_policy as read_policy,
)
from bucket_robot.runtime.startup import derive_wait_profile as derive_wait_profile
