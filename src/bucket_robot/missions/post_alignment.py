"""Post-distance alignment shares the main mission's left-pose handoff policy."""

from bucket_robot.missions.recovery import RecoveryGate


class PostAlignmentGate(RecoveryGate):
    """Stop at the left pose, verify the same target, and ignore image centering."""
