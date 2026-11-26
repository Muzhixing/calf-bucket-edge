"""Left-pose completion with the operator-accepted image-centering tolerance."""

from bucket_robot.missions.recovery import RecoveryGate


class PostAlignmentGate(RecoveryGate):
    def __init__(self, seconds):
        super().__init__(seconds)
        self.left_latched = False
        self.confirm_since = self.confirm_stamp = None
        self.confirm_frames = 0

    def evaluate(self, now, target, identity, a, settled, **kwargs):
        if self.left_latched:
            kwargs["motion_permitted"] = False
        velocity, reason = super().evaluate(now, target, identity, a, settled, **kwargs)
        if self.latched:
            return velocity, reason
        commanded = kwargs.get("commanded_a", a)
        if self.allow_a and a >= 131 and commanded >= 131 and settled:
            self.left_latched = True
        if not self.left_latched:
            return velocity, reason
        # Once at the left pose, never resume forward motion just because the
        # image center changes. Only same-target freshness and stop ACK remain.
        self.speed = 0.0
        self.velocity = 0
        self.disable = True
        valid = (
            self.allow_a and 131 <= a <= 132 and settled and kwargs.get("chassis_stopped", False)
        )
        if valid:
            stamp = target["timestamp"]
            if self.confirm_since is None:
                self.confirm_since = stamp
            if stamp != self.confirm_stamp:
                self.confirm_stamp = stamp
                self.confirm_frames += 1
            if self.confirm_frames >= 3 and stamp - self.confirm_since >= 0.3:
                self.phase = "aligned"
                self.reason = "left_reference_reached"
                return 0, self.reason
        else:
            self.confirm_since = self.confirm_stamp = None
            self.confirm_frames = 0
        self.phase = "verifying_left_alignment"
        self.reason = "waiting_stopped_same_target_not_image_center"
        return 0, None
