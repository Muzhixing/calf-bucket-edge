"""Effective mission configuration; no control calculations or hardware access."""

from dataclasses import asdict, dataclass
from importlib.resources import files
import json
from bucket_robot.missions.recovery import RecoverySettings
from bucket_robot.missions.timed_translate import TimedTranslate


@dataclass(frozen=True)
class MissionProfile:
    name: str
    cruise_speed: int
    track_b: bool
    a_priority_departure: bool
    distance_first: bool
    recover_loss: bool
    execute_post: bool
    aim: str = "bucket_center"
    chassis_stop_before_B: bool = True
    decelerate_distance_m: float = 0.40
    stop_distance_m: tuple = (0.25, 0.20)

    @classmethod
    def from_options(cls, args):
        return cls(
            args.profile,
            args.cruise_speed,
            args.track_b,
            args.a_start_only,
            args.distance_first,
            args.recover_loss,
            args.execute_post_geometry,
        )

    def as_dict(self):
        return dict(asdict(self), post_translate=TimedTranslate().as_dict())

    def recovery_settings(self):
        return RecoverySettings(
            cruise=self.cruise_speed,
            acceleration_per_s=40 if self.cruise_speed > 20 else 8,
            deceleration_per_s=80 if self.cruise_speed > 20 else 10,
            initial_confirm_s=0 if self.a_priority_departure else 0.7,
            initial_confirm_frames=1 if self.a_priority_departure else 3,
        )


PROFILE_CATALOG = json.loads(
    files("bucket_robot.resources").joinpath("mission_profiles.json").read_text()
)
PRODUCTION_DEFAULTS = PROFILE_CATALOG["profiles"]["production-center"]
