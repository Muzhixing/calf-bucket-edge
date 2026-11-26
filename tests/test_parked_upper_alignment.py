import tempfile
import time
import unittest
from pathlib import Path

from bucket_robot.motion.arm import Limits
from bucket_robot.motion.combined import ABCombinedPort
from bucket_robot.missions.vertical import align_parked
from bucket_robot.resources import default_reference
from bucket_robot.tracking.controller import TrackingController
from bucket_robot.tracking.policy import FollowPolicy
from tests.test_power_cycle import HomeMCU
from tests.test_tracking import Sender


class ParkedUpperAlignmentTests(unittest.TestCase):
    def test_shared_controller_adjusts_only_B_and_stops_with_integer_position(self):
        mcu = HomeMCU()
        port = None
        try:
            mcu.a, mcu.b, mcu.target, mcu.c = 100, 0, 0, 0
            mcu.zero_flags, mcu.fault, mcu.speed = True, 0, 5
            limits = Limits(default_reference())
            port = ABCombinedPort(
                limits, lambda *e: None, port=mcu.path, sender_factory=Sender, check_holders=False
            )

            class Observer:
                def get(self):
                    top = 380 - 5 * mcu.b
                    return dict(
                        identity=1,
                        metadata={},
                        target=dict(
                            timestamp=time.monotonic(),
                            box=[500, top, 780, top + 200],
                            offset_px=[0, top + 100 - 360],
                        ),
                    )

            with tempfile.TemporaryDirectory() as folder:
                controller = TrackingController(FollowPolicy(limits, 700, 700, 640, 360))
                result = align_parked(
                    Observer(), port, controller, Path(folder), lambda *e: None, timeout=12
                )
                self.assertTrue(result["moved"])
                self.assertFalse(result["final_state"]["enabled"])
                b = result["final_state"]["lift_position_mm"]
                self.assertEqual(b, round(b))
                self.assertGreater(b, 0)
                self.assertTrue(
                    all(f[1:3] == bytes([0, 0]) and f[4] == 100 and f[6] == 0 for f in mcu.commands)
                )
                self.assertEqual((limits.a_min, limits.a_max), (0, 160))
        finally:
            if port:
                port.shutdown()
            mcu.close()
