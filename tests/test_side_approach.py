import math
import unittest
from bucket_robot.missions.approach import ApproachGate


class ApproachGateTests(unittest.TestCase):
    def measurement(self, frame=1, distance=0.30, **changes):
        return dict(
            dict(
                frame_id=frame,
                capture_time=10.0,
                valid=True,
                distance_m=distance,
                lower_m=distance - 0.005,
                distance_kind="stereo_optical_midpoint_to_bucket_surface",
            ),
            **changes,
        )

    def test_twenty_cm_stops_immediately(self):
        self.assertEqual(
            ApproachGate().decide(10.1, self.measurement(distance=0.20)), "target_distance_reached"
        )
        self.assertEqual(
            ApproachGate().decide(10.1, self.measurement(distance=0.21)), "distance_stop_margin"
        )

    def test_invalid_depth_does_not_keep_previous_motion(self):
        gate = ApproachGate()
        self.assertEqual(gate.decide(10.1, self.measurement()), "left_pulse")
        self.assertEqual(
            gate.decide(10.2, self.measurement(2, valid=False, distance_m=None)),
            "distance_unavailable",
        )

    def test_age_nan_basis_and_reuse(self):
        for changes, expected in [
            ({"capture_time": 7.0}, "measurement_too_old"),
            ({"distance_m": math.nan}, "invalid_distance"),
            ({"distance_kind": "camera_forward_Z_bucket_body"}, "distance_basis_mismatch"),
        ]:
            self.assertEqual(ApproachGate().decide(10.1, self.measurement(**changes)), expected)
        gate = ApproachGate()
        gate.decide(10.1, self.measurement())
        with self.assertRaises(RuntimeError):
            gate.decide(10.2, self.measurement())

    def test_distance_jump_and_cumulative_budget(self):
        gate = ApproachGate()
        gate.decide(10.1, self.measurement(distance=0.40))
        self.assertEqual(
            gate.decide(10.2, self.measurement(2, distance=0.30)), "unexpected_distance_change"
        )
        gate = ApproachGate(max_drive=0.6)
        gate.decide(10.1, self.measurement())
        gate.drive_seconds = 0.6
        self.assertEqual(
            gate.decide(10.2, self.measurement(2, distance=0.295)), "drive_budget_exhausted"
        )

    def test_slow_result_requires_stationary_proof(self):
        self.assertEqual(ApproachGate().decide(13.0, self.measurement()), "measurement_too_old")
        self.assertEqual(
            ApproachGate().decide(13.0, self.measurement(stationary_verified=True)), "left_pulse"
        )
        self.assertEqual(
            ApproachGate().decide(16.0, self.measurement(stationary_verified=True)),
            "measurement_too_old",
        )

    def test_no_move_before_left_alignment_or_on_target_swap(self):
        gate = ApproachGate()
        obs = dict(identity=1, target=dict(timestamp=10.0, offset_px=[0, 0]))
        with self.assertRaises(RuntimeError):
            gate.target_check(10.1, obs, 73)
        gate.target_check(10.1, obs, 132)
        with self.assertRaises(RuntimeError):
            gate.target_check(10.1, dict(obs, identity=2), 132)
        with self.assertRaises(RuntimeError):
            gate.target_check(11.0, obs, 132)

    def test_vertical_crop_needs_measured_horizontal_extent(self):
        t = dict(
            timestamp=10.0,
            offset_px=[0, 240],
            partial_visible=True,
            horizontal_bearing_valid=True,
            bearing_evidence="current_detection_and_visible_appearance",
        )
        ApproachGate().target_check(10.1, dict(identity=1, target=t), 132)
        for changes in ({"horizontal_bearing_valid": False}, {"bearing_evidence": "prediction"}):
            with self.assertRaises(RuntimeError):
                ApproachGate().target_check(10.1, dict(identity=1, target=dict(t, **changes)), 132)


# Exercise the actual UART owner against a PTY, including disable acknowledgement.
class SidePortTests(unittest.TestCase):
    def test_heading_fault_still_confirms_stop(self):
        import json, tempfile, time
        from pathlib import Path
        from bucket_robot.missions.approach import SidePort, HeldArmLimits
        from tests.test_track_forward_probe import MCU, Sender

        m = MCU()
        m.a = 132
        p = None
        try:
            with tempfile.TemporaryDirectory() as folder:
                ref = Path(folder) / "ref.json"
                ref.write_text(
                    json.dumps({"control_policy": {"original_sender_module_on_rk3588": "unused"}})
                )
                p = SidePort(
                    HeldArmLimits(ref),
                    lambda *args: None,
                    port=m.path,
                    sender_factory=Sender,
                    check_holders=False,
                )
                p.start_pulse("left", 8, 0.6)
                m.yaw = 40
                with self.assertRaisesRegex(RuntimeError, "heading_drift"):
                    until = time.monotonic() + 0.5
                    while time.monotonic() < until:
                        p.tick()
                        time.sleep(0.01)
                self.assertIsNone(p.shutdown())
                self.assertFalse(p.state["enabled"])
                self.assertEqual(p.state["state"], 0)
        finally:
            if p:
                p.shutdown()
            m.close()

    def test_lateral_packet_and_confirmed_stop(self):
        import json, tempfile, time
        from pathlib import Path
        from bucket_robot.missions.approach import SidePort, HeldArmLimits
        from tests.test_track_forward_probe import MCU, Sender

        class SideMCU(MCU):
            def packet(self):
                f = bytearray(super().packet())
                if self.enabled and self.commands and self.commands[-1][2]:
                    f[4] = 3
                f[18] = 0
                for v in f[1:18]:
                    f[18] ^= v
                return bytes(f)

        m = SideMCU()
        m.a = 132
        p = None
        try:
            with tempfile.TemporaryDirectory() as folder:
                ref = Path(folder) / "reference.json"
                ref.write_text(
                    json.dumps({"control_policy": {"original_sender_module_on_rk3588": "unused"}})
                )
                p = SidePort(
                    HeldArmLimits(ref),
                    lambda *x: None,
                    port=m.path,
                    sender_factory=Sender,
                    check_holders=False,
                )
                p.start_pulse("left", 8, 0.6)
                until = time.monotonic() + 2
                while p.phase != "idle" and time.monotonic() < until:
                    p.tick()
                    time.sleep(0.01)
                self.assertEqual(p.phase, "idle")
                self.assertTrue(p.drive_observed)
                self.assertFalse(p.state["enabled"])
                self.assertEqual(m.commands[0][1:3], bytes([0, 8]))
                self.assertEqual(m.commands[-1][1:3], bytes([0, 0]))
                self.assertEqual(m.commands[-1][8], 0)
                self.assertTrue(all(f[4:8] == bytes([132, 0, 0, 0]) for f in m.commands))
        finally:
            if p:
                p.shutdown()
            m.close()


if __name__ == "__main__":
    unittest.main()
