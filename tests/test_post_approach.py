import copy, unittest
from bucket_robot.missions.post_approach import pending_extension_plan


class PostApproachTests(unittest.TestCase):
    def valid(self):
        return dict(
            error=None,
            stop_error=None,
            reason="distance_stop_margin",
            final_state=dict(
                enabled=False,
                state=0,
                system_fault=0,
                motor_fault=0,
                servo_software_deg=131,
                lift_position_mm=50,
                slide_position_mm=0,
            ),
            last_measurement=dict(
                valid=True,
                distance_m=0.223,
                distance_kind="stereo_optical_midpoint_to_bucket_surface",
            ),
        )

    def test_plan_keeps_metric_target_and_blocks_both_actions_without_odometry(self):
        p = pending_extension_plan(self.valid())
        self.assertEqual(p["phase"], "waiting_displacement_measurement")
        self.assertEqual(p["sequence"][0]["absolute_target_mm"], 150)
        self.assertEqual(p["sequence"][1]["relative_target_mm"], 100)
        self.assertFalse(p["C_extension_sent"])
        self.assertFalse(p["chassis_command_sent"])
        self.assertFalse(p["simultaneous"])
        self.assertIsNone(p["displacement_source"])

    def test_distance_first_completion_can_handoff_from_left_front_pose(self):
        approach = self.valid()
        approach["mode"] = "distance_first"
        approach["final_state"]["servo_software_deg"] = 97
        plan = pending_extension_plan(approach)
        self.assertEqual(plan["approach_A_deg"], 97)
        self.assertFalse(plan["C_extension_sent"])
        self.assertFalse(plan["chassis_command_sent"])

    def test_fault_nonarrival_invalid_distance_and_already_extended_C_rejected(self):
        for key, value in [
            ("error", "fault"),
            ("stop_error", "no_ack"),
            ("reason", "drive_budget_exhausted"),
        ]:
            p = self.valid()
            p[key] = value
            with self.assertRaises(ValueError):
                pending_extension_plan(p)
        for key, value in [
            ("enabled", True),
            ("slide_position_mm", 150),
            ("servo_software_deg", 73),
        ]:
            p = self.valid()
            p["final_state"][key] = value
            with self.assertRaises(ValueError):
                pending_extension_plan(p)
        p = self.valid()
        p["last_measurement"]["valid"] = False
        with self.assertRaises(ValueError):
            pending_extension_plan(p)


if __name__ == "__main__":
    unittest.main()
