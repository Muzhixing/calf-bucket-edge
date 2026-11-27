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

    def test_plan_keeps_geometry_as_diagnostic_and_requires_explicit_timed_execution(self):
        p = pending_extension_plan(self.valid())
        self.assertEqual(p["phase"], "planned")
        self.assertEqual(p["sequence"][-1]["absolute_target_mm"], 150)
        self.assertAlmostEqual(p["sequence"][2]["diagnostic_half_left_mm"], 111.5)
        self.assertFalse(p["C_extension_sent"])
        self.assertFalse(p["chassis_command_sent"])
        self.assertFalse(p["simultaneous"])
        self.assertEqual(p["displacement_source"], "operator_timing_no_odometry")
        self.assertTrue(p["execution_requires_explicit_opt_in"])
        action = p["sequence"][2]
        self.assertEqual(action["execution"]["duration_s"], 2)
        self.assertEqual(action["execution"]["speed_percent"], 40)
        self.assertFalse(action["require_metric_displacement_feedback"])

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
