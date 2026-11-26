import math
import unittest
from bucket_robot.missions.post_geometry import camera_triangle, alignment_extension_plan


class PostGeometryTests(unittest.TestCase):
    def test_projection_uses_physical_reference_scale_not_command_difference(self):
        g = camera_triangle(0.2, 124)
        self.assertAlmostEqual(g["alpha_estimate_deg"], 90 * 7 / 58)
        self.assertAlmostEqual(g["a_forward_mm"], 37.69, delta=0.1)
        self.assertAlmostEqual(g["b_left_mm"], 196.42, delta=0.1)
        self.assertAlmostEqual(math.hypot(g["a_forward_mm"], g["b_left_mm"]), 200)
        self.assertFalse(g["physical_angle_calibrated"])

    def test_forward_and_left_endpoints_and_half_sideway(self):
        forward = camera_triangle(0.2, 73)
        left = camera_triangle(0.2, 131)
        self.assertAlmostEqual(forward["a_forward_mm"], 200)
        self.assertAlmostEqual(forward["b_left_mm"], 0)
        self.assertAlmostEqual(left["a_forward_mm"], 0)
        self.assertAlmostEqual(left["b_left_mm"], 200)
        p = alignment_extension_plan(0.2, 124)
        self.assertAlmostEqual(
            p["sequence"][2]["relative_target_mm"], p["geometry"]["b_left_mm"] / 2
        )
        self.assertEqual(p["sequence"][-1]["absolute_target_mm"], 150)
        self.assertFalse(p["chassis_command_sent"])
        self.assertFalse(p["C_extension_sent"])
        self.assertEqual(p["phase"], "waiting_displacement_measurement")
        self.assertEqual(alignment_extension_plan(0.2, 132)["sequence"][0]["relative_target_mm"], 0)

    def test_invalid_and_nonfinite_data_cannot_produce_actions(self):
        for d, a in [
            (0, 124),
            (-0.2, 124),
            (0.2, 72),
            (0.2, 140),
            (float("nan"), 124),
            (0.2, float("inf")),
        ]:
            with self.assertRaises(ValueError):
                camera_triangle(d, a)
