"""Thermal telemetry uses Linux millidegrees and expires independently of control."""

from pathlib import Path
import tempfile
import unittest

from bucket_robot.runtime.thermal import ThermalMonitor


class ThermalTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.now = 1.0
        self.monitor = ThermalMonitor(self.root, clock=lambda: self.now)

    def tearDown(self):
        self.folder.cleanup()

    def zone(self, index, kind, temperature):
        path = self.root / ("thermal_zone" + str(index))
        path.mkdir(exist_ok=True)
        (path / "type").write_text(kind)
        (path / "temp").write_text(temperature)

    def test_actual_sensor_types_units_and_unknown_extra_zones(self):
        self.zone(5, "gpu-thermal", "62846\n")
        self.zone(7, "iwlwifi_1", "53000\n")
        self.zone(8, "custom-sensor", "-5000")
        self.monitor.sample()
        rows = {r["type"]: r for r in self.monitor.snapshot()["sensors"]}
        self.assertEqual(rows["gpu-thermal"]["celsius"], 62.846)
        self.assertEqual(rows["iwlwifi_1"]["label"], "Wi-Fi")
        self.assertEqual(rows["custom-sensor"]["celsius"], -5)
        self.assertIsNone(rows["npu-thermal"]["celsius"])

    def test_missing_or_failed_sensor_cannot_reuse_an_old_temperature(self):
        self.zone(0, "soc-thermal", "65000")
        self.zone(6, "npu-thermal", "64000")
        self.monitor.sample()
        (self.root / "thermal_zone6/temp").unlink()
        self.zone(5, "gpu-thermal", "not-a-temperature")
        self.monitor.sample()
        rows = {r["type"]: r for r in self.monitor.snapshot()["sensors"]}
        self.assertEqual(rows["soc-thermal"]["celsius"], 65)
        self.assertIsNone(rows["npu-thermal"]["celsius"])
        self.assertIsNone(rows["gpu-thermal"]["celsius"])

    def test_no_samples_or_stopped_sampler_never_reports_fresh_temperature(self):
        self.assertIsNone(self.monitor.snapshot()["age_s"])
        self.zone(0, "soc-thermal", "65000")
        self.monitor.sample()
        self.now += 3.01
        self.assertTrue(all(r["celsius"] is None for r in self.monitor.snapshot()["sensors"]))
        self.monitor.sample()
        self.assertEqual(self.monitor.snapshot()["sensors"][0]["celsius"], 65)
