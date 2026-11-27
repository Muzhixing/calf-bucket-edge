import importlib.util
from pathlib import Path
import tempfile
import unittest
import uuid

from bucket_robot.motion.arm import Limits
from bucket_robot.resources import default_reference
from bucket_robot.runtime.startup import PowerCycleStore, StartupPort
from tests.test_power_cycle import HomeMCU
from tests.test_tracking import Sender

spec = importlib.util.spec_from_file_location(
    "field_initialize", Path(__file__).parents[1] / "tools/field/initialize_origins.py"
)
field = importlib.util.module_from_spec(spec)
spec.loader.exec_module(field)


class FieldInitializationTests(unittest.TestCase):
    def setUp(self):
        self.mcu = HomeMCU()
        self.mcu.a = 73
        self.temp = tempfile.TemporaryDirectory()
        self.ledger = Path(self.temp.name) / "state.json"
        self.limits = Limits(default_reference())

    def tearDown(self):
        self.mcu.close()
        self.temp.cleanup()

    def factory(self, limits, log):
        return StartupPort(
            limits, log, port=self.mcu.path, sender_factory=Sender, check_holders=False
        )

    def initialize(self, **kwargs):
        return field.initialize(
            self.limits,
            self.ledger,
            lambda *event: None,
            port_factory=self.factory,
            wait_seconds=0,
            timeout=4,
            **kwargs,
        )

    def test_cold_boot_registers_once_disabled_and_process_restart_reuses(self):
        result = self.initialize()
        self.assertTrue(result["zero_registered_now"])
        self.assertFalse(result["final_state"]["enabled"])
        self.assertEqual([c[8] for c in self.mcu.commands], [0, 6])
        reused = self.initialize()
        self.assertTrue(reused["origins_reused"])
        self.assertEqual([c[8] for c in self.mcu.commands], [0, 6])
        self.assertTrue(all(c[1:3] == bytes([0, 0]) and not c[8] & 1 for c in self.mcu.commands))

    def test_fresh_mcu_boot_after_ready_allows_one_new_registration(self):
        self.initialize()
        self.mcu.zero_flags = False
        self.mcu.fault = 1
        self.mcu.count = 0
        result = self.initialize()
        self.assertTrue(result["zero_registered_now"])
        self.assertEqual([c[8] for c in self.mcu.commands], [0, 6, 0, 6])

    def test_unconfirmed_prior_write_is_not_repeated(self):
        profile = field.derive_wait_profile(self.limits)
        store = PowerCycleStore(self.ledger, profile["wait_seconds"], profile["wait_basis"])
        store.confirm(str(uuid.uuid4()))
        store.phase("disable_pending")
        store.before_zero()
        store.close()
        with self.assertRaisesRegex(RuntimeError, "unconfirmed_no_retry"):
            self.initialize()
        self.assertEqual(self.mcu.commands, [])

    def test_standing_policy_is_explicit_and_validated(self):
        policy = field.read_policy(Path(__file__).parents[1] / "tools/field/startup_policy.json")
        self.assertFalse(policy["require_repeated_position_confirmation"])

    def test_registered_interrupted_pose_returns_without_zero_registration_or_chassis(self):
        self.mcu.zero_flags = True
        self.mcu.fault = 0
        self.mcu.a = 89
        self.mcu.b = self.mcu.target = 6.6
        result = self.initialize(return_home=True)
        self.assertTrue(result["returned_home"])
        self.assertFalse(result["zero_registered_now"])
        self.assertEqual(result["final_state"]["servo_software_deg"], 73)
        self.assertEqual(result["final_state"]["lift_position_mm"], 0)
        self.assertFalse(result["final_state"]["enabled"])
        self.assertTrue(all(c[1:3] == bytes([0, 0]) and not c[8] & 6 for c in self.mcu.commands))
        moves = [list(c[4:7]) for c in self.mcu.commands if c[8] & 1]
        self.assertEqual(moves, [[89, 0, 0], [73, 0, 0]])
        self.mcu.commands.clear()
        self.mcu.processed = 0
        self.assertFalse(self.initialize(return_home=True)["returned_home"])
        self.assertEqual(self.mcu.commands, [])

    def test_disabled_only_initialization_does_not_move_registered_nonhome(self):
        self.mcu.zero_flags = True
        self.mcu.fault = 0
        self.mcu.a = 89
        self.mcu.b = self.mcu.target = 6.6
        with self.assertRaisesRegex(RuntimeError, "registered_home_not_ready"):
            self.initialize()
        self.assertEqual(self.mcu.commands, [])

    def test_return_retracts_C_before_B_then_A(self):
        self.mcu.zero_flags = True
        self.mcu.fault = 0
        self.mcu.a = 81
        self.mcu.c = 150
        self.mcu.b = self.mcu.target = 21
        result = self.initialize(return_home=True)
        moves = [list(c[4:7]) for c in self.mcu.commands if c[8] & 1]
        self.assertEqual(moves, [[81, 21, 0], [81, 0, 0], [73, 0, 0]])
        for index, command in enumerate(self.mcu.commands):
            if index and command[8] & 1:
                self.assertEqual(self.mcu.commands[index - 1][8], 0)
        self.assertEqual(result["final_state"]["slide_position_mm"], 0)
        self.assertTrue(all(c[1:3] == bytes([0, 0]) and not c[8] & 6 for c in self.mcu.commands))

    def test_registered_fault_prevents_return_before_any_write(self):
        self.mcu.zero_flags = True
        self.mcu.fault = 3
        self.mcu.a = 89
        with self.assertRaisesRegex(RuntimeError, "firmware_fault"):
            self.initialize(return_home=True)
        self.assertEqual(self.mcu.commands, [])
