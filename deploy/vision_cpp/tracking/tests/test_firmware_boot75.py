"""New A75 firmware uses the existing at-most-once timed homing workflow."""
import json
from pathlib import Path
import time
import unittest
import uuid

from arm import Limits, reset_telemetry
from startup import PowerCycleStore, RestartEvidence, StartupPort, StartupSequence
from test_tracking import Reference, Sender
from test_power_cycle import HomeMCU
from test_auto_restart import BOOT


class Firmware75Tests(unittest.TestCase):
    home_angle=75
    def setUp(self):
        self.ref=Reference()
        config=json.loads(self.ref.path.read_text())
        config['mechanisms']={'A':{'structural_zero_command':self.home_angle}}
        config['power_cycle_startup']={'whole_power_on_pose':[self.home_angle,0,0],'fixed_wait_seconds':8}
        config['operating_envelope'].update(A_command_degrees=[0,160],C_trial_hold_mm=0)
        self.ref.path.write_text(json.dumps(config));self.limits=Limits(self.ref.path)
        self.mcu=HomeMCU();self.mcu.a=self.home_angle
        self.store=PowerCycleStore(Path(self.ref.directory.name)/'state.json',8,'existing observed homing plus margin',boot_id='test')
        self.seq=None

    def tearDown(self):
        if self.seq:self.seq.close(None if self.seq.done else 'test_end')
        self.store.close();self.mcu.close();self.ref.directory.cleanup()

    def make_sequence(self):
        self.store.confirm(str(uuid.uuid4()))
        factory=lambda limits,log:StartupPort(limits,log,port=self.mcu.path,sender_factory=Sender,check_holders=False)
        self.seq=StartupSequence(self.store,self.limits,lambda *a:None,factory)
        return self.seq

    def finish(self):
        self.seq.started-=9
        deadline=time.monotonic()+7
        while not self.seq.done and time.monotonic()<deadline:self.seq.tick();time.sleep(.01)
        self.assertTrue(self.seq.done,self.seq.stage)

    def test_boot_signature_is_configured_and_requires_fresh_stable_batches(self):
        boot={**BOOT,'servo_software_deg':self.home_angle}
        self.assertFalse(reset_telemetry(boot))
        self.assertTrue(reset_telemetry(boot,self.limits.boot_pose))
        self.assertFalse(reset_telemetry(BOOT,self.limits.boot_pose))
        evidence=RestartEvidence(self.limits.boot_pose)
        for now in (1,1.25,1.5):self.assertFalse(evidence.observe([boot]*5,now))
        self.assertTrue(evidence.observe([boot],1.75))
        self.assertFalse(evidence.observe([{**boot,'lift_zero_confirmed':True}],2))

    def test_wait_then_register_once_without_redundant_A_or_wheel_enable(self):
        seq=self.make_sequence()
        deadline=time.monotonic()+1.8
        while time.monotonic()<deadline:seq.tick();time.sleep(.01)
        self.assertEqual(self.mcu.commands,[])
        self.finish()
        self.assertEqual([f[8] for f in self.mcu.commands],[0,6])
        self.assertTrue(all(list(f[4:7])==[self.home_angle,0,0] for f in self.mcu.commands))
        self.assertTrue(self.store.ready);self.assertFalse(self.mcu.enabled)
        seq.close();self.store.close()
        self.store=PowerCycleStore(Path(self.ref.directory.name)/'state.json',8,'test',boot_id='test')
        self.assertTrue(self.store.ready)
        with self.assertRaises(RuntimeError):self.store.before_zero()

    def test_existing_A_sweep_still_finishes_at75_with_BC_stationary(self):
        self.limits.startup_policy['a_startup_selftest']=True
        self.make_sequence();self.finish()
        enabled=[list(f[4:7]) for f in self.mcu.commands if f[8]&1]
        self.assertEqual(enabled,[[0,0,0],[160,0,0],[self.home_angle,0,0]])
        self.assertEqual(sum(f[8]==6 for f in self.mcu.commands),1)
        self.assertTrue(self.store.ready);self.assertFalse(self.mcu.enabled)


class Firmware73Tests(Firmware75Tests):
    home_angle=73
