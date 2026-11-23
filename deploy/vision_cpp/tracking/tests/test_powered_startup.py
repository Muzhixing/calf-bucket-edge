"""Already-powered MCU starts tracking from live origins without re-zeroing."""
import time
import unittest

from arm import pose
from startup import PowerCycleStore, RegisteredHomeEvidence, registered_home
import test_auto_restart as auto


HOME=dict(enabled=False,state=0,result=0,lift_zero_confirmed=True,slide_zero_confirmed=True,
          motor_fault=0,system_fault=0,angle_valid=True,heading_reference_established=True,
          yaw_deg=0.,servo_software_deg=73,lift_position_mm=0,slide_position_mm=0,
          command_count_mod256=2,uart_error_count_mod256=0)


class PoweredEvidenceTests(unittest.TestCase):
    def test_requires_stable_fresh_home_and_both_origins(self):
        self.assertTrue(registered_home(HOME,73))
        for key,value in [('enabled',True),('lift_zero_confirmed',False),('slide_zero_confirmed',False),
                          ('system_fault',1),('motor_fault',1),('angle_valid',False),
                          ('heading_reference_established',False),('yaw_deg',float('nan')),
                          ('servo_software_deg',75),('lift_position_mm',1),('slide_position_mm',1),
                          ('state',4),('result',6)]:
            self.assertFalse(registered_home({**HOME,key:value},73),key)
        e=RegisteredHomeEvidence(73)
        self.assertFalse(e.observe([HOME]*30,1))
        self.assertFalse(e.observe([HOME]*30,1))
        self.assertFalse(e.observe([HOME],1.25));self.assertFalse(e.observe([HOME],1.5))
        self.assertTrue(e.observe([HOME],1.75))
        self.assertFalse(e.observe([],2.5))
        self.assertFalse(e.observe([HOME],2.75))
        for key in ('command_count_mod256','uart_error_count_mod256'):
            e=RegisteredHomeEvidence(73)
            for i in range(6):self.assertFalse(e.observe([{**HOME,key:i}],1+i*.25))
        e=RegisteredHomeEvidence(73)
        self.assertFalse(e.observe([HOME,{**HOME,'command_count_mod256':3}],1))
        self.assertEqual(e.samples,0)


class PoweredStartupTests(unittest.TestCase):
    # Reuse the serial simulator and session runner, not the parent test cases.
    tearDown=auto.AutoRestartTests.tearDown
    port=auto.AutoRestartTests.port
    start=auto.AutoRestartTests.start
    spin=auto.AutoRestartTests.spin

    def setUp(self):
        auto.AutoRestartTests.setUp(self)
        self.ref.limits.a_home=73;self.ref.limits.boot_pose=[73,0,0]
        self.ref.limits.a_min=0;self.ref.limits.a_max=160;self.ref.limits.c=0
        self.ref.limits.startup_policy['a_startup_selftest']=True
        self.mcu.a=73;self.mcu.zero_flags=True;self.mcu.fault=0;self.mcu.count=2
        # Reproduce the real stale ledger. A 30-second reset wait must not apply.
        self.store.wait_seconds=30
        self.store.phase('uncertain',zero_attempted=True,origins_observed=True,
                         reason='tracking_state_uncertain:telemetry_timeout')

    def test_powered_controller_selftests_and_tracks_without_reset_or_zero(self):
        s=self.start()
        self.spin(lambda:self.store.ready and s.active,timeout=12,fast=False)
        enabled=[list(f[4:7]) for f in self.mcu.commands if f[8]&1]
        self.assertEqual(enabled,[[0,0,0],[160,0,0],[73,0,0]])
        self.assertFalse(any(f[8]==6 for f in self.mcu.commands))
        self.assertFalse(self.store.data['zero_attempted'])
        self.assertTrue(self.store.data['reuse_existing_origins'])
        self.assertEqual(self.store.data['wait_seconds_used'],0)
        self.assertEqual(pose(s.arm['telemetry']),[73,0,0])
        # A subsequent normal app restart reuses ready state without another sweep.
        s.stop_event.set();self.worker.join(4);self.assertFalse(self.worker.is_alive());self.worker=None
        self.store.close();before=len(self.mcu.commands)
        self.store=PowerCycleStore(self.path,30,'test',boot_id='test')
        self.assertTrue(self.store.ready)
        s=self.start();self.spin(lambda:s.active and s.arm['telemetry'] is not None,fast=False)
        self.assertFalse(any(f[8]&7 for f in self.mcu.commands[before:]))
        self.assertIsNone(s.startup_sequence)

    def test_live_origins_do_not_bypass_visual_checks(self):
        gate={'ready':False};s=self.start();s.startup_checks=lambda:dict(gate)
        self.spin(lambda:s.startup_sequence is not None,fast=False)
        time.sleep(1.8)
        self.assertFalse(self.mcu.commands);self.assertFalse(s.active)
        gate['ready']=True
        self.spin(lambda:self.store.ready and s.active,timeout=10,fast=False)
        self.assertFalse(any(f[8]==6 for f in self.mcu.commands))

    def test_origin_loss_after_adoption_blocks_motion_and_registration(self):
        gate={'ready':False};s=self.start();s.startup_checks=lambda:dict(gate)
        self.spin(lambda:s.startup_sequence is not None,fast=False)
        self.mcu.zero_flags=False
        time.sleep(.5);gate['ready']=True
        self.spin(lambda:self.store.data['phase']=='uncertain',fast=False)
        self.assertFalse(self.mcu.commands);self.assertFalse(s.active)
        self.assertTrue(s.registered_startup_attempted)

    def test_operator_pause_survives_fresh_registered_home(self):
        self.store.phase('uncertain',operator_paused=True)
        s=self.start();time.sleep(1.8)
        self.assertTrue(s.operator_paused);self.assertFalse(s.active)
        self.assertFalse(self.mcu.commands);self.assertIsNone(s.startup_sequence)
        with self.assertRaisesRegex(ValueError,'operator_paused'):self.store.adopt_registered(HOME,73)

    def test_adoption_cannot_rewrite_origins_and_persists_distinct_evidence(self):
        self.store.adopt_registered(HOME,73)
        self.store.phase('disable_pending')
        with self.assertRaisesRegex(RuntimeError,'already_attempted'):self.store.before_zero()
        self.assertFalse(self.store.data['zero_attempted'])
        self.store.phase('ready',ready_telemetry=HOME)
        self.store.close();self.store=PowerCycleStore(self.path,30,'test',boot_id='test')
        self.assertTrue(self.store.ready);self.assertFalse(self.store.data['zero_attempted'])


if __name__=='__main__':unittest.main()
