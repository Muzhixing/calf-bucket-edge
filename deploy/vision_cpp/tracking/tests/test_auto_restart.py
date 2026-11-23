"""Automatic boot detection, exclusive serial ownership and pause semantics."""
import json
from pathlib import Path
import threading
import time
import unittest
import uuid

from arm import ContinuousArmPort, reset_telemetry
from runtime import Session
from startup import PowerCycleStore, RestartEvidence, RestartWatchPort, StartupPort, can_auto_prepare
from test_tracking import Reference, Sender
from test_power_cycle import HomeMCU, INTRINSICS

BOOT=dict(enabled=False,lift_zero_confirmed=False,slide_zero_confirmed=False,command_count_mod256=0,
          servo_software_deg=90,lift_position_mm=0,slide_position_mm=0,motor_fault=0,system_fault=1,state=4)

class EvidenceTests(unittest.TestCase):
    def test_counter_reset_or_serial_reconnect_alone_is_insufficient(self):
        self.assertTrue(reset_telemetry(BOOT))
        for key,value in [('lift_zero_confirmed',True),('slide_zero_confirmed',True),('enabled',True),
                          ('servo_software_deg',75),('slide_position_mm',10),('motor_fault',1),('command_count_mod256',1)]:
            self.assertFalse(reset_telemetry({**BOOT,key:value}),key)
    def test_coalesced_buffer_cannot_fake_stability_and_gaps_reset_evidence(self):
        d=RestartEvidence();self.assertFalse(d.observe([BOOT]*20,1))
        self.assertFalse(d.observe([BOOT]*20,1));self.assertEqual(d.samples,1)
        self.assertFalse(d.observe([BOOT],1.25));self.assertFalse(d.observe([BOOT],1.5))
        self.assertTrue(d.observe([BOOT],1.75))
        self.assertFalse(d.observe([BOOT],3));self.assertEqual(d.samples,1)
        self.assertFalse(d.observe([{**BOOT,'lift_zero_confirmed':True}],3.25));self.assertEqual(d.samples,0)

class AutoRestartTests(unittest.TestCase):
    def setUp(self):
        self.ref=Reference();self.ref.limits.startup_policy.update(automatic_restart_detection=True,mcu_restart_returns_all_axes=True)
        self.mcu=HomeMCU();self.path=Path(self.ref.directory.name)/'state.json'
        self.store=PowerCycleStore(self.path,5,'PTY test bound',boot_id='test');self.worker=None;self.session=None
    def tearDown(self):
        if self.session:self.session.stop_event.set()
        if self.worker:self.worker.join(4);self.assertFalse(self.worker.is_alive())
        self.store.close();self.mcu.close();self.ref.directory.cleanup()
    def port(self,kind):
        return lambda limits,log:kind(limits,log,port=self.mcu.path,sender_factory=Sender,check_holders=False)
    def start(self):
        s=self.session=Session(self.ref.directory.name,self.ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS,
             startup_store=self.store,startup_port_factory=self.port(StartupPort),watch_port_factory=self.port(RestartWatchPort),port_factory=self.port(ContinuousArmPort))
        s.startup_checks=lambda:{'ready':True}  # This suite isolates serial/epoch handling; pipeline gates have separate integration tests.
        self.worker=threading.Thread(target=s.arm_loop);self.worker.start();return s
    def spin(self,predicate,timeout=9,fast=True):
        end=time.monotonic()+timeout;accelerated=set()
        while time.monotonic()<end:
            s=self.session
            with s.lock:
                s.metadata={'arrival':time.monotonic()}
                seq=s.startup_sequence
                if fast and seq is not None and id(seq) not in accelerated:
                    seq.started-=6;accelerated.add(id(seq))
            if predicate():return
            time.sleep(.02)
        self.fail((self.session.reason,self.session.arm,self.store.data))
    def reset_mcu(self):
        self.mcu.drop=True;self.mcu.enabled=False;self.mcu.count=0
        self.mcu.a=90;self.mcu.b=self.mcu.target=0;self.mcu.c=0
        self.mcu.zero_flags=False;self.mcu.fault=1;self.mcu.processed=len(self.mcu.commands)
        self.mcu.drop=False
    def test_automatic_start_and_second_real_reset_prepare_once_each_without_ui(self):
        s=self.start();self.spin(lambda:self.store.ready and s.active)
        self.assertEqual([self.mcu.a,self.mcu.b,self.mcu.c],[75,0,10])
        self.assertEqual(sum(f[8]==6 for f in self.mcu.commands),1)
        self.reset_mcu();self.spin(lambda:sum(f[8]==6 for f in self.mcu.commands)==2 and self.store.ready and s.active,timeout=12)
        self.assertEqual([self.mcu.a,self.mcu.b,self.mcu.c],[75,0,10])
        self.assertEqual(sum(f[8]==6 for f in self.mcu.commands),2)
        messages=[x['message'] for x in s.status()['tracking']['events']]
        self.assertTrue(any('检测到 STM32 复位' in m for m in messages))
        self.assertTrue(any('A75°' in m for m in messages))
    def test_pause_during_wait_is_durable_and_never_sends_zero(self):
        s=self.start();self.spin(lambda:s.startup_sequence is not None,fast=False)
        s.command('stop',{});self.spin(lambda:s.startup_sequence is None,fast=False)
        self.assertTrue(self.store.data['operator_paused']);self.assertFalse(self.mcu.commands)
        s.stop_event.set();self.worker.join(3);self.worker=None;self.store.close()
        self.store=PowerCycleStore(self.path,5,'PTY test bound',boot_id='test')
        s=self.start();time.sleep(.4)
        self.assertTrue(s.operator_paused);self.assertFalse(s.active);self.assertFalse(self.mcu.commands)
    def test_ambiguous_registration_is_not_retried_on_same_boot_signature(self):
        self.store.confirm(str(uuid.uuid4()));self.store.phase('disable_pending');self.store.before_zero();self.store.invalidate('ack_lost')
        self.assertFalse(can_auto_prepare(self.store))
        s=self.start();self.spin(lambda:s.reason=='startup_registration_uncertain',fast=False)
        self.assertFalse(self.mcu.commands);self.assertTrue(self.store.data['zero_attempted'])
    def test_reconnecting_prepared_controller_does_not_register_again(self):
        self.mcu.zero_flags=True;self.mcu.fault=0;self.mcu.a=75;self.mcu.c=10
        self.store.confirm(str(uuid.uuid4()));self.store.phase('disable_pending');self.store.before_zero()
        self.store.phase('ready',ready_telemetry={'origins':'confirmed'})
        s=self.start();self.spin(lambda:s.active and s.arm['telemetry'] is not None)
        self.assertFalse(any(f[8]==6 for f in self.mcu.commands));self.assertIsNone(s.startup_sequence)
    def test_live_reset_does_not_send_old_epoch_stop_or_target(self):
        self.mcu.zero_flags=True;self.mcu.fault=0;self.mcu.a=75;self.mcu.c=10
        port=self.port(ContinuousArmPort)(self.ref.limits,lambda *a:None)
        try:
            port.follow([76,0,10]);end=time.monotonic()+1
            while port.pending and time.monotonic()<end:port.tick();time.sleep(.02)
            self.reset_mcu();before=len(self.mcu.commands);end=time.monotonic()+1
            with self.assertRaisesRegex(RuntimeError,'mcu_restart_detected'):
                while time.monotonic()<end:port.tick();time.sleep(.02)
            self.assertIsNone(port.shutdown());self.assertEqual(len(self.mcu.commands),before)
        finally:port.shutdown()
    def test_stop_takes_effect_even_if_state_disk_fails(self):
        s=self.session=Session(self.ref.directory.name,self.ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS,startup_store=self.store)
        s.active=True;s.auto_follow=True;s.startup_request='pending'
        def failed(*a,**k):raise OSError('disk failed')
        self.store.phase=failed
        s.command('stop',{})
        self.assertFalse(s.active);self.assertFalse(s.auto_follow);self.assertIsNone(s.startup_request)
        self.assertTrue(s.operator_paused)
    def test_resume_prepared_waits_for_fresh_camera_without_manual_retries(self):
        self.store.confirm(str(uuid.uuid4()));self.store.phase('ready',operator_paused=True)
        s=self.session=Session(self.ref.directory.name,self.ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS,startup_store=self.store)
        s.command('start',{})
        self.assertTrue(s.auto_follow);self.assertFalse(s.active);self.assertFalse(s.operator_paused)
        self.assertFalse(self.mcu.commands)
    def test_monitor_close_never_writes_even_when_another_owner_enabled_motor(self):
        self.mcu.enabled=True
        watch=self.port(RestartWatchPort)(self.ref.limits,lambda *a:None)
        try:
            time.sleep(.15);self.assertFalse(watch.reset_ready())
        finally:watch.shutdown()
        self.assertFalse(self.mcu.commands)

if __name__=='__main__':unittest.main()
