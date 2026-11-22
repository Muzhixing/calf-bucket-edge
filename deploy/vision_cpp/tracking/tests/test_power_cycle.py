"""Whole-machine assumption workflow: persistent at-most-once attempts, PTY only."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import uuid
from unittest.mock import patch

from test_tracking import Reference, MCU, Sender
from arm import ContinuousArmPort, Limits
from startup import PowerCycleStore, StartupPort, StartupSequence, derive_wait_profile
from runtime import Session

INTRINSICS=dict(fx=691.1954,fy=691.1954,cx=489.3364,cy=374.265)

class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'state.json';self.store=None
    def tearDown(self):
        if self.store:self.store.close()
        self.temp.cleanup()
    def open(self,**kwargs):
        self.store=PowerCycleStore(self.path,5,'test measured bound',boot_id='host-1',**kwargs);return self.store
    def test_duplicate_confirmation_and_prewrite_marker_survive_process_restart(self):
        s=self.open();rid=str(uuid.uuid4());self.assertTrue(s.confirm(rid));self.assertFalse(s.confirm(rid))
        s.phase('disable_pending');s.before_zero();self.assertTrue(json.loads(self.path.read_text())['zero_attempted'])
        with self.assertRaises(RuntimeError):s.before_zero()
        s.close();s=self.open();self.assertEqual(s.data['phase'],'uncertain');self.assertFalse(s.confirm(rid))
        self.assertTrue(s.data['zero_attempted']);self.assertFalse(s.ready)
    def test_restart_during_wait_never_resumes_initialization(self):
        s=self.open();rid=str(uuid.uuid4());s.confirm(rid);s.close();s=self.open()
        self.assertEqual(s.data['phase'],'uncertain');self.assertFalse(s.confirm(rid))
    def test_completed_cycle_survives_process_restart_but_host_boot_cannot_authorize_new_cycle(self):
        s=self.open();s.confirm(str(uuid.uuid4()));s.phase('disable_pending');s.before_zero()
        s.phase('ready',ready_telemetry={'proof':'simulated'});s.close();s=self.open();self.assertTrue(s.ready)
        s.close();self.store=PowerCycleStore(self.path,5,'test',boot_id='host-2')
        self.assertFalse(self.store.ready);self.assertEqual(self.store.data['phase'],'uncertain')
    def test_file_lock_prevents_competing_initializers(self):
        self.open()
        with self.assertRaisesRegex(RuntimeError,'owned'):PowerCycleStore(self.path,5,'test',boot_id='host-1')
    def test_corrupt_state_never_authorizes_or_overwrites(self):
        self.path.write_text('{broken');s=self.open();self.assertFalse(s.ready)
        with self.assertRaises(ValueError):s.confirm(str(uuid.uuid4()))
        self.assertEqual(self.path.read_text(),'{broken')
    def test_missing_timing_evidence_blocks_confirmation(self):
        self.store=PowerCycleStore(self.path,boot_id='host-1')
        with self.assertRaisesRegex(ValueError,'not_calibrated'):self.store.confirm(str(uuid.uuid4()))
        self.assertFalse(self.path.exists())
    def test_failed_fsync_prevents_zero_attempt_from_returning(self):
        s=self.open();s.confirm(str(uuid.uuid4()));s.phase('disable_pending')
        with patch('startup.os.fsync',side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):s.before_zero()
        self.assertFalse(s.ready);self.assertEqual(s.error,'startup_state_write_failed')
    def test_range_expansion_automatically_increases_wait(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup)
        conf=json.loads(ref.path.read_text());conf['operating_envelope']['C_command_mm']=[0,30]
        conf['power_cycle_startup']={'homing_speed_floor':{'A_deg':30,'B_mm':10,'C_mm':10},'speed_evidence':'fixture test bound','duration_factor':1.5,'startup_and_settle_margin_s':5}
        ref.path.write_text(json.dumps(conf));a=derive_wait_profile(Limits(ref.path))
        conf['operating_envelope']['A_command_degrees']=[40,140];conf['operating_envelope']['C_command_mm']=[0,60]
        ref.path.write_text(json.dumps(conf));b=derive_wait_profile(Limits(ref.path))
        self.assertGreater(b['wait_seconds'],a['wait_seconds']);self.assertEqual(b['distances']['C_mm'],60)
        conf['power_cycle_startup']['homing_speed_floor']['B_mm']=None
        ref.path.write_text(json.dumps(conf));self.assertIsNone(derive_wait_profile(Limits(ref.path))['wait_seconds'])

class HomeMCU(MCU):
    def __init__(self):
        self.zero_flags=False;self.fault=1;self.processed=0
        super().__init__();self.a=90;self.b=self.target=0.;self.c=0
    def frame(self):
        for f in self.commands[self.processed:]:
            if f[8]==0:self.fault=0
            if f[8]==6:self.zero_flags=True
        self.processed=len(self.commands)
        f=bytearray(super().frame());f[5]=(f[5]&~6)|(6 if self.zero_flags else 0);f[15]=self.fault
        if self.fault:f[4]=4;f[3]=6
        f[18]=0
        for v in f[1:18]:f[18]^=v
        return f

class StartupPTYTests(unittest.TestCase):
    def setUp(self):
        self.ref=Reference();self.mcu=HomeMCU();self.events=[]
        self.path=Path(self.ref.directory.name)/'state.json';self.store=PowerCycleStore(self.path,5,'test measured bound',boot_id='test-host')
        self.sequence=None
    def tearDown(self):
        if self.sequence:self.sequence.close(None if self.sequence.done else 'test_end')
        self.store.close();self.mcu.close();self.ref.directory.cleanup()
    def factory(self,limits,log):return StartupPort(limits,log,port=self.mcu.path,sender_factory=Sender,check_holders=False)
    def begin(self):
        self.store.confirm(str(uuid.uuid4()));self.sequence=StartupSequence(self.store,self.ref.limits,lambda k,v:self.events.append((k,v)),self.factory)
        self.sequence.started-=6
        return self.sequence
    def run_until(self,predicate,timeout=5):
        end=time.monotonic()+timeout
        while not predicate() and time.monotonic()<end:self.sequence.tick();time.sleep(.01)
        self.assertTrue(predicate(),self.sequence.stage)
    def test_zero_once_then_actual_A75_then_C10_ready(self):
        seq=self.begin();self.run_until(lambda:seq.done)
        flags=[f[8] for f in self.mcu.commands];self.assertEqual(flags.count(6),1)
        enabled=[f[4:7] for f in self.mcu.commands if f[8]&1]
        self.assertEqual(enabled,[bytes([75,0,0]),bytes([75,0,10])]);self.assertFalse(self.mcu.enabled)
        self.assertTrue(self.store.ready)
        assumption=next(v for k,v in self.events if k=='homing_assumed_after_wait')
        self.assertIs(assumption['hardware_complete_signal'],False)
        # Close/reopen serial and process ledger: reuse live zeros; no flags6.
        seq.close();self.store.close();self.store=PowerCycleStore(self.path,5,'test',boot_id='test-host')
        before=len(self.mcu.commands)
        port=ContinuousArmPort(self.ref.limits,lambda *a:None,port=self.mcu.path,sender_factory=Sender,check_holders=False)
        port.shutdown();self.assertEqual(len(self.mcu.commands),before);self.assertTrue(self.store.ready)
    def test_existing_zero_flags_block_repeat_even_after_new_operator_request(self):
        self.mcu.zero_flags=True;self.mcu.fault=0
        seq=self.begin()
        with self.assertRaisesRegex(RuntimeError,'already_registered'):seq.tick()
        self.assertEqual(self.mcu.commands,[])
    def test_wait_cancellation_never_opens_serial(self):
        self.store.confirm(str(uuid.uuid4()))
        seq=StartupSequence(self.store,self.ref.limits,lambda *a:None,lambda *a:self.fail('opened serial'))
        seq.tick();seq.close('cancelled');self.assertEqual(self.mcu.commands,[]);self.assertFalse(self.store.ready)
    def test_ambiguous_zero_ack_is_not_retried_after_reconnect(self):
        seq=self.begin();self.run_until(lambda:seq.stage=='zero')
        self.mcu.drop=True
        with self.assertRaisesRegex(RuntimeError,'ack_timeout'):
            end=time.monotonic()+2
            while time.monotonic()<end:seq.tick();time.sleep(.01)
        seq.close('ack_timeout');self.assertEqual(sum(f[8]==6 for f in self.mcu.commands),1)
        self.store.close();self.store=PowerCycleStore(self.path,5,'test',boot_id='test-host')
        self.assertFalse(self.store.ready);self.assertTrue(self.store.data['zero_attempted'])
        self.assertFalse(self.store.confirm(self.store.data['cycle_id']))
    def test_mcu_only_reset_does_not_trigger_registration(self):
        seq=self.begin();self.run_until(lambda:seq.done);seq.close();before=len(self.mcu.commands)
        self.mcu.zero_flags=False;self.mcu.a=90;self.mcu.b=self.mcu.target=0.;self.mcu.c=0;self.mcu.count=0
        with self.assertRaisesRegex(RuntimeError,'origins_or_heading'):
            ContinuousArmPort(self.ref.limits,lambda *a:None,port=self.mcu.path,sender_factory=Sender,check_holders=False)
        self.assertEqual(len(self.mcu.commands),before)
    def test_write_ahead_failure_sends_no_zero_frame(self):
        seq=self.begin();self.run_until(lambda:seq.stage=='disable' and self.mcu.commands)
        with patch.object(self.store,'before_zero',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.run_until(lambda:seq.stage=='zero')
        self.assertFalse(any(f[8]==6 for f in self.mcu.commands))

    def test_full_session_initializes_once_then_tracks_and_holds(self):
        def tracking_factory(limits,log):
            return ContinuousArmPort(limits,log,port=self.mcu.path,sender_factory=Sender,check_holders=False)
        s=Session(self.ref.directory.name,self.ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS,
                  startup_store=self.store,startup_port_factory=self.factory,port_factory=tracking_factory)
        request={'request_id':str(uuid.uuid4()),'whole_machine_power_cycled':True}
        s.command('confirm_power_cycle',request)
        worker=threading.Thread(target=s.arm_loop);worker.start()
        def feed(seconds,dx):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                with s.lock:
                    s.metadata={'arrival':time.monotonic()};s.target_id=1;s.vision_state='tracking'
                    s.tracking=None if dx is None else {'box':[400,200,650,450],'offset_px':[dx,0],'timestamp':time.monotonic(),'method':'flow'}
                time.sleep(.02)
        try:
            deadline=time.monotonic()+1
            while s.startup_sequence is None and time.monotonic()<deadline:time.sleep(.01)
            self.assertIsNotNone(s.startup_sequence)
            s.startup_sequence.started-=6
            feed(5.5,-80)
            self.assertTrue(self.store.ready,s.reason)
            self.assertEqual(sum(f[8]==6 for f in self.mcu.commands),1)
            self.assertTrue(any(f[8]==1 and f[4]!=75 and f[6]==10 for f in self.mcu.commands),s.reason)
            feed(.5,None);self.assertFalse(self.mcu.enabled)
            before=len(self.mcu.commands);feed(.5,0)
            self.assertFalse(self.mcu.enabled);self.assertFalse(any(f[8]&1 for f in self.mcu.commands[before:]))
            s.command('stop',{});feed(.2,None)
            s.command('confirm_power_cycle',request)  # Replayed HTTP request is a no-op.
            self.assertIsNone(s.startup_request)
        finally:
            s.stop_event.set();worker.join(4);self.assertFalse(worker.is_alive())

    def test_cancel_A_move_stops_and_keeps_attempt_consumed(self):
        seq=self.begin();self.run_until(lambda:seq.stage=='a');seq.close('operator_stop')
        self.assertFalse(self.mcu.enabled);self.assertFalse(self.store.ready)
        self.assertEqual(sum(f[8]==6 for f in self.mcu.commands),1)

class SessionGateTests(unittest.TestCase):
    def test_fresh_process_missing_record_does_not_open_tracking_port(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup)
        store=PowerCycleStore(Path(ref.directory.name)/'state.json',5,'test',boot_id='test');self.addCleanup(store.close)
        opened=[]
        session=Session(ref.directory.name,ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS,startup_store=store,port_factory=lambda *a:opened.append(True))
        session.metadata={'arrival':time.monotonic()}
        worker=threading.Thread(target=session.arm_loop);worker.start();time.sleep(.12);session.stop_event.set();worker.join(2)
        self.assertFalse(opened);self.assertFalse(session.auto_follow)
        with self.assertRaisesRegex(ValueError,'confirmation_required'):session.command('start',{})
        session.command('auto',{});self.assertFalse(session.auto_follow)
    def test_confirmation_requires_literal_true_and_duplicate_is_idempotent(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup)
        store=PowerCycleStore(Path(ref.directory.name)/'state.json',5,'test',boot_id='test');self.addCleanup(store.close)
        s=Session(ref.directory.name,ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS,startup_store=store)
        rid=str(uuid.uuid4())
        with self.assertRaises(ValueError):s.command('confirm_power_cycle',{'request_id':rid,'whole_machine_power_cycled':'yes'})
        body={'request_id':rid,'whole_machine_power_cycled':True};s.command('confirm_power_cycle',body);s.command('confirm_power_cycle',body)
        self.assertEqual(len(store.data['seen_requests']),1)
        s.command('stop',{});self.assertIsNone(s.startup_request);self.assertEqual(store.data['phase'],'uncertain')

if __name__=='__main__':unittest.main()
