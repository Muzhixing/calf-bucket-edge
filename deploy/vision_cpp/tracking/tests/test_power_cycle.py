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
from startup import PowerCycleStore, StartupPort, StartupSequence, derive_wait_profile, NativeWarmup
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
        conf['power_cycle_startup']={'observed_home_max_s':5,'settle_margin_s':3,'reference_return_distances':{'A_deg':30,'B_mm':96,'C_mm':30},'evidence':'user observed about 5s'}
        ref.path.write_text(json.dumps(conf));a=derive_wait_profile(Limits(ref.path))
        conf['operating_envelope']['A_command_degrees']=[40,140];conf['operating_envelope']['C_command_mm']=[0,60]
        ref.path.write_text(json.dumps(conf));b=derive_wait_profile(Limits(ref.path))
        self.assertEqual(a['wait_seconds'],8);self.assertEqual(b['wait_seconds'],13);self.assertEqual(b['distances']['C_mm'],60)
        conf['power_cycle_startup']['observed_home_max_s']=None
        ref.path.write_text(json.dumps(conf))
        with self.assertRaises(ValueError):derive_wait_profile(Limits(ref.path))
    def test_explicit_wait_stays_eight_after_range_expansion(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup)
        conf=json.loads(ref.path.read_text());conf['operating_envelope'].update(A_command_degrees=[0,160],C_command_mm=[0,100])
        conf['power_cycle_startup']={'fixed_wait_seconds':8}
        ref.path.write_text(json.dumps(conf));p=derive_wait_profile(Limits(ref.path))
        self.assertEqual(p['wait_seconds'],8)
        self.assertFalse(json.loads(p['wait_basis'])['range_scaling_applied'])
        for value in (True,float('nan'),4,301):
            conf['power_cycle_startup']['fixed_wait_seconds']=value
            ref.path.write_text(json.dumps(conf))
            with self.assertRaises(ValueError):derive_wait_profile(Limits(ref.path))


class WarmupTests(unittest.TestCase):
    def test_advancing_real_inference_frames_without_bucket_are_enough(self):
        warm=NativeWarmup()
        for n in range(10):
            m={'frame_id':n,'arrival':1+n*.03,'backend':'cpp','detector':{'npu_run_ms':12},'detections':[]}
            warm.observe(m,now=m['arrival'])
            self.assertEqual(warm.status(m['arrival'])['ready'],n==9)
        self.assertFalse(warm.status(3)['ready'])
    def test_duplicate_and_invalid_inference_cannot_fake_warmup(self):
        warm=NativeWarmup();m={'frame_id':1,'arrival':1.,'backend':'cpp','detector':{'npu_run_ms':12},'detections':[]}
        for _ in range(20):warm.observe(m,now=1.)
        self.assertEqual(warm.count,1);self.assertFalse(warm.status(1.)['ready'])
        warm.observe({**m,'detector':{'npu_run_ms':float('nan')}},now=1.)
        self.assertEqual(warm.count,0)

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
    def test_C_zero_startup_sends_only_A_move(self):
        self.ref.limits.c=0
        seq=self.begin();self.run_until(lambda:seq.done)
        enabled=[f[4:7] for f in self.mcu.commands if f[8]&1]
        self.assertEqual(enabled,[bytes([75,0,0])])
        self.assertEqual(sum(f[8]==6 for f in self.mcu.commands),1)
        self.assertEqual(self.mcu.c,0);self.assertTrue(self.store.ready)
    def test_A_only_sweep_hits_both_limits_and_returns_without_BC_travel(self):
        self.ref.limits.a_min=0;self.ref.limits.a_max=160
        self.ref.limits.c=0;self.ref.limits.startup_policy['a_startup_selftest']=True
        seq=self.begin();self.run_until(lambda:seq.done,timeout=6)
        enabled=[list(f[4:7]) for f in self.mcu.commands if f[8]&1]
        self.assertEqual(enabled,[[0,0,0],[160,0,0],[75,0,0]])
        self.assertEqual(sum(f[8]==6 for f in self.mcu.commands),1)
        self.assertTrue(all(f[5:7]==bytes([0,0]) for f in self.mcu.commands))
        self.assertEqual([75,0,0],[self.mcu.a,self.mcu.b,self.mcu.c])
        self.assertTrue(self.store.ready);self.assertFalse(self.mcu.enabled)
        self.assertEqual([v['stage'] for k,v in self.events if k=='startup_a_check'],['left','right','return'])
    def test_A_only_sweep_cancellation_does_not_advance_or_register_again(self):
        self.ref.limits.c=0;self.ref.limits.startup_policy['a_startup_selftest']=True
        seq=self.begin();self.run_until(lambda:seq.stage=='a_left')
        self.assertIsNone(seq.close('operator_stop'))
        self.assertFalse(self.store.ready);self.assertFalse(self.mcu.enabled)
        self.assertEqual(sum(f[8]==6 for f in self.mcu.commands),1)
        self.assertEqual([f[4] for f in self.mcu.commands if f[8]&1],[self.ref.limits.a_min])
        self.store.close();self.store=PowerCycleStore(self.path,5,'test',boot_id='test-host')
        self.assertFalse(self.store.ready);self.assertTrue(self.store.data['zero_attempted'])
    def test_A_sweep_missing_ack_cannot_advance_to_next_limit(self):
        self.ref.limits.c=0;self.ref.limits.startup_policy['a_startup_selftest']=True
        seq=self.begin();self.run_until(lambda:seq.stage=='a_left')
        self.mcu.drop=True;time.sleep(.03)
        # Deadline expiry must fail without adding another sweep target.
        seq.port.deadline=time.monotonic()-1
        with self.assertRaisesRegex(RuntimeError,'movement_timeout'):seq.tick()
        self.assertFalse(self.store.ready)
        self.assertEqual([f[4] for f in self.mcu.commands if f[8]&1],[self.ref.limits.a_min])
        self.mcu.drop=False

    def test_existing_zero_flags_block_repeat_even_after_new_operator_request(self):
        self.mcu.zero_flags=True;self.mcu.fault=0
        seq=self.begin()
        with self.assertRaisesRegex(RuntimeError,'already_registered'):self.run_until(lambda:seq.stage=='connect')
        self.assertEqual(self.mcu.commands,[])
    def test_wait_cancellation_reads_only_and_sends_no_commands(self):
        self.store.confirm(str(uuid.uuid4()))
        self.sequence=StartupSequence(self.store,self.ref.limits,lambda *a:None,self.factory)
        self.sequence.tick();self.sequence.close('cancelled')
        self.assertEqual(self.mcu.commands,[]);self.assertFalse(self.store.ready)

    def test_wait_and_self_checks_overlap_and_both_are_required(self):
        seq=self.begin();ready=False
        seq.ready_check=lambda:{'ready':ready}
        end=time.monotonic()+1.8
        while time.monotonic()<end:seq.tick();time.sleep(.01)
        self.assertEqual(seq.stage,'waiting');self.assertEqual(self.mcu.commands,[])
        ready=True;seq.tick()
        self.assertEqual(seq.stage,'connect')  # no second serial preflight or extra homing wait
        self.assertEqual(self.mcu.commands,[])
        seq.tick();self.run_until(lambda:any(f[8]==0 for f in self.mcu.commands))

    def test_ready_self_checks_do_not_shorten_homing_wait(self):
        self.store.confirm(str(uuid.uuid4()))
        seq=StartupSequence(self.store,self.ref.limits,lambda *a:None,self.factory,lambda:{'ready':True})
        self.sequence=seq
        end=time.monotonic()+1.8
        while time.monotonic()<end:seq.tick();time.sleep(.01)
        self.assertEqual(seq.stage,'waiting');self.assertEqual(self.mcu.commands,[])
        seq.started-=6;seq.tick();self.assertEqual(seq.stage,'connect')

    def test_lost_self_check_during_motion_cancels_without_reregistering(self):
        seq=self.begin();self.run_until(lambda:seq.stage=='a')
        seq.ready_check=lambda:{'ready':False}
        with self.assertRaisesRegex(RuntimeError,'self_check_lost'):seq.tick()
        seq.close('camera_unavailable');self.assertFalse(self.mcu.enabled)
        self.assertEqual(sum(f[8]==6 for f in self.mcu.commands),1)

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

    def test_full_session_initializes_once_then_tracks_and_holds(self,action='confirm_power_cycle'):
        def tracking_factory(limits,log):
            return ContinuousArmPort(limits,log,port=self.mcu.path,sender_factory=Sender,check_holders=False)
        s=Session(self.ref.directory.name,self.ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS,
                  startup_store=self.store,startup_port_factory=self.factory,port_factory=tracking_factory)
        request={'request_id':str(uuid.uuid4()),'whole_machine_power_cycled':True,'mcu_restarted':True}
        s.command(action,request)
        worker=threading.Thread(target=s.arm_loop);worker.start()
        def feed(seconds,dx):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                with s.lock:
                    stamp=time.monotonic()
                    s.metadata={'arrival':stamp};s.target_id=1;s.vision_state='tracking'
                    s.preview_metadata={'arrival':stamp,'frame_id':int(stamp*1000),'backend':'cpp','detector':{'npu_run_ms':12},'detections':[]}
                    s.native_warmup.observe(s.preview_metadata);s.jpeg=b'preview';s.video_arrival=stamp
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
            s.command(action,request)  # Replayed HTTP request is a no-op.
            self.assertIsNone(s.startup_request)
        finally:
            s.stop_event.set();worker.join(4);self.assertFalse(worker.is_alive())

    def test_confirmed_mcu_restart_full_sequence_and_tracking(self):
        self.ref.limits.startup_policy['mcu_restart_returns_all_axes']=True
        self.test_full_session_initializes_once_then_tracks_and_holds('confirm_mcu_restart')

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
    def test_camera_initialization_failure_waits_without_consuming_new_cycle(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup)
        store=PowerCycleStore(Path(ref.directory.name)/'state.json',8,'user observed 5s plus 3s',boot_id='test');self.addCleanup(store.close)
        s=Session(ref.directory.name,ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS,startup_store=store)
        rid=str(uuid.uuid4());s.command('confirm_power_cycle',{'request_id':rid,'whole_machine_power_cycled':True})
        class Capture:
            n=0
            def read(self):
                self.n+=1
                if self.n==2:raise RuntimeError('camera_initializing')
                if self.n==3:s.stop_event.set()
                return object(),{'frame_id':self.n,'arrival':time.monotonic(),'backend':'cpp','detections':[],'detector':{'npu_run_ms':12}}
        class Hub:
            def publish(self,*a):pass
            def fail(self,*a):pass
        s.capture_loop(Capture(),Hub())
        self.assertEqual(s.startup_request,rid);self.assertEqual(store.data['phase'],'waiting')
        self.assertFalse(store.data['zero_attempted']);self.assertFalse(s.startup_checks()['ready'])

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
