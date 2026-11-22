"""Advisory metrics must distinguish their evidence and never extend startup."""
import json
from pathlib import Path
import tempfile
import time
import threading
import unittest
import uuid
from unittest.mock import patch
import numpy as np
from diagnostics import StartupDiagnostics, distribution
from runtime import Session
from startup import PowerCycleStore
from test_tracking import Reference


class DiagnosticTests(unittest.TestCase):
    def test_quantiles_maximum_and_threshold_sample_count(self):
        r=distribution(list(range(1,101)),50)
        self.assertEqual(r,dict(samples=100,p95_ms=95,max_ms=100,long_frames=50,long_threshold_ms=50))
        self.assertIsNone(distribution([],50)['p95_ms'])

    def test_window_freezes_at_eight_seconds_without_wait_or_late_samples(self):
        d=StartupDiagnostics(8,now=10)
        d.metric('inference',12,now=11);self.assertIsNone(d.finish(now=17.99))
        report=d.finish(now=18);d.metric('inference',900,now=19)
        self.assertEqual(report['metrics']['inference']['samples'],1)
        self.assertIs(d.finish(now=80),report);self.assertEqual(report['status'],'样本不足')

    def test_no_cross_clock_subtraction_and_no_fake_end_to_end_value(self):
        d=StartupDiagnostics(8,now=100)
        d.frame({'frame_id':1,'arrival':1234567.,'published_at':1234567.02,'detector':{'npu_run_ms':12}},now=101)
        r=d.finish(now=108)
        self.assertAlmostEqual(r['metrics']['native_post_read']['max_ms'],20,places=3)
        self.assertEqual(r['metrics']['host_to_preview']['samples'],0)
        self.assertNotIn('end_to_end',r['metrics'])

    def test_smooth_texture_static_bucket_is_not_camera_failure(self):
        d=StartupDiagnostics(8,now=0);image=np.full((72,128,3),128,np.uint8)
        for n in range(240):
            d.frame({'frame_id':n,'arrival':n/30,'detector':{'npu_run_ms':12},'detections':[]},image,now=n/30)
        r=d.finish(now=8)
        self.assertEqual(r['camera']['frames'],240);self.assertEqual(r['camera']['candidate_frames'],0)
        self.assertGreater(r['camera']['observations']['low_detail_or_blur_hint'],0)
        self.assertFalse(r['issues']);self.assertIn('unavailable',r['camera']['native_decode_errors'])

    def test_consumer_skips_are_not_asserted_to_be_camera_drops(self):
        d=StartupDiagnostics(8,now=0)
        for n,stamp in ((1,.1),(3,.16),(4,.6),(2,.61)):
            d.frame({'frame_id':n,'arrival':stamp},now=stamp)
        r=d.finish(now=8)
        self.assertEqual(r['camera']['observations']['consumer_skipped_ids'],1)
        self.assertNotIn('consumer_skipped_ids',r['issues'])
        self.assertIn('camera_interval:long_frames',r['issues'])
        self.assertIn('frame_clock_or_sequence_reset',r['issues'])

    def test_uart_baseline_growth_and_coalescing(self):
        d=StartupDiagnostics(8,now=0)
        frame=lambda n:{'uart_error_count_mod256':n}
        d.serial([frame(19)],{'valid_frames':1},now=.1)
        d.serial([frame(19),frame(20)],{'valid_frames':3,'bad_checksum':1},now=.35)
        d.serial([frame(22)],{'valid_frames':4,'bad_checksum':1},now=.6)
        r=d.finish(now=8);s=r['serial']
        self.assertEqual(s['uart_initial'],19);self.assertEqual(s['uart_new_unambiguous'],3)
        self.assertEqual(s['uart_growth_batches'],2);self.assertEqual(s['coalesced_frames'],1)
        self.assertEqual(r['metrics']['serial_batch_interval']['samples'],2)
        self.assertEqual(s['parser']['bad_checksum'],1)

    def test_wrap_or_reset_not_reported_as_255_new_errors(self):
        d=StartupDiagnostics(8,now=0)
        for i,n in enumerate((254,255,0,1,0)):
            d.serial([{'uart_error_count_mod256':n}],{},now=.1+i*.2)
        s=d.finish(now=8)['serial']
        self.assertEqual(s['uart_new_unambiguous'],2)
        self.assertEqual(s['uart_wrap_or_reset_ambiguous'],2)

    def test_zero_flags_lost_rebases_error_count_without_claiming_homing(self):
        d=StartupDiagnostics(8,now=0)
        d.serial([{'uart_error_count_mod256':40,'lift_zero_confirmed':True,'slide_zero_confirmed':True}],{},now=.1)
        d.serial([{'uart_error_count_mod256':0,'lift_zero_confirmed':False,'slide_zero_confirmed':False}],{},now=.2)
        r=d.finish(now=8)
        self.assertEqual(r['serial']['reset_signals'],1)
        self.assertEqual(r['serial']['uart_new_unambiguous'],0)
        self.assertNotIn('homed',r)

    def test_silence_at_end_is_reported_without_waiting_for_another_frame(self):
        d=StartupDiagnostics(8,now=0)
        d.frame({'frame_id':1,'arrival':1},now=1)
        d.serial([{'uart_error_count_mod256':0}],{},now=1)
        r=d.finish(now=8)
        self.assertIn('camera:no_recent_frame_at_window_end',r['issues'])
        self.assertIn('serial:no_recent_frame_at_window_end',r['issues'])

    def test_invalid_timing_is_not_accepted(self):
        d=StartupDiagnostics(8,now=0)
        for x in (float('nan'),float('inf'),-1,None):d.metric('inference',x,now=1)
        self.assertEqual(d.finish(now=8)['metrics']['inference']['samples'],0)


class RestartEntryTests(unittest.TestCase):
    def setUp(self):
        self.ref=Reference();self.addCleanup(self.ref.directory.cleanup)
        self.store=PowerCycleStore(Path(self.ref.directory.name)/'cycle.json',8,'user5+3',boot_id='test')
        self.addCleanup(self.store.close)
        self.session=Session(self.ref.directory.name,self.ref.limits,smooth=True,automatic=True,startup_store=self.store)
        self.ref.limits.startup_policy['mcu_restart_returns_all_axes']=True

    def test_confirmed_mcu_motor_restart_uses_same_once_only_wait(self):
        request={'request_id':str(uuid.uuid4()),'mcu_restarted':True}
        with patch.object(StartupDiagnostics,'probe_async'):
            self.session.command('confirm_mcu_restart',request)
            diag=self.session.startup_diagnostics
            self.session.command('confirm_mcu_restart',request)
        self.assertIs(diag,self.session.startup_diagnostics)
        self.assertEqual(self.store.data['restart_event'],'confirmed_mcu_motor_restart')
        self.assertEqual(self.store.data['wait_seconds_used'],8)
        self.assertFalse(self.store.data['zero_attempted']);self.assertFalse(self.session.auto_follow)
        self.store.close()
        reopened=PowerCycleStore(self.store.path,8,'user5+3',boot_id='test');self.addCleanup(reopened.close)
        self.assertFalse(reopened.ready);self.assertFalse(reopened.confirm(request['request_id'],'confirmed_mcu_motor_restart'))

    def test_unconfirmed_model_of_mcu_restart_cannot_register(self):
        self.ref.limits.startup_policy['mcu_restart_returns_all_axes']=False
        with self.assertRaisesRegex(ValueError,'convention_not_confirmed'):
            self.session.command('confirm_mcu_restart',{'request_id':str(uuid.uuid4()),'mcu_restarted':True})
        self.assertFalse(self.store.path.exists())

    def test_reset_report_invalidates_ready_without_starting_motion_or_new_registration(self):
        self.store.confirm(str(uuid.uuid4()));self.store.phase('disable_pending');self.store.before_zero()
        self.store.phase('ready',ready_telemetry={'test':True})
        self.session.active=True;self.session.auto_follow=True;self.session.tracking={'old':'target'}
        self.session.command('report_mcu_reset',{})
        self.assertFalse(self.store.ready);self.assertTrue(self.store.data['zero_attempted'])
        self.assertFalse(self.session.active);self.assertFalse(self.session.auto_follow)
        self.assertIsNone(self.session.startup_request);self.assertIsNone(self.session.tracking)

    def test_preparation_discards_old_target_and_waits_for_a_new_frame(self):
        s=self.session;s.intrinsics=dict(fx=691.,fy=691.,cx=640.,cy=360.)
        s.metadata={'arrival':time.monotonic(),'_host_received_at':time.monotonic()}
        old={'box':[300,200,600,450],'offset_px':[-80,0],'timestamp':time.monotonic()}
        s.tracking=old;s.startup_request='test-cycle'
        class Completed:
            stage='c';done=True;port=None;started=time.monotonic()-8
            def tick(self):pass
            def close(self):return None
        s.startup_sequence=Completed()
        self.assertTrue(s.advance_startup());self.assertIsNone(s.tracking)
        self.assertGreater(s.follow_after,s.metadata['_host_received_at'])
        opened=[];s.port_factory=lambda *args:opened.append(True)
        s.tracking=old  # Even a late stale result cannot start a control owner.
        worker=threading.Thread(target=s.arm_loop);worker.start()
        time.sleep(.06);s.stop_event.set();worker.join(2)
        self.assertFalse(worker.is_alive());self.assertFalse(opened);self.assertFalse(s.active)

    def test_optional_observer_exception_does_not_propagate_to_control(self):
        self.session.startup_diagnostics=StartupDiagnostics(8)
        with patch.object(self.session.startup_diagnostics,'frame',side_effect=RuntimeError('probe failure')):
            self.assertIsNone(self.session.diagnostic_call('frame',{}))
        self.assertEqual(self.session.startup_diagnostics.errors['diagnostic_internal_error:frame'],1)

    def test_diagnostic_anomaly_is_advisory_not_a_new_readiness_gate(self):
        self.session.startup_diagnostics=StartupDiagnostics(8,now=0)
        self.session.startup_diagnostics.error('camera:historical_fault',now=1)
        self.session.startup_diagnostics.finish(now=8)
        # Existing readiness inputs still determine the gate; no diagnostic verdict participates.
        self.session.intrinsics={'fx':1,'fy':1,'cx':0,'cy':0}
        now=time.monotonic()
        self.session.preview_metadata=self.session.metadata={'arrival':now}
        self.session.jpeg=b'x';self.session.video_arrival=now
        self.session.native_warmup.count=10;self.session.native_warmup.last_arrival=now
        self.assertTrue(self.session.startup_checks()['ready'])

if __name__=='__main__':unittest.main()
