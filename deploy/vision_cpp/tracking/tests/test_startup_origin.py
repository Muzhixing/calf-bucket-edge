"""Candidate startup integration, PTY only; no physical serial devices."""
import time
import threading
import unittest
from test_tracking import Reference, MCU, Sender
from arm import ContinuousArmPort
from runtime import Session

class FeedbackMCU(MCU):
    def frame(self):
        f=bytearray(super().frame())
        f[5]|=getattr(self,'feedback',0xC0)
        fault=getattr(self,'fault',0)
        if fault==1 and any(not (c[8]&1) for c in self.commands):fault=0
        f[15]=fault
        if fault:f[4]=4;f[3]=6
        f[18]=0
        for v in f[1:18]:f[18]^=v
        return f

class StartupTests(unittest.TestCase):
    def setUp(self):
        self.ref=Reference();self.mcu=FeedbackMCU();self.port=None
        self.mcu.a=75;self.mcu.b=self.mcu.target=0.;self.mcu.c=0
    def tearDown(self):
        if self.port:self.port.shutdown()
        self.mcu.close();self.ref.directory.cleanup()
    def connect(self):
        self.port=ContinuousArmPort(self.ref.limits,lambda *a:None,port=self.mcu.path,sender_factory=Sender,check_holders=False)
        return self.port
    def finish(self):
        end=time.monotonic()+2
        while self.port.phase!='idle' and time.monotonic()<end:
            self.port.tick();time.sleep(.01)
        self.assertEqual(self.port.phase,'idle')
    def test_verified_zero_prepares_C_then_allows_A(self):
        p=self.connect();self.assertEqual(self.mcu.commands,[])
        self.assertTrue(p.needs_preparation());self.assertTrue(p.prepare())
        self.assertFalse(p.follow([77,0,10]))
        self.finish();self.assertFalse(p.needs_preparation());self.assertFalse(self.mcu.enabled)
        self.assertEqual([f[4:7] for f in self.mcu.commands],[bytes([75,0,10])]*2)
        time.sleep(.11);self.assertTrue(p.follow([77,0,10]))
    def test_C_preparation_can_be_cancelled(self):
        p=self.connect();p.prepare();p.stop();self.finish()
        self.assertFalse(self.mcu.enabled)
        self.assertEqual([f[8] for f in self.mcu.commands],[1,0])
    def test_partial_C_preparation_can_resume_without_rezero(self):
        self.mcu.c=3.2;p=self.connect();p.prepare();self.finish()
        self.assertEqual(self.mcu.c,10)
        self.assertTrue(all(f[8] in (0,1) for f in self.mcu.commands))
    def test_missing_feedback_blocks_C_start_without_writes(self):
        self.mcu.feedback=0
        with self.assertRaisesRegex(RuntimeError,'startup_pose'):self.connect()
        self.assertEqual(self.mcu.commands,[])
    def test_wrong_A_start_blocks_preparation_without_writes(self):
        self.mcu.a=90
        with self.assertRaisesRegex(RuntimeError,'startup_pose'):self.connect()
        self.assertEqual(self.mcu.commands,[])
    def test_only_verified_stopped_startup_communication_fault_gets_disable_ack(self):
        self.mcu.fault=1;p=self.connect()
        self.assertEqual(len(self.mcu.commands),1)
        self.assertEqual(self.mcu.commands[0][8],0);self.assertFalse(self.mcu.enabled)
        self.assertEqual(p.state['system_fault'],0);self.assertTrue(p.needs_preparation())
    def test_motor_or_origin_fault_never_auto_cleared(self):
        self.mcu.fault=5
        with self.assertRaisesRegex(RuntimeError,'firmware_fault'):self.connect()
        self.assertEqual(self.mcu.commands,[])
    def test_automatic_session_prepares_then_tracks_and_pauses(self):
        def factory(limits,log):
            return ContinuousArmPort(limits,log,port=self.mcu.path,sender_factory=Sender,check_holders=False)
        session=Session(self.ref.directory.name,self.ref.limits,smooth=True,automatic=True,port_factory=factory,intrinsics=dict(fx=691.1954,fy=691.1954,cx=489.3364,cy=374.265))
        worker=threading.Thread(target=session.arm_loop);worker.start()
        def feed(seconds,dx):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                with session.lock:
                    session.metadata={'arrival':time.monotonic()};session.target_id=1
                    session.vision_state='tracking' if dx is not None else 'reacquiring'
                    session.tracking=None if dx is None else {'box':[400,200,650,450], 'offset_px':[dx,0], 'timestamp':time.monotonic(),'method':'flow'}
                time.sleep(.02)
        try:
            feed(3.7,-80)
            enabled=[f for f in self.mcu.commands if f[8]&1]
            self.assertTrue(enabled,session.reason)
            self.assertEqual(enabled[0][4:7],bytes([75,0,10]))
            self.assertTrue(any(f[4]!=75 and f[6]==10 for f in enabled[1:]),session.reason)
            feed(.5,None);self.assertFalse(self.mcu.enabled)
            n=len(self.mcu.commands);feed(.5,0)
            self.assertFalse(self.mcu.enabled)
            self.assertFalse(any(f[8]&1 for f in self.mcu.commands[n:]))
        finally:
            session.stop_event.set();worker.join(timeout=4)
            self.assertFalse(worker.is_alive())

    def test_old_firmware_at_working_C_remains_compatible(self):
        self.mcu.feedback=0;self.mcu.c=10;self.mcu.a=78;self.mcu.b=self.mcu.target=40
        p=self.connect();self.assertFalse(p.needs_preparation())
        self.assertTrue(p.follow([80,40,10]))

if __name__=='__main__':unittest.main()
