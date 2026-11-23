import threading
import time
import unittest

from control import AlignmentGate, alignment_radius
from runtime import Session
from arm import ContinuousArmPort
from test_tracking import Reference, MCU, Sender
from test_continuous import INTRINSICS


def target(dx,dy,stamp):
    return {'box':[400,200,650,450],'offset_px':[dx,dy],'timestamp':stamp,'method':'flow'}


class AlignmentTests(unittest.TestCase):
    def test_radius_tracks_box_scale_and_circle_rejects_diagonal_corner(self):
        g=AlignmentGate()
        self.assertEqual(alignment_radius(target(0,0,1)),25)
        self.assertFalse(g.update(target(20,20,1)))
        self.assertTrue(g.update(target(15,20,1.1)))
        small=target(0,0,1);small['box']=[400,200,500,400]
        self.assertEqual(alignment_radius(small),10)

    def test_hysteresis_requires_fresh_sustained_exit(self):
        g=AlignmentGate();self.assertTrue(g.update(target(0,0,1)))
        for t in (1.1,1.2,1.3):self.assertTrue(g.update(target(35,0,t)))
        self.assertTrue(g.update(target(40,0,1.4)))
        for _ in range(5):self.assertTrue(g.update(target(40,0,1.4)))
        self.assertTrue(g.update(target(0,0,1.45)))
        self.assertTrue(g.update(target(40,0,1.5)))
        self.assertFalse(g.update(target(40,0,1.57)))

    def test_box_noise_does_not_shrink_held_circle_and_reset_clears_it(self):
        g=AlignmentGate();g.update(target(0,0,1))
        smaller=target(30,0,1.1);smaller['box']=[400,200,500,300]
        self.assertTrue(g.update(smaller));self.assertEqual(g.radius,25)
        g.reset();self.assertFalse(g.update(smaller))

    def test_bottom_limit_holds_horizontal_center_without_claiming_centered(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup);g=AlignmentGate()
        self.assertTrue(g.update(target(11,50,1),[78,96,10],ref.limits))
        self.assertEqual(g.status()['alignment_hold_mode'],'range_limit')
        self.assertEqual(g.status()['alignment_blocked_axes'],['B'])
        self.assertTrue(g.update(target(30,60,1.1),[78,96,10],ref.limits))
        self.assertTrue(g.update(target(50,60,1.2),[78,96,10],ref.limits))
        self.assertFalse(g.update(target(50,60,1.27),[78,96,10],ref.limits))
    def test_reachable_vertical_reversal_releases_limit_hold_immediately(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup);g=AlignmentGate()
        self.assertTrue(g.update(target(0,50,1),[78,96,10],ref.limits))
        self.assertFalse(g.update(target(0,-30,1.1),[78,96,10],ref.limits))
        self.assertEqual(g.status()['alignment_blocked_axes'],[])
        self.assertTrue(g.update(target(10,-50,1.2),[78,0,10],ref.limits))
        self.assertTrue(g.constrained)
    def test_unreached_limit_or_inward_pending_motion_does_not_hide_error(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup);g=AlignmentGate()
        self.assertFalse(g.update(target(0,50,1),[78,95.9,10],ref.limits,[78,96,10]))
        self.assertFalse(g.update(target(0,50,1.1),[78,96,10],ref.limits,[78,80,10]))
        self.assertFalse(g.update(target(0,50,1.2),[78,40,10],ref.limits))
    def test_A_limit_keeps_vertical_error_correctable(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup);g=AlignmentGate()
        self.assertTrue(g.update(target(80,5,1),[70,40,10],ref.limits))
        self.assertEqual(g.status()['alignment_blocked_axes'],['A'])
        self.assertTrue(g.update(target(80,70,1.1),[70,40,10],ref.limits))
        self.assertFalse(g.update(target(80,70,1.17),[70,40,10],ref.limits))

    def run_session(self,exercise,b=40):
        ref=Reference();mcu=MCU();mcu.speed=1;mcu.b=mcu.target=b
        def factory(limits,log):
            return ContinuousArmPort(limits,log,port=mcu.path,sender_factory=Sender,check_holders=False)
        s=Session(ref.directory.name,ref.limits,smooth=True,automatic=True,
                  intrinsics=INTRINSICS,port_factory=factory)
        worker=threading.Thread(target=s.arm_loop);worker.start()
        def feed(seconds,dx,dy=0,age=0):
            until=time.monotonic()+seconds
            while time.monotonic()<until:
                with s.lock:
                    s.metadata={'arrival':time.monotonic()};s.target_id=1
                    s.tracking=target(dx,dy,time.monotonic()-age);s.vision_state='tracking'
                time.sleep(.02)
        try:exercise(s,mcu,feed)
        finally:s.stop_event.set();worker.join(4);mcu.close();ref.directory.cleanup()

    def test_centered_start_emits_no_enable_and_leaving_resumes(self):
        def exercise(s,mcu,feed):
            feed(2.2,10,5)
            self.assertEqual(mcu.commands,[]);self.assertEqual(s.arm['phase'],'holding')
            self.assertTrue(s.active);self.assertTrue(s.auto_follow)
            feed(.4,34,0);self.assertEqual(mcu.commands,[])
            feed(.6,65,0);self.assertTrue(any(f[8] for f in mcu.commands),s.reason)
            feed(.4,5,5);self.assertFalse(mcu.enabled)
            n=len(mcu.commands);feed(.5,30,0)
            self.assertEqual(n,len(mcu.commands))
            # A may already be at its lower boundary after the previous move.
            self.assertIn(s.arm['phase'],('holding','range_holding'))
            self.assertTrue(s.arm['control']['alignment_hold'])
            feed(.6,-65,0);self.assertTrue(any(f[8] for f in mcu.commands[n:]),s.reason)
            feed(.4,65,0,age=.8);self.assertFalse(mcu.enabled)
        self.run_session(exercise)

    def test_entering_circle_cancels_inflight_vertical_move(self):
        def exercise(s,mcu,feed):
            feed(2.4,0,90)
            self.assertTrue(mcu.enabled,s.reason)
            self.assertLess(mcu.b,mcu.target)
            feed(.5,5,5);self.assertFalse(mcu.enabled)
            n=len(mcu.commands);b=mcu.b
            feed(.5,5,5)
            self.assertEqual(n,len(mcu.commands));self.assertAlmostEqual(mcu.b,b)
            self.assertTrue(s.active);self.assertEqual(s.arm['phase'],'holding')
        self.run_session(exercise)

    def test_bottom_limit_stops_microcommands_and_recovers_both_directions(self):
        def exercise(s,mcu,feed):
            feed(2.2,11,50)
            self.assertEqual(mcu.commands,[]);self.assertEqual(s.arm['phase'],'range_holding')
            self.assertEqual(s.reason,'working_range_limit');self.assertTrue(s.active)
            for dx in (18,-18,10):feed(.12,dx,50)
            self.assertEqual(mcu.commands,[])
            feed(.6,60,50)
            self.assertTrue(any(f[8]&1 and f[4]!=78 for f in mcu.commands))
            self.assertTrue(all(f[5]==96 for f in mcu.commands))
            feed(.4,0,50);self.assertFalse(mcu.enabled)
            n=len(mcu.commands);feed(.3,10,50);self.assertEqual(len(mcu.commands),n)
            feed(.6,0,-70)
            self.assertTrue(any(f[8]&1 and f[5]<96 for f in mcu.commands[n:]))
            self.assertTrue(s.active,s.reason)
        self.run_session(exercise,b=96)
