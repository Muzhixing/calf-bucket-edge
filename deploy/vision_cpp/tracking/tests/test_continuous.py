import json
import math
import threading
import time
import unittest

import test_tracking
from test_tracking import Reference, MCU, Sender
from arm import ContinuousArmPort, Limits, pose
from control import FollowPolicy
from runtime import Session

INTRINSICS=dict(fx=691.1954,fy=691.1954,cx=489.3364,cy=374.265)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.ref=Reference();self.addCleanup(self.ref.directory.cleanup)
        self.policy=FollowPolicy(self.ref.limits,**INTRINSICS)
    def target(self,dx,stamp):
        return {'box':[520,220,760,480],'offset_px':[dx,0],'timestamp':stamp,'method':'flow'}
    def test_confirmed_angle_limits_are_loaded_and_enforced(self):
        config=json.loads(self.ref.path.read_text())
        config['operating_envelope']['A_command_degrees']=[60,110]
        self.ref.path.write_text(json.dumps(config));limits=Limits(self.ref.path)
        limits.check([60,29,10]);limits.check([110,96,10])
        for q in ([59,29,10],[111,29,10]):
            with self.assertRaises(RuntimeError):limits.check(q)
        p=FollowPolicy(limits,**INTRINSICS);q=70
        for i in range(150):q=p.next_a(self.target(500,1+i*.05),q,q,1+i*.05+.08)
        self.assertEqual(q,60)
        for invalid in ([-1,110],[60,181],[110,60],[60.5,110]):
            config['operating_envelope']['A_command_degrees']=invalid
            self.ref.path.write_text(json.dumps(config))
            with self.assertRaises(RuntimeError):Limits(self.ref.path)

    def test_expanded_range_and_retracted_startup_configuration(self):
        from pathlib import Path
        from startup import derive_wait_profile
        limits=Limits(Path(__file__).resolve().parents[3]/'serial_rx'/'arm_reference.json')
        self.assertEqual((limits.a_min,limits.a_max,limits.c_bounds,limits.c),(0,160,[0,100],0))
        limits.check([0,0,0]);limits.check([160,96,100])
        for q in ([-1,0,0],[161,0,0],[75,0,101]):
            with self.assertRaises(RuntimeError):limits.check(q)
        self.assertEqual(derive_wait_profile(limits)['wait_seconds'],8)
    def test_image_center_is_not_principal_point(self):
        p=self.policy;p.observe(self.target(0,1),80)
        self.assertAlmostEqual(p.world,80)
        p.observe(self.target(100,1.1),80)
        self.assertLess(p.world,80)
        self.assertAlmostEqual(p.diagnostics['bearing_error_deg'],math.degrees(math.atan((740-489.3364)/691.1954)-math.atan((640-489.3364)/691.1954)))
    def test_camera_motion_is_not_object_velocity(self):
        p=self.policy
        for i in range(15):
            a=78+i*.2
            angle=(a-78)/p.units_per_rad+p.center_ray
            dx=math.tan(angle)*p.fx+p.cx-640
            p.observe(self.target(dx,1+i*.05),a)
        self.assertAlmostEqual(p.world,78,places=5)
        self.assertAlmostEqual(p.velocity,0,places=5)

    def test_detector_bridge_does_not_extrapolate_a_box_transition(self):
        p=self.policy
        for i in range(12):p.observe(self.target(-10*i,1+i*.05),80)
        self.assertGreater(p.velocity,0)
        target=self.target(-130,1.6);target['method']='detector_bridge'
        p.observe(target,80)
        self.assertEqual(p.velocity,0);self.assertEqual(len(p.observations),1)
    def test_bearing_does_not_require_depth_and_commands_are_bounded(self):
        p=self.policy;current=80
        for i in range(150):
            result=p.next_a(self.target(200 if i<60 else -200,1+i*.08),current,current,1+i*.08+.08)
            self.assertLessEqual(abs(result-current),4)
            self.assertTrue(70<=result<=100)
            current=result
        self.assertGreater(current,80)
    def test_depth_only_scales_associated_vertical_translation(self):
        t=self.target(0,1);t['offset_px'][1]=20
        d=dict(zip(('left','top','right','bottom'),t['box']))
        a=self.policy.next_b(t,40,{'distance_m':.15,'distance_age_ms':20,'selected':d})
        b=self.policy.next_b(t,40,{'distance_m':.6,'distance_age_ms':20,'selected':d})
        self.assertLess(a,b)
        self.assertEqual(self.policy.next_b(t,40,{'distance_m':.15,'distance_age_ms':900,'selected':d}),54)
    def test_continuous_policy_reduces_delay_in_bounded_motion_simulation(self):
        # Ideal 20 px/degree response with 80 ms observation delay, 10 Hz
        # commands, and 100 degree/s software servo. This is not a field result.
        p=self.policy;positions=[80.]*8;a=80.;old=80.;old_next=0;new_error=[];old_error=[]
        for i in range(1000):
            now=1+i*.01;goal=80+5*math.sin(i*.01*2*math.pi*.18)
            if i%10==0:
                captured=positions[-8]
                pastgoal=80+5*math.sin((i*.01-.08)*2*math.pi*.18)
                target=self.target((captured-pastgoal)*20,now-.08)
                command=p.next_a(target,captured,a,now)
            a+=max(-1.,min(1.,command-a));positions.append(a)
            if now>=old_next:
                old+=max(-4,min(4,round(goal-old)));old_next=now+.7
            if i>100:new_error.append(abs(a-goal));old_error.append(abs(old-goal))
        self.assertLess(sum(new_error),.8*sum(old_error))

    def test_delayed_camera_feedback_settles_without_static_oscillation(self):
        for physical_delay in (.10,.14,.18):
            p=FollowPolicy(self.ref.limits,**INTRINSICS)
            history=[76.]*100;command=76;a=76.;errors=[]
            for i in range(800):
                now=2+i*.01;capture_index=len(history)-9
                actual_index=capture_index-round(physical_delay/.01)
                estimated_index=capture_index-round(p.feedback_delay/.01)
                actual=history[max(0,actual_index)];estimated=history[max(0,estimated_index)]
                if i%4==0:
                    target=self.target((actual-85)*20,now-.08)
                    command=p.next_a(target,estimated,command,now)
                a+=max(-1.,min(1.,command-a));history.append(a)
                if i>400:errors.append(abs(actual-85)*20)
            self.assertLessEqual(max(errors),20,physical_delay)



class ContinuousSerialTests(unittest.TestCase):
    def setUp(self):
        self.ref=Reference();self.mcu=MCU();self.events=[]
        self.port=ContinuousArmPort(self.ref.limits,lambda k,v:self.events.append((k,v)),port=self.mcu.path,sender_factory=Sender,check_holders=False)
    def tearDown(self):
        self.port.shutdown();self.mcu.close();self.ref.directory.cleanup()
    def acknowledge(self):
        until=time.monotonic()+1
        while (self.port.pending_counter is not None or not self.port.can_send()) and time.monotonic()<until:
            self.port.tick();time.sleep(.01)
        self.assertIsNone(self.port.pending_counter)
    def test_retargets_without_disable_or_settle_wait(self):
        self.assertTrue(self.port.follow([80,40,10]));self.acknowledge()
        self.assertTrue(self.port.follow([82,40,10]));self.acknowledge()
        end=time.monotonic()+.65
        while time.monotonic()<end:self.port.tick();time.sleep(.01)
        self.assertEqual(self.port.phase,'following');self.assertTrue(self.mcu.enabled)
        self.assertEqual([f[8] for f in self.mcu.commands],[1,1])
        self.port.stop()
        end=time.monotonic()+1
        while self.port.phase!='idle' and time.monotonic()<end:self.port.tick();time.sleep(.01)
        self.assertFalse(self.mcu.enabled)
    def test_A_updates_preserve_inflight_B_goal(self):
        self.mcu.speed=1
        self.port.follow([78,44,10]);self.acknowledge()
        self.port.follow([80,44,10]);self.acknowledge()
        self.assertLess(self.mcu.b,44);self.assertEqual(self.mcu.target,44)
        with self.assertRaises(RuntimeError):self.port.follow([80,42,10])
        self.assertEqual([f[5] for f in self.mcu.commands],[44,44])
    def test_longer_B_move_keeps_A_responsive_and_preserves_C(self):
        self.mcu.speed=1
        self.assertTrue(self.port.follow([78,64,10]));self.acknowledge()
        self.assertTrue(self.port.follow([80,64,10]));self.acknowledge()
        self.assertLess(self.mcu.b,64)
        self.assertEqual([f[5] for f in self.mcu.commands],[64,64])
        self.assertTrue(all(f[6]==10 and f[8]==1 for f in self.mcu.commands))
    def test_bounded_command_window_and_count_wrap(self):
        self.port.expected=self.mcu.count=255;self.port.state['command_count_mod256']=255
        self.mcu.drop=True
        self.port.follow([80,40,10]);time.sleep(.05)
        self.assertTrue(self.port.follow([82,40,10]));time.sleep(.05)
        self.assertTrue(self.port.follow([84,40,10]));time.sleep(.05)
        self.assertFalse(self.port.follow([86,40,10]))
        self.mcu.drop=False;self.acknowledge();self.assertEqual(self.port.expected,2)
    def test_illegal_pose_never_transmits(self):
        for q in ([101,40,10],[79,41,10],[78,40,11],[83,40,10],[78,65,10]):
            with self.assertRaises(RuntimeError):self.port.follow(q)
        self.assertEqual(self.mcu.commands,[])


class AutomaticTests(unittest.TestCase):
    def test_horizontal_error_does_not_block_lift_start_or_A_updates(self):
        ref=Reference();mcu=MCU();mcu.speed=1
        def factory(limits,log):return ContinuousArmPort(limits,log,port=mcu.path,sender_factory=Sender,check_holders=False)
        s=Session(ref.directory.name,ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS,port_factory=factory)
        worker=threading.Thread(target=s.arm_loop);worker.start()
        try:
            end=time.monotonic()+3.5
            while time.monotonic()<end:
                with s.lock:
                    s.metadata={'arrival':time.monotonic()};s.target_id=1;s.vision_state='tracking'
                    s.tracking={'box':[400,200,650,450],'offset_px':[80,100],'timestamp':time.monotonic(),'method':'flow'}
                moving=[f for f in mcu.commands if f[8]&1 and f[5]>40]
                if len(moving)>=2 and moving[-1][4]!=moving[0][4]:break
                time.sleep(.02)
            moving=[f for f in mcu.commands if f[8]&1 and f[5]>40]
            self.assertTrue(moving,s.reason);self.assertEqual(moving[0][5],64)
            self.assertTrue(any(f[4]!=moving[0][4] for f in moving[1:]))
            self.assertTrue(all(f[6]==10 for f in mcu.commands));self.assertTrue(s.active,s.reason)
        finally:s.stop_event.set();worker.join(4);mcu.close();ref.directory.cleanup()

    def test_real_lift_reversal_still_stops_inflight_motion(self):
        ref=Reference();mcu=MCU();mcu.speed=1
        def factory(limits,log):return ContinuousArmPort(limits,log,port=mcu.path,sender_factory=Sender,check_holders=False)
        s=Session(ref.directory.name,ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS,port_factory=factory)
        worker=threading.Thread(target=s.arm_loop);worker.start()
        try:
            end=time.monotonic()+4
            while time.monotonic()<end:
                moving=any(f[8] and f[5]>40 for f in mcu.commands)
                with s.lock:
                    s.metadata={'arrival':time.monotonic()};s.target_id=1;s.vision_state='tracking'
                    s.tracking={'box':[400,200,650,450],'offset_px':[0,-100 if moving else 100],'timestamp':time.monotonic(),'method':'flow'}
                if moving and any(not f[8] for f in mcu.commands):break
                time.sleep(.02)
            self.assertTrue(any(f[8] and f[5]>40 for f in mcu.commands))
            self.assertTrue(any(not f[8] for f in mcu.commands))
            self.assertTrue(s.active,s.reason)
        finally:s.stop_event.set();worker.join(4);mcu.close();ref.directory.cleanup()

    def test_fractional_lift_resume_cannot_block_horizontal_tracking(self):
        for start_b,dy in ((16.6,-100),(16.4,100)):
            with self.subTest(start_b=start_b):
                ref=Reference();mcu=MCU();mcu.b=mcu.target=start_b;mcu.speed=1
                def factory(limits,log):return ContinuousArmPort(limits,log,port=mcu.path,sender_factory=Sender,check_holders=False)
                s=Session(ref.directory.name,ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS,port_factory=factory)
                worker=threading.Thread(target=s.arm_loop);worker.start()
                try:
                    end=time.monotonic()+3.5
                    while time.monotonic()<end:
                        with s.lock:
                            s.metadata={'arrival':time.monotonic()};s.target_id=1;s.vision_state='tracking'
                            s.tracking={'box':[400,200,650,450],'offset_px':[180,dy],'timestamp':time.monotonic(),'method':'flow'}
                        if any(f[8] and f[4]!=78 for f in mcu.commands) and abs(mcu.b-round(start_b))<.01:break
                        time.sleep(.02)
                    enabled=[f for f in mcu.commands if f[8]]
                    self.assertTrue(enabled,s.reason)
                    self.assertEqual(list(enabled[0][4:7]),[78,round(start_b),10])
                    self.assertTrue(any(f[4]!=78 for f in enabled),mcu.commands)
                    self.assertFalse(any(not f[8] for f in mcu.commands),mcu.commands)
                    # After normalization, independent B corrections may already
                    # have started despite horizontal error (new parallel policy).
                    later=[f for f in enabled[1:] if f[5]!=round(start_b)]
                    self.assertTrue(all((f[5]-round(start_b))*dy>0 for f in later))
                    self.assertTrue(0<=mcu.b<=96)
                    self.assertTrue(s.active,s.reason)
                finally:s.stop_event.set();worker.join(4);mcu.close();ref.directory.cleanup()

    def test_automatic_start_loss_reacquire_and_sticky_stop(self):
        ref=Reference();mcu=MCU()
        def factory(limits,log):return ContinuousArmPort(limits,log,port=mcu.path,sender_factory=Sender,check_holders=False)
        s=Session(ref.directory.name,ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS,port_factory=factory)
        worker=threading.Thread(target=s.arm_loop);worker.start()
        def feed(seconds,dx=None,identity=1,age=0):
            until=time.monotonic()+seconds
            while time.monotonic()<until:
                with s.lock:
                    s.metadata={'arrival':time.monotonic()};s.target_id=identity
                    s.tracking=None if dx is None else {'box':[400,200,650,450],'offset_px':[dx,0],'timestamp':time.monotonic()-age,'method':'flow'}
                    s.vision_state='searching' if dx is None else 'tracking'
                time.sleep(.02)
        try:
            feed(2.2,60);self.assertTrue(any(f[8] for f in mcu.commands),s.reason)
            n=len(mcu.commands);feed(.65,-100,age=.8)
            self.assertTrue(s.active);self.assertTrue(s.auto_follow)
            self.assertFalse(mcu.enabled);self.assertEqual(s.reason,'waiting_for_target_frame')
            self.assertFalse(any(f[8] for f in mcu.commands[n:]))
            n=len(mcu.commands);feed(.6,-60)
            self.assertTrue(any(f[8] for f in mcu.commands[n:]),s.reason)
            self.assertNotEqual(s.reason,'waiting_for_target_frame')
            feed(.5);self.assertTrue(s.active);self.assertFalse(mcu.enabled)
            n=len(mcu.commands);feed(.6,-60,2)
            self.assertTrue(any(f[8] for f in mcu.commands[n:]));self.assertTrue(s.active,s.reason)
            s.command('stop',{});feed(.4,-60,2);n=len(mcu.commands)
            feed(.3,80,3);self.assertFalse(s.active);self.assertFalse(s.auto_follow)
            self.assertFalse(mcu.enabled);self.assertEqual(n,len(mcu.commands))
        finally:s.stop_event.set();worker.join(timeout=4);mcu.close();ref.directory.cleanup()
    def test_camera_frame_delay_keeps_lock_and_automatic_intent(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup)
        s=Session(ref.directory.name,ref.limits,smooth=True,automatic=True,intrinsics=INTRINSICS)
        image=test_tracking.TrackingTests.scene();identities=[]
        class Capture:
            count=0
            def read(self):
                # Model a 30 Hz camera. CPU speed must not collapse the entire
                # acquisition window below the tracker's temporal confirmation.
                time.sleep(1/30)
                self.count+=1
                if self.count==14:
                    identities.append(s.target_id)
                    raise RuntimeError('stale_camera')
                if self.count==15:
                    self_wait=s.reason
                    assert self_wait=='waiting_for_target_frame',self_wait
                    assert s.tracking is None
                if self.count>=22:s.stop_event.set()
                return image,{'arrival':time.monotonic(),'frame_id':self.count,'detections':[{'left':480,'top':250,'right':720,'bottom':450,'score':.9}]}
        s.active=True
        worker=threading.Thread(target=s.vision_loop,args=(Capture(),));worker.start();worker.join(5)
        s.stop_event.set();worker.join(2)
        self.assertFalse(worker.is_alive());self.assertTrue(s.active,s.reason);self.assertTrue(s.auto_follow)
        self.assertTrue(identities and identities[0]>0);self.assertEqual(s.target_id,identities[0])
        self.assertIsNotNone(s.tracking)
    def test_observe_only_cannot_automatically_open_serial(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup);opened=[]
        s=Session(ref.directory.name,ref.limits,smooth=True,automatic=True,motion_allowed=False,intrinsics=INTRINSICS,port_factory=lambda *a:opened.append(True))
        s.metadata={'arrival':time.monotonic()}
        worker=threading.Thread(target=s.arm_loop);worker.start();time.sleep(.1);s.stop_event.set();worker.join(2)
        self.assertFalse(s.active);self.assertFalse(s.auto_follow);self.assertEqual(opened,[])

if __name__=='__main__':unittest.main()
