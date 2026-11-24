import json,time,uuid,unittest
from pathlib import Path
from unittest.mock import patch
from test_firmware_boot75 import Firmware73Tests
from arm import Limits,pose
from control import FollowPolicy
from startup import PowerCycleStore

class BCSelftestTests(Firmware73Tests):
    def setUp(self):
        super().setUp()
        c=json.loads(self.ref.path.read_text())
        c['control_policy']['parallel_bc']=True
        c['operating_envelope']['C_command_mm']=[0,100]
        c['power_cycle_startup'].update(a_startup_selftest=True,bc_startup_selftest=True)
        c['tracking_motion']={'b_firmware_max_mm_s':12.5,'c_firmware_max_mm_s':800*81/2060,
                              'b_far_position_gain':1.,'b_fast_error_mm':12.5}
        self.ref.path.write_text(json.dumps(c));self.limits=Limits(self.ref.path)

    # Inherited generic tests expect an A-only sequence; test only the BC cases here.
    test_existing_A_sweep_still_finishes_at75_with_BC_stationary=None
    test_wait_then_register_once_without_redundant_A_or_wheel_enable=None

    def drive(self,predicate,timeout=12):
        deadline=time.monotonic()+timeout
        while not predicate() and time.monotonic()<deadline:self.seq.tick();time.sleep(.01)
        self.assertTrue(predicate(),self.seq.stage)

    def test_full_sequence_at_working_limits_then_return_with_one_registration(self):
        self.make_sequence();self.seq.started-=9
        self.drive(lambda:self.seq.stage=='bc_b_out')
        self.assertGreater(self.seq.port.deadline-self.seq.port.started,11)
        self.drive(lambda:self.seq.done)
        enabled=[list(f[4:7]) for f in self.mcu.commands if f[8]&1]
        self.assertEqual(enabled,[[0,0,0],[160,0,0],[73,0,0],[73,96,0],[73,0,0],[73,0,100],[73,0,0]])
        self.assertEqual(sum(f[8]==6 for f in self.mcu.commands),1)
        self.assertFalse(self.mcu.enabled);self.assertEqual(pose(self.seq.port.state),[73,0,0])
        self.seq.close();self.store.close()
        self.store=PowerCycleStore(Path(self.ref.directory.name)/'state.json',8,'test',boot_id='test')
        self.assertTrue(self.store.ready)
        with self.assertRaises(RuntimeError):self.store.before_zero()

    def test_cancel_during_outward_does_not_start_return_or_repeat_zero(self):
        self.make_sequence();self.seq.started-=9;self.drive(lambda:self.seq.stage=='bc_b_out')
        self.seq.close('cancelled');n=len(self.mcu.commands);time.sleep(.15)
        self.assertEqual(n,len(self.mcu.commands));self.assertFalse(self.mcu.enabled)
        self.assertEqual(sum(f[8]==6 for f in self.mcu.commands),1)
        self.store.close();self.store=PowerCycleStore(Path(self.ref.directory.name)/'state.json',8,'test',boot_id='test')
        self.assertFalse(self.store.ready)
        with self.assertRaises(RuntimeError):self.store.before_zero()

    def test_eight_second_gate_and_no_motion_when_checks_fail(self):
        self.make_sequence()
        self.seq.ready_check=lambda:{'ready':False};self.seq.started-=9
        self.drive(lambda:self.seq.port is not None)
        until=time.monotonic()+1.8
        while time.monotonic()<until:self.seq.tick();time.sleep(.01)
        self.assertEqual(self.mcu.commands,[])
        self.seq.ready_check=lambda:{'ready':True};self.seq.started=time.monotonic()
        for _ in range(5):self.seq.tick()
        self.assertEqual(self.mcu.commands,[]);self.assertEqual(self.seq.wait_seconds,8)

    def test_sweep_requires_parallel_firmware_and_only_approved_endpoints(self):
        c=json.loads(self.ref.path.read_text());c['control_policy']['parallel_bc']=False
        self.ref.path.write_text(json.dumps(c))
        with self.assertRaisesRegex(RuntimeError,'parallel_firmware'):Limits(self.ref.path)
        self.make_sequence();self.seq.started-=9;self.drive(lambda:self.seq.stage=='a')
        self.drive(lambda:self.seq.port.phase=='idle')
        # Sequence may already have advanced to BC; arbitrary coordinates still forbidden.
        with self.assertRaises(RuntimeError):self.seq.port.startup_move([73,90,100],bc_sweep=True)

    def test_reuse_live_origins_waits_eight_seconds_without_reregistering(self):
        from receive import decode
        from startup import StartupSequence,StartupPort
        from test_tracking import Sender
        self.mcu.zero_flags=True;self.mcu.fault=0
        self.store.adopt_registered(decode(self.mcu.frame()),73)
        self.limits.startup_policy['a_startup_selftest']=False
        factory=lambda limits,log:StartupPort(limits,log,port=self.mcu.path,sender_factory=Sender,check_holders=False)
        self.seq=StartupSequence(self.store,self.limits,lambda *a:None,factory)
        self.assertEqual(self.seq.wait_seconds,8)
        self.seq.tick();self.assertEqual(self.mcu.commands,[])
        self.seq.started-=9;self.drive(lambda:self.seq.done)
        self.assertFalse(any(f[8]==6 for f in self.mcu.commands))
        self.assertEqual([list(f[4:7]) for f in self.mcu.commands if f[8]&1],[[73,96,0],[73,0,0],[73,0,100],[73,0,0]])

    def test_fast_far_correction_preserves_deadband_bounds_and_near_damping(self):
        from test_power_cycle import INTRINSICS
        policy=FollowPolicy(self.limits,**INTRINSICS)
        t={'offset_px':[0,20],'box':[10,10,110,110]}
        self.assertEqual(policy.next_b(t,40,{}),60) # previous 0.7 factor would give54
        t['offset_px'][1]=15;self.assertEqual(policy.next_b(t,40,{}),40)
        t['offset_px'][1]=50;self.assertEqual(policy.next_b(t,90,{}),96)
        t['offset_px'][1]=-50;self.assertEqual(policy.next_b(t,3,{}),0)
        t['offset_px'][1]=20
        m={'distance_m':.15,'distance_age_ms':20,'selected':dict(zip(('left','top','right','bottom'),t['box']))}
        self.assertEqual(policy.next_b(t,40,m),44) # min gain .25, 5mm *.7 rounds4

class ArrivalTests(unittest.TestCase):
    def test_rounded_endpoint_waits_for_firmware_axis_completion(self):
        from arm import ArmPort
        now=time.monotonic();p=ArmPort.__new__(ArmPort);stopped=[]
        p.phase='moving';p.rx_at=now;p.expected=0;p.move_counter=1
        p.origin=[73,0,0];p.target=[73,96,100];p.started=now-1;p.deadline=now+2;p.arrived_at=now-.2
        s={'command_count_mod256':1,'result':0,'enabled':True,'state':2,
           'servo_software_deg':73,'lift_position_mm':96.,'slide_position_mm':100.}
        p.poll=lambda:[s];p.check_state=lambda s:None;p.stop=lambda:stopped.append(True)
        p.tick();self.assertEqual(stopped,[])
        s['state']=5;p.arrived_at=now-.2;p.tick();self.assertEqual(stopped,[True])
