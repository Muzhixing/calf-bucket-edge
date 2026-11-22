import json,math,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import cv2,numpy as np
from ego import BackgroundMotion
from response import ResponseProfile,response_angles,fit_response
from control import FollowPolicy
from test_tracking import Reference
from test_continuous import INTRINSICS
import test_alignment


def background():
    rng=np.random.default_rng(63)
    small=rng.integers(20,230,(180,320,3),dtype=np.uint8)
    return cv2.resize(cv2.GaussianBlur(small,(3,3),.6),(1280,720))


def calibration_data():
    stamps=np.arange(0,12.01,.025)
    # Bidirectional ramps and holds, all inside the accepted A envelope.
    angle=80+5*np.sin(stamps*2*np.pi*.42)+1.3*np.sin(stamps*2*np.pi*1.1)
    commands=np.c_[stamps,angle]
    seen=np.arange(.8,11.9,1/30)
    x=24*response_angles(commands,seen,.09,.075)-1300
    x+=np.random.default_rng(5).normal(0,.15,len(x))
    return dict(base_stationary=True,target_stationary=True,commands=commands.tolist(),observations=np.c_[seen,x].tolist())


class BackgroundTests(unittest.TestCase):
    def test_camera_translation_uses_background_not_moving_bucket(self):
        im=background();im[220:470,470:770]=200
        moved=cv2.warpAffine(im,np.float32([[1,0,8],[0,1,4]]),(1280,720))
        moved[210:480,420:800]=60  # Independent foreground motion stays excluded.
        b=BackgroundMotion();self.assertFalse(b.update(im,1.,[[470,220,770,470]])['valid'])
        r=b.update(moved,1.05,[[420,210,800,480]])
        self.assertTrue(r['valid'],r)
        self.assertAlmostEqual(r['matrix'][0][2],8,delta=1.)
        self.assertAlmostEqual(r['matrix'][1][2],4,delta=1.)

    def test_local_texture_and_blank_frame_cannot_prove_camera_motion(self):
        im=np.zeros((720,1280,3),np.uint8)+60
        im[:200,:250]=background()[:200,:250]
        b=BackgroundMotion();b.update(im,1.,[[470,220,770,470]])
        self.assertFalse(b.update(im,1.05,[[470,220,770,470]])['valid'])
        self.assertIsNone(b.update(im,1.10,[]))
        self.assertFalse(b.update(im,1.15,[[470,220,770,470]])['valid'])

    def test_duplicate_gap_and_target_loss_reset_evidence(self):
        im=background();b=BackgroundMotion();boxes=[[470,220,770,470]]
        b.update(im,1.,boxes);self.assertIsNone(b.update(im,1.,boxes))
        self.assertEqual(b.update(im,1.4,boxes)['reason'],'frame_gap')
        self.assertIsNone(b.update(im,1.5,[]));self.assertIsNone(b.previous)

    def test_disagreeing_background_planes_are_rejected(self):
        im=background();moved=im.copy()
        moved[:360]=cv2.warpAffine(im,np.float32([[1,0,16],[0,1,0]]),(1280,720))[:360]
        moved[360:]=cv2.warpAffine(im,np.float32([[1,0,-16],[0,1,0]]),(1280,720))[360:]
        b=BackgroundMotion();boxes=[[470,220,770,470]];b.update(im,1.,boxes)
        self.assertFalse(b.update(moved,1.05,boxes)['valid'])


class CompensationPolicyTests(unittest.TestCase):
    def setUp(self):self.ref=Reference();self.addCleanup(self.ref.directory.cleanup)

    def target(self,i,dx,shift,valid=True):
        return dict(box=[500,220,760,480],offset_px=[dx,0],timestamp=1+i/30,method='flow',quality=.95,
                    ego_motion=dict(valid=valid,confidence=.95,timestamp=1+i/30,previous_timestamp=1+(i-1)/30,
                                    matrix=[[1,0,shift],[0,1,0]]))

    def test_camera_motion_does_not_become_target_feedforward(self):
        enhanced=FollowPolicy(self.ref.limits,**INTRINSICS,response_profile=ResponseProfile());original=FollowPolicy(self.ref.limits,**INTRINSICS)
        for i in range(35):
            t=self.target(i,-80+i*4,4);enhanced.observe(t,80)
            original.observe({k:v for k,v in t.items() if k!='ego_motion'},80)
        self.assertEqual(enhanced.diagnostics['velocity_source'],'background')
        self.assertAlmostEqual(enhanced.velocity,0.,delta=.1)
        self.assertGreater(abs(original.velocity),4)

    def test_real_motion_survives_camera_compensation_and_reversal(self):
        p=FollowPolicy(self.ref.limits,**INTRINSICS,response_profile=ResponseProfile())
        for i in range(30):p.observe(self.target(i,-80+i*4,1),80)
        self.assertLess(p.velocity,-3)
        for i in range(30,50):p.observe(self.target(i,36-(i-29)*4,1),80)
        self.assertGreater(p.velocity,3)
        t=self.target(50,-48,1,False);p.observe(t,80)
        self.assertEqual(p.diagnostics['velocity_source'],'telemetry')
        self.assertIsNone(p.visual_motion.stamp)

    def test_old_or_malformed_background_cannot_extend_prediction(self):
        p=FollowPolicy(self.ref.limits,**INTRINSICS,response_profile=ResponseProfile())
        p.observe(self.target(0,80,4),80)
        t=self.target(1,84,4);t['ego_motion']['timestamp']=1.
        p.observe(t,80);self.assertEqual(p.diagnostics['velocity_source'],'telemetry')
        t=self.target(2,88,4);t['ego_motion']['matrix'][0][2]=float('nan')
        p.observe(t,80);self.assertEqual(p.diagnostics['velocity_source'],'telemetry')
        t=self.target(3,92,4);t['ego_motion']['previous_timestamp']=None
        p.observe(t,80);self.assertEqual(p.diagnostics['velocity_source'],'telemetry')

    def test_uncalibrated_background_is_diagnostic_only(self):
        p=FollowPolicy(self.ref.limits,**INTRINSICS)
        old=FollowPolicy(self.ref.limits,**INTRINSICS)
        for i in range(35):
            t=self.target(i,-80+i*4,4)
            p.observe(t,80);old.observe({k:v for k,v in t.items() if k!='ego_motion'},80)
            self.assertEqual(p.velocity,old.velocity)
            self.assertEqual(p.diagnostics['velocity_source'],'telemetry')

    def test_background_evidence_never_overrides_center_hold_or_target_loss(self):
        original=test_alignment.target
        def with_background(dx,dy,stamp):
            t=original(dx,dy,stamp)
            t['ego_motion']=dict(valid=True,confidence=1.,timestamp=stamp,previous_timestamp=stamp-.02,
                                 matrix=[[1,0,15],[0,1,0]])
            return t
        def exercise(s,mcu,feed):
            feed(2.2,5,5);self.assertEqual(mcu.commands,[])
            feed(.7,65);self.assertTrue(any(f[8] for f in mcu.commands))
            feed(.4,5,5);self.assertFalse(mcu.enabled)
            n=len(mcu.commands);feed(.3,5,5);self.assertEqual(len(mcu.commands),n)
            with s.lock:s.tracking=None
            feed(.3,65,age=1.);self.assertFalse(mcu.enabled)
        policy=FollowPolicy
        def calibrated(*args,**kwargs):
            kwargs['response_profile']=ResponseProfile()
            return policy(*args,**kwargs)
        with patch('test_alignment.target',side_effect=with_background),patch('runtime.FollowPolicy',side_effect=calibrated):
            test_alignment.AlignmentTests().run_session(exercise)


class ResponseTests(unittest.TestCase):
    def test_causal_step_delay_and_lag(self):
        a=response_angles([(0,80),(1,90),(2,70)],[.9,1.05,1.2,1.3],.1,.1)
        np.testing.assert_allclose(a[:2],[80,80])
        self.assertAlmostEqual(a[2],90-10/math.e,places=6)
        self.assertLess(a[2],a[3]);self.assertLess(a[3],90)

    def test_identification_recovers_heldout_response_and_rejects_no_excitation(self):
        data=calibration_data();fit=fit_response(data)
        self.assertTrue(fit['validation']['accepted'],fit)
        self.assertAlmostEqual(fit['gain_px_per_degree'],24,delta=.2)
        self.assertAlmostEqual(fit['delay_s'],.09,delta=.02)
        self.assertAlmostEqual(fit['tau_s'],.075,delta=.03)
        data['target_stationary']=False
        with self.assertRaises(ValueError):fit_response(data)
        data=calibration_data();data['commands']=[[t,80] for t,a in data['commands']]
        with self.assertRaises(ValueError):fit_response(data)

    def test_validation_rejects_drift_not_explained_by_actuator(self):
        data=calibration_data()
        for i,row in enumerate(data['observations']):
            if i>.7*len(data['observations']):row[1]+=30
        self.assertFalse(fit_response(data)['validation']['accepted'])

    def test_profile_provenance_calibration_and_scope(self):
        scope=dict(B_mm=[25,35],C_mm=10,distance_m=[.8,1.2])
        data=dict(schema=1,source='field_log',calibration_sha256='a'*64,scope=scope,
                  gain_px_per_degree=24,delay_s=.09,tau_s=.075,validation={'accepted':True})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'profile.json';path.write_text(json.dumps(data))
            p=ResponseProfile.load(path,'a'*64)
            self.assertTrue(p.eligible({'lift_position_mm':30,'slide_position_mm':10},{'distance_m':1.,'distance_age_ms':20}))
            self.assertFalse(p.eligible({'lift_position_mm':50,'slide_position_mm':10},{'distance_m':1.,'distance_age_ms':20}))
            with self.assertRaises(ValueError):ResponseProfile.load(path,'b'*64)
            data['source']='simulation';path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):ResponseProfile.load(path,'a'*64)
            data['source']='field_log';data['calibration_sha256']=None
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):ResponseProfile.load(path,None)

    def test_model_angle_uses_history_and_scope_switch_resets_predictor(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup)
        r=ResponseProfile(24,.09,.075,dict(B_mm=[25,35],C_mm=10,distance_m=[.8,1.2]))
        p=FollowPolicy(ref.limits,**INTRINSICS,response_profile=r)
        class Port:
            history=[(0,80),(1,90),(2,70)]
            def angle_at(self,stamp):return 77
        self.assertLess(p.frame_angle(Port(),1.2),90)
        t=dict(box=[500,220,760,480],offset_px=[80,0],timestamp=1.,method='flow')
        p.observe(t,80)
        p.configure_context({'lift_position_mm':50,'slide_position_mm':10},{},t)
        self.assertEqual(p.feedback_delay,.14);self.assertIsNone(p.stamp)
        self.assertEqual(p.frame_angle(Port(),1.2),77)

if __name__=='__main__':unittest.main()
