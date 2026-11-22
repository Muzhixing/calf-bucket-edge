import unittest
from unittest.mock import patch
import numpy as np
import cv2
import test_tracking
from test_visual_robust import detection
from vision import TargetTracker,TargetLock
from motion import MotionFilter
from control import FollowPolicy


class PredictiveTests(unittest.TestCase):
    def test_cv_motion_filter_direction_reversal_and_jitter(self):
        m=MotionFilter();speeds=[]
        for i in range(35):
            t=i/30;m.update([80+8*t],t);speeds.append(float(m.reliable_velocity(25)[0]))
        self.assertGreater(speeds[-1],6)
        for i in range(1,12):m.update([80+8*34/30-8*i/30],(34+i)/30)
        self.assertLess(m.reliable_velocity(25)[0],-5)
        m=MotionFilter()
        for i in range(100):m.update([80+.03*(-1)**i],i/30)
        self.assertEqual(m.reliable_velocity(25)[0],0)

    def test_single_outlier_cannot_create_large_direction_prediction(self):
        m=MotionFilter()
        for i in range(20):m.update([80.],i/30)
        self.assertFalse(m.update([90.],20/30))
        self.assertEqual(m.reliable_velocity(25)[0],0)
        self.assertLess(abs(m.position[0]-80),.1)
        self.assertTrue(m.update([80.],21/30))
        self.assertLess(abs(m.position[0]-80),.1)

    def test_motion_gap_and_duplicate_cannot_extend_prediction(self):
        m=MotionFilter();m.update([80],1.);m.update([81],1.1)
        before=m.position.copy();self.assertFalse(m.update([100],1.1))
        np.testing.assert_array_equal(m.position,before)
        m.update([83],2.);self.assertEqual(m.velocity[0],0)

    def test_low_scores_cannot_initialize_or_reacquire(self):
        image=test_tracking.TrackingTests.scene();box=[480,250,720,450]
        manager=TargetLock()
        for i in range(20):
            self.assertIsNone(manager.update(image,{'arrival':1+i*.04,'detections':[],
                                                   'tracking_candidates':[detection(box,.4)]}))
        manager.select(image,box,2.)
        self.assertIsNone(manager.update(np.zeros_like(image)+25,{'arrival':2.04,'detections':[]}))
        # A different object with weak model hits cannot establish an identity.
        wrong=image.copy();wrong[250:450,480:720]=255-wrong[250:450,480:720]
        for i in range(10):
            self.assertIsNone(manager.update(wrong,{'arrival':2.08+i*.04,'detections':[],
                                                   'tracking_candidates':[detection(box,.4)]}))

    def test_moving_initial_lock_uses_geometry_to_confirm_medium_detections(self):
        image=test_tracking.TrackingTests.scene();manager=TargetLock();result=None
        for i in range(6):
            moved=cv2.warpAffine(image,np.float32([[1,0,i*5],[0,1,0]]),(1280,720))
            result=manager.update(moved,{'arrival':1+i*.04,'detections':
                [detection([480+i*5,250,720+i*5,450],.61)] if i<3 else []})
            if result is not None:break
        self.assertIsNotNone(result);self.assertEqual(result['method'],'detector_flow_lock')
        self.assertLessEqual(result['timestamp']-1,.161)

    def test_initial_single_false_hit_and_occlusion_do_not_lock(self):
        image=test_tracking.TrackingTests.scene();manager=TargetLock();box=[480,250,720,450]
        self.assertIsNone(manager.update(image,{'arrival':1,'detections':[detection(box,.9)]}))
        for i in range(1,12):
            self.assertIsNone(manager.update(np.zeros_like(image)+25,{'arrival':1+i*.04,'detections':[]}))

    def test_two_stage_association_rejects_ambiguous_and_prefers_strong(self):
        image=test_tracking.TrackingTests.scene();box=[480,250,720,450];tracker=TargetTracker(image,box,1.)
        weak={**detection(box,.4),'continuation_only':True};strong=detection(box,.9)
        self.assertIs(tracker.associated_detection([weak,strong],box),strong)
        self.assertIs(tracker.associated_detection([weak],box),weak)
        self.assertIsNone(tracker.associated_detection([weak,dict(weak)],box))
        self.assertIsNone(tracker.associated_detection([detection(box,.4)],box))
        self.assertIsNone(tracker.associated_detection([{**weak,'left':700,'right':940}],box))

    def test_low_association_cannot_train_reference_or_hide_occlusion(self):
        image=test_tracking.TrackingTests.scene();box=[480,250,720,450];tracker=TargetTracker(image,box,1.)
        weak={**detection(box,.4),'continuation_only':True}
        for i in range(1,20):tracker.update(image,1+i*.04,[weak])
        self.assertEqual(len(tracker.references),1)
        with self.assertRaises(RuntimeError):tracker.update(np.zeros_like(image)+25,1.8,[weak])

    def test_weak_detections_support_verified_blur_beyond_short_grace(self):
        image=test_tracking.TrackingTests.scene();box=[480,250,720,450]
        tracker=TargetTracker(image,box,1.)
        blurred=cv2.GaussianBlur(image,(11,11),2.5)
        weak={**detection(box,.4),'continuation_only':True};methods=[]
        for i in range(1,10):methods.append(tracker.update(blurred,1+i*.04,[weak])['method'])
        self.assertIn('flow_supported',methods[5:])
        self.assertEqual(len(tracker.references),1)
        with self.assertRaises(RuntimeError):tracker.update(np.zeros_like(image)+25,1.4,[weak])

    def test_A_priority_defers_B_during_horizontal_motion(self):
        ref=test_tracking.Reference();self.addCleanup(ref.directory.cleanup)
        p=FollowPolicy(ref.limits,690.8,690.8,695.7,374.9)
        t={'offset_px':[0,100]}
        p.velocity=8
        for now in (1.,1.2,1.4):self.assertFalse(p.allow_b(t,now))
        p.velocity=0;self.assertFalse(p.allow_b(t,1.5));self.assertTrue(p.allow_b(t,1.65))
        t['offset_px'][0]=50;self.assertFalse(p.allow_b(t,1.7))

    def test_untextured_false_detections_cannot_escape_automatic_reacquisition(self):
        image=test_tracking.TrackingTests.scene();manager=TargetLock();manager.select(image,[480,250,720,450],1.)
        for i in range(20):
            self.assertIsNone(manager.update(np.zeros_like(image)+25,{'arrival':1.04+i*.04,
                'detections':[detection([500,250,740,450],.95)]}))


if __name__=='__main__':unittest.main()
