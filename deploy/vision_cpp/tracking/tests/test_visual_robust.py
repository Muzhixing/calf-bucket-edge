import unittest
from unittest.mock import patch

import cv2
import numpy as np
import test_tracking
from vision import TargetTracker


def detection(box,score=.9):
    return dict(zip(('left','top','right','bottom','score'),[*box,score]))


def moving_frames():
    """Fresh translated, mildly sheared and intermittently blurred object."""
    original=test_tracking.TrackingTests.scene()
    corners=np.float32([[480,250],[720,250],[720,450],[480,450]])
    for i in range(36):
        dx=35*np.sin(i*.16);dy=12*np.sin(i*.25);shear=.035*np.sin(i*.12)
        matrix=np.float32([[1,shear,dx-shear*350],[0,1,dy]])
        image=cv2.warpAffine(original,matrix,(1280,720))
        if i%12 in (5,6,7):image=cv2.GaussianBlur(image,(11,11),2.5)
        pts=cv2.transform(corners[None],matrix)[0]
        box=np.r_[pts.min(axis=0),pts.max(axis=0)].round().astype(int).tolist()
        yield image,box,1+i*.05


class RobustVisualTests(unittest.TestCase):
    def test_moving_blurred_target_keeps_current_measurements(self):
        tracker=None;methods=[]
        for image,box,stamp in moving_frames():
            if tracker is None:tracker=TargetTracker(image,box,stamp);continue
            result=tracker.update(image,stamp,[detection(box)])
            methods.append(result['method'])
            self.assertEqual(result['timestamp'],stamp)
            expected=(np.array(box[:2])+box[2:])/2
            actual=np.array(result['offset_px'])+[640,360]
            self.assertLess(np.linalg.norm(expected-actual),14)
        self.assertTrue(any(m in ('flow_supported','flow_blur','detector_bridge') for m in methods),methods)

    def warm(self):
        image=test_tracking.TrackingTests.scene();box=[480,250,720,450]
        tracker=TargetTracker(image,box,1.)
        for stamp in (1.05,1.10):tracker.update(image,stamp,[detection(box)])
        return image,box,tracker

    def test_detector_bridge_is_fresh_bounded_and_does_not_train_reference(self):
        image,box,tracker=self.warm();references=[r.copy() for r in tracker.references]
        with patch.object(tracker,'flow_transform',side_effect=RuntimeError('target_lost_flow')), \
             patch.object(tracker,'measured_continuity',return_value=False), \
             patch.object(tracker,'local_search',side_effect=RuntimeError('target_appearance_lost')):
            for stamp in (1.15,1.20,1.25,1.30,1.40):
                result=tracker.update(image,stamp,[detection(box)])
                self.assertEqual(result['method'],'detector_bridge');self.assertEqual(result['timestamp'],stamp)
            with self.assertRaises(RuntimeError):tracker.update(image,1.50,[detection(box)])
        self.assertEqual(len(tracker.references),len(references))
        for old,new in zip(references,tracker.references):np.testing.assert_array_equal(old,new)

    def test_bridge_rejects_occlusion_wrong_appearance_and_ambiguous_detection(self):
        for kind in ('occlusion','wrong','ambiguous','missing'):
            with self.subTest(kind=kind):
                image,box,tracker=self.warm();ds=[detection(box)]
                if kind=='occlusion':image=np.zeros_like(image)+25
                if kind=='wrong':image[250:450,480:720]=np.random.default_rng(6).integers(0,255,(200,240,3),dtype=np.uint8)
                if kind=='ambiguous':ds.append(detection([500,250,740,450]))
                if kind=='missing':ds=[]
                with patch.object(tracker,'flow_transform',side_effect=RuntimeError('target_lost_flow')), \
                     patch.object(tracker,'local_search',side_effect=RuntimeError('target_appearance_lost')):
                    with self.assertRaises(RuntimeError):tracker.update(image,1.15,ds)
                self.assertEqual(tracker.stamp,1.10)

    def test_single_model_hit_cannot_enable_bridge(self):
        image=test_tracking.TrackingTests.scene();box=[480,250,720,450];tracker=TargetTracker(image,box,1.)
        with patch.object(tracker,'flow_transform',side_effect=RuntimeError('target_lost_flow')), \
             patch.object(tracker,'local_search',side_effect=RuntimeError('target_appearance_lost')):
            with self.assertRaises(RuntimeError):tracker.update(image,1.05,[detection(box)])

    def test_blur_without_detector_cannot_extend_its_own_grace_period(self):
        image,box,tracker=self.warm()
        gray=cv2.cvtColor(cv2.GaussianBlur(image,(11,11),2.5),cv2.COLOR_BGR2GRAY)
        for stamp in (1.15,1.20,1.25):
            result=tracker.accept(gray,np.eye(3),stamp,'flow',[])
            self.assertEqual(result['method'],'flow_blur')
        self.assertEqual(tracker.last_strict_at,1.10)
        with self.assertRaises(RuntimeError):tracker.accept(gray,np.eye(3),1.35,'flow',[])


if __name__=='__main__':unittest.main()
