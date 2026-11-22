import unittest

import cv2
import numpy as np
import test_tracking
from vision import TargetLock
from test_visual_robust import detection


class ReappearanceTests(unittest.TestCase):
    def lost(self):
        image=test_tracking.TrackingTests.scene()
        manager=TargetLock();manager.select(image,[480,250,720,450],1.)
        blank=np.zeros_like(image)+25
        self.assertIsNone(manager.update(blank,{'arrival':1.05,'detections':[]}))
        return image,blank,manager

    def moved(self,image,dx):
        return cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,0]]),(1280,720)),[480+dx,250,720+dx,450]

    def test_single_known_detection_then_fresh_motion_relocks(self):
        for score in (.9,.7,.61):
            with self.subTest(score=score):
                image,blank,manager=self.lost();generation=manager.generation
                moved,box=self.moved(image,-300)
                self.assertIsNone(manager.update(moved,{'arrival':1.2,'detections':[detection(box,score)]}))
                self.assertIsNotNone(manager.provisional)
                result=None
                for i in range(1,5):
                    moved,box=self.moved(image,-300+3*i)
                    result=manager.update(moved,{'arrival':1.2+i*.04,'detections':[]})
                    if result is not None:break
                self.assertIsNotNone(result);self.assertEqual(result['method'],'appearance_relock')
                self.assertEqual(result['timestamp'],1.2+i*.04)
                self.assertLessEqual(result['candidate_verify_s'],.161)
                self.assertEqual(manager.generation,generation+1)
                self.assertLess(abs(result['offset_px'][0]-(-340+3*i)),2)

    def test_occlusion_after_candidate_never_publishes_target(self):
        image,blank,manager=self.lost();moved,box=self.moved(image,-300)
        manager.update(moved,{'arrival':1.2,'detections':[detection(box)]})
        self.assertIsNotNone(manager.provisional)
        for stamp in (1.24,1.28,1.4,1.8):
            self.assertIsNone(manager.update(blank,{'arrival':stamp,'detections':[detection(box)]}))
            self.assertIsNone(manager.provisional)

    def test_wrong_appearance_and_two_matching_buckets_do_not_seed(self):
        image,blank,manager=self.lost();moved,box=self.moved(image,-300)
        wrong=moved.copy();wrong[250:450,180:420]=np.random.default_rng(4).integers(0,255,(200,240,3),dtype=np.uint8)
        self.assertIsNone(manager.update(wrong,{'arrival':1.2,'detections':[detection(box)]}))
        self.assertIsNone(manager.provisional)
        twin=moved.copy();twin[250:450,850:1090]=image[250:450,480:720]
        self.assertIsNone(manager.update(twin,{'arrival':1.25,'detections':[detection(box),detection([850,250,1090,450])]}))
        self.assertIsNone(manager.provisional)

    def test_duplicate_frames_and_frame_gap_cannot_verify(self):
        image,blank,manager=self.lost();moved,box=self.moved(image,-300)
        manager.update(moved,{'arrival':1.2,'detections':[detection(box)]})
        for stamp in (1.2,1.19,1.2):
            self.assertIsNone(manager.update(moved,{'arrival':stamp,'detections':[]}))
        self.assertEqual(manager.provisional['frames'],1)
        self.assertIsNone(manager.update(moved,{'arrival':1.34,'detections':[]}))
        self.assertIsNone(manager.provisional)

    def test_memory_survives_search_timeout_but_is_bounded(self):
        for return_at,expected in ((6.3,True),(16.4,False)):
            with self.subTest(return_at=return_at):
                image,blank,manager=self.lost()
                self.assertIsNone(manager.update(blank,{'arrival':6.2,'detections':[]}))
                self.assertEqual(manager.state,'searching');self.assertIsNotNone(manager.memory)
                moved,box=self.moved(image,-300)
                self.assertIsNone(manager.update(moved,{'arrival':return_at,'detections':[detection(box)]}))
                result=None
                for dt in (.04,.08):result=manager.update(moved,{'arrival':return_at+dt,'detections':[]})
                self.assertEqual(result is not None,expected)

    def test_manual_mode_does_not_automatically_adopt_detection(self):
        image,blank,manager=self.lost();manager.mode='manual';moved,box=self.moved(image,-300)
        self.assertIsNone(manager.update(moved,{'arrival':1.2,'detections':[detection(box)]}))
        self.assertIsNone(manager.provisional)

    def test_changed_view_uses_sparse_detection_and_fresh_geometry(self):
        for score,hits in ((.7,2),(.61,3)):
            with self.subTest(score=score):
                image,blank,manager=self.lost()
                changed=image.copy();changed[250:450,480:720]=image[250:450,480:720][::-1,::-1]
                result=None
                for i in range(6):
                    moved,box=self.moved(changed,i*3)
                    result=manager.update(moved,{'arrival':1.2+i*.04,
                                                'detections':[detection(box,score)] if i<hits else []})
                    if result is not None:break
                self.assertIsNotNone(result);self.assertEqual(result['method'],'detector_flow_relock')
                self.assertEqual(result['candidate_detection_votes'],hits)
                self.assertGreaterEqual(result['candidate_verify_s'],.12)
                self.assertLessEqual(result['candidate_verify_s'],.161)

    def test_changed_view_single_hit_and_occlusion_cannot_confirm(self):
        for case in ('single','occlusion'):
            with self.subTest(case=case):
                image,blank,manager=self.lost()
                changed=image.copy();changed[250:450,480:720]=image[250:450,480:720][::-1,::-1]
                box=[480,250,720,450]
                self.assertIsNone(manager.update(changed,{'arrival':1.2,'detections':[detection(box,.7)]}))
                self.assertFalse(manager.provisional['known'])
                for i in range(1,9):
                    frame=blank if case=='occlusion' else changed
                    ds=[detection(box,.61)] if case=='occlusion' else []
                    self.assertIsNone(manager.update(frame,{'arrival':1.2+i*.04,'detections':ds}))
                self.assertEqual(manager.generation,1)


if __name__=='__main__':unittest.main()
