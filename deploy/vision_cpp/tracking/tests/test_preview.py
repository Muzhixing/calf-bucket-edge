import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
import cv2
from unittest.mock import patch

import numpy as np
from preview import LatestFrames
from runtime import Session, handler_for
from test_tracking import Reference


class PreviewTests(unittest.TestCase):
    def test_capture_failure_stops_intent_without_waiting_for_tracker(self):
        for error,active,had_frame,expected in (
            ('camera disconnected',True,True,False),
            ('camera disconnected',False,True,False),
            ('stale_camera',True,True,True),
            ('connection refused',False,False,True),
        ):
            with self.subTest(error=error,active=active,had_frame=had_frame):
                ref=Reference();self.addCleanup(ref.directory.cleanup)
                s=Session(ref.directory.name,ref.limits,smooth=True,automatic=True)
                s.active=active;s.tracking={'timestamp':time.monotonic()}
                if had_frame:s.preview_metadata={'frame_id':1}
                class FailedCapture:
                    def read(self):
                        s.stop_event.set()
                        raise RuntimeError(error)
                s.capture_loop(FailedCapture(),LatestFrames())
                self.assertIsNone(s.tracking)
                self.assertEqual(s.auto_follow,expected)
                self.assertEqual(s.active,active and expected)

    def test_readers_skip_backlog_independently_and_failure_invalidates_frame(self):
        h=LatestFrames();a=h.reader();b=h.reader();image=np.zeros((10,10,3),np.uint8)
        for i in range(1,4):h.publish(image,{'frame_id':i,'arrival':time.monotonic()})
        self.assertEqual(a.read()[1]['frame_id'],3);self.assertEqual(b.read()[1]['frame_id'],3)
        with self.assertRaisesRegex(RuntimeError,'frame_not_advanced'):h.next(3,timeout=0)
        h.fail('camera disconnected')
        with self.assertRaisesRegex(RuntimeError,'camera disconnected'):a.read()
        h.publish(image,{'frame_id':4,'arrival':time.monotonic()})
        self.assertEqual(a.read()[1]['frame_id'],4)
        h.publish(image,{'frame_id':5,'arrival':time.monotonic()-1})
        with self.assertRaisesRegex(RuntimeError,'stale_camera'):b.read()

    def test_preview_keeps_advancing_when_tracker_blocks_and_draws_no_circles(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup)
        s=Session(ref.directory.name,ref.limits,motion_allowed=False,smooth=True)
        h=LatestFrames();image=np.zeros((720,1280,3),np.uint8)+60
        class SlowManager:
            def __init__(self,mode):
                self.reacquired=self.generation=0;self.state='searching';self.reason=''
                self.tracker=self.provisional=None;self.expired=False
            def update(self,image,meta):time.sleep(.3);return None
        with patch('runtime.TargetLock',SlowManager),patch('runtime.cv2.circle') as circle:
            threads=[threading.Thread(target=s.vision_loop,args=(h.reader(),)),
                     threading.Thread(target=s.preview_loop,args=(h,))]
            for t in threads:t.start()
            try:
                for i in range(35):
                    h.publish(image,{'frame_id':i+1,'arrival':time.monotonic(),'detections':[]})
                    time.sleep(1/30)
                self.assertGreater(s.video_id,30)
                self.assertGreater(s.video_fps,20)
                self.assertLess(s.track_fps,5)
                self.assertIsNotNone(s.jpeg);circle.assert_not_called()
                self.assertIsNotNone(s.remote_jpeg)
                self.assertEqual(cv2.imdecode(np.frombuffer(s.remote_jpeg,np.uint8),cv2.IMREAD_COLOR).shape,(540,960,3))
                self.assertFalse(s.active)
            finally:
                s.stop_event.set()
                for t in threads:t.join(2)
                self.assertTrue(all(not t.is_alive() for t in threads))

    def test_remote_snapshot_is_latest_only_and_rejects_stale_image(self):
        ref=Reference();self.addCleanup(ref.directory.cleanup)
        s=Session(ref.directory.name,ref.limits,motion_allowed=False)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(s))
        worker=threading.Thread(target=server.serve_forever);worker.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}/frame.jpg?remote=1'
            for n in range(10):
                s.remote_jpeg=f'frame{n}'.encode();s.remote_video_id=n;s.remote_arrival=time.monotonic()
            with urllib.request.urlopen(url) as response:
                self.assertEqual(response.read(),b'frame9')
                self.assertEqual(response.headers['X-Frame-Id'],'9')
                self.assertLess(float(response.headers['X-Frame-Age-Ms']),400)
            s.remote_arrival=time.monotonic()-5
            with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(url)
            self.assertEqual(error.exception.code,503)
        finally:
            server.shutdown();server.server_close();worker.join(2)
