"""One latest-frame slot shared by independent tracking and preview readers."""
import threading
import time


class LatestFrames:
    def __init__(self):
        self.condition=threading.Condition();self.item=None;self.error=None

    def publish(self,image,metadata):
        with self.condition:
            self.item=(image,metadata);self.error=None;self.condition.notify_all()

    def fail(self,error):
        with self.condition:
            self.item=None;self.error=str(error);self.condition.notify_all()

    def next(self,previous,timeout=.15):
        with self.condition:
            self.condition.wait_for(lambda:self.error is not None or
                self.item is not None and self.item[1]['frame_id']!=previous,timeout=timeout)
            if self.error is not None:raise RuntimeError(self.error)
            if self.item is None or self.item[1]['frame_id']==previous:raise RuntimeError('frame_not_advanced')
            if not 0<=time.monotonic()-self.item[1]['arrival']<=.4:raise RuntimeError('stale_camera')
            return self.item

    def reader(self):
        hub=self
        class Reader:
            last=-1
            def read(self):
                item=hub.next(self.last);self.last=item[1]['frame_id'];return item
        return Reader()
