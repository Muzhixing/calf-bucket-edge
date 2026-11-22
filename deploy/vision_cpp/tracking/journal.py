"""Bounded asynchronous diagnostics; disk latency never holds control locks."""
from collections import deque
import json
from pathlib import Path
import threading
import time


class AsyncJournal:
    SAMPLES={'tracking_sample','preview_sample'}

    def __init__(self,folder,sample_limit=512,limit=2048,writer=None):
        self.folder=Path(folder);self.sample_limit=sample_limit;self.limit=limit
        self.jobs=deque();self.condition=threading.Condition();self.closed=False
        self.error=None;self.dropped_samples=0;self.dropped_images=0;self.images=0
        self.max_pending=0;self.max_write_ms=0.;self.writer=writer
        self.thread=threading.Thread(target=self.run,name='diagnostic-writer',daemon=True)
        self.thread.start()

    def append(self,event,data):
        # Payloads are completed observation snapshots, never live tracker objects.
        record={'timestamp':time.monotonic(),'event':event,'data':data}
        with self.condition:
            if self.closed:return
            low_priority=event in self.SAMPLES or event=='operator_command' and data.get('action')=='heartbeat'
            if low_priority and len(self.jobs)>=self.sample_limit:
                self.dropped_samples+=1;return
            if len(self.jobs)>=self.limit:
                self.error='diagnostic_critical_queue_full';return
            self.jobs.append(('event',record));self.max_pending=max(self.max_pending,len(self.jobs))
            self.condition.notify()

    def image(self,name,image):
        with self.condition:
            if self.closed or self.images>=4 or len(self.jobs)>=self.sample_limit:
                self.dropped_images+=1;return
            self.jobs.append(('image',(name,image)));self.images+=1;self.condition.notify()

    def run(self):
        try:
            with (self.folder/'tracking_events.jsonl').open('a') as stream:
                last_flush=time.monotonic()
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda:self.jobs or self.closed,timeout=.2)
                        if not self.jobs:
                            if self.closed:break
                            job=None
                        else:
                            job=self.jobs.popleft()
                            if job[0]=='image':self.images-=1
                    began=time.monotonic()
                    if job is not None:
                        kind,value=job
                        if kind=='event':
                            line=json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n'
                            (self.writer or stream.write)(line)
                        else:
                            import cv2
                            if not cv2.imwrite(str(self.folder/value[0]),value[1]):
                                self.dropped_images+=1
                    now=time.monotonic()
                    if now-last_flush>=.2:stream.flush();last_flush=now
                    self.max_write_ms=max(self.max_write_ms,(time.monotonic()-began)*1000)
        except Exception as exc:self.error='diagnostic_writer_failed: '+str(exc)

    def status(self):
        with self.condition:
            return {'pending':len(self.jobs),'max_pending':self.max_pending,
                    'dropped_samples':self.dropped_samples,'dropped_images':self.dropped_images,
                    'error':self.error,'max_write_ms':self.max_write_ms}

    def close(self,timeout=3.):
        with self.condition:self.closed=True;self.condition.notify_all()
        self.thread.join(timeout)
        return not self.thread.is_alive()
