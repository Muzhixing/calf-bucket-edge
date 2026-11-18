"""Instrument the original frozen Python modules. No camera, display or serial.

Async mode calls preview_pipeline.run unchanged with a recorded capture source.
Sync mode measures every pair for numerical/maximum-throughput comparison.
Imports at module scope deliberately install the same profiling in spawned workers.
"""
import argparse, collections, json, os, sys, threading, time
from pathlib import Path
ROOT=Path(os.environ.get('BUCKET_V2_ROOT','/home/elf/deploy/vision_v2')).resolve()
sys.path[:0]=[str(ROOT/'tools'),str(ROOT)]
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ['BUCKET_CALIBRATION_PROFILE']='v2_radial2'
os.environ['RKNN_MODEL_PATH']=str(ROOT/'models/seed2_REG_FP16.rknn')
import cv2, numpy as np
from app.infrastructure import camera_config as calib, detector
from app.infrastructure.ranging_service import RangingService
import preview_pipeline as pipeline
from bucket_decode import decode
from diagnostics import range_audit
from verify import verify

stages=collections.defaultdict(list)
def measured(name,fn,*args,**kwargs):
    t=time.monotonic();value=fn(*args,**kwargs);stages[name].append((time.monotonic()-t)*1000);return value

class RuntimeProfile:
    def __init__(self,inner):self.inner=inner
    def __getattr__(self,key):return getattr(self.inner,key)
    def set_inputs(self,*args,**kwargs):return measured('npu_inputs_set_ms',self.inner.set_inputs,*args,**kwargs)
    def run(self,*args,**kwargs):return measured('npu_run_ms',self.inner.run,*args,**kwargs)
    def get_outputs(self,*args,**kwargs):return measured('npu_get_outputs_ms',self.inner.get_outputs,*args,**kwargs)
class Detector(detector.RknnDetector):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rknn.rknn_runtime=RuntimeProfile(self.rknn.rknn_runtime)
        print('MODEL_ABSOLUTE',self.model_path,'CORE',self.core_mask_name,flush=True)
        libs=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'librknnrt' in line})
        print('LOADED_RKNN_LIBRARIES',libs,flush=True)
    def infer(self,im):
        inp=measured('preprocess_ms',detector.resize_image,im,detector.MODEL_SIZE,True)[None]
        raw=measured('npu_inference_get_ms',detector.infer,self.rknn,inp)
        answer,_=measured('postprocess_ms',decode,detector,raw)
        return answer
detector.RknnDetector=Detector

class Matcher:
    def __init__(self,inner):self.inner=inner
    def compute(self,*args):return measured('sgbm_ms',self.inner.compute,*args)

class Worker(pipeline.StereoWorker):
    def __init__(self,*args):
        super().__init__(*args)
        assert self.wls is None,'This benchmark expects the deployed effective WLS-disabled configuration'
        self.stereo=Matcher(self.stereo)
    def measure(self,box):
        stages.clear();start=time.monotonic();left,raw_right=self.images
        right=measured('right_remap_ms',cv2.remap,raw_right,calib.right_map1,calib.right_map2,cv2.INTER_LINEAR)
        lg=measured('grayscale_ms',cv2.cvtColor,left,cv2.COLOR_BGR2GRAY)
        rg=measured('grayscale_ms',cv2.cvtColor,right,cv2.COLOR_BGR2GRAY)
        disp=measured('disparity_total_ms',self.service._compute_disparity,self.stereo,lg,rg,None,None,scale=self.scale)
        z=measured('roi_ms',self.service._calculate_distance_for_box,disp,*box)
        audit=measured('audit_duplicate_roi_ms',range_audit,self.service,calib,disp,*box)
        assert z==audit['value_m']
        end=time.monotonic();audit['_profile']={k:sum(v) for k,v in stages.items()}
        audit['_profile'].update(worker_start=start,worker_end=end,worker_total_ms=(end-start)*1000)
        return z,audit,None
pipeline.StereoWorker=Worker

original_remap_rows=pipeline.remap_rows
def remap_rows(*args):return measured('left_remap_slice_ms',original_remap_rows,*args)
pipeline.remap_rows=remap_rows
original_empty=np.empty
def profiled_empty(*args,**kwargs):return measured('numpy_empty_allocation_ms',original_empty,*args,**kwargs)
np.empty=profiled_empty
original_stereo_images=pipeline.stereo_images
class CopyArray(np.ndarray):
    def __setitem__(self,key,value):return measured('shared_copy_ms',super().__setitem__,key,value)
def stereo_images(buffer):return original_stereo_images(buffer).view(CopyArray)
pipeline.stereo_images=stereo_images

OriginalPool=pipeline.ProcessPoolExecutor
class ProfileFuture:
    def __init__(self,inner,submitted):self.inner=inner;self.submitted=submitted
    def done(self):return self.inner.done()
    def result(self):
        result=self.inner.result();p=result[1]['_profile']
        p['submit_to_worker_ms']=(p['worker_start']-self.submitted)*1000
        return result
class ProfilePool(OriginalPool):
    def submit(self,*args,**kwargs):
        start=time.monotonic();inner=super().submit(*args,**kwargs)
        stages['executor_submit_ms'].append((time.monotonic()-start)*1000)
        return ProfileFuture(inner,start)
pipeline.ProcessPoolExecutor=ProfilePool

class Replay:
    def __init__(self,images,fps):self.images=images;self.fps=fps;self.seq=0;self.next=None
    def read(self):
        t=time.monotonic()
        if self.next is None:self.next=t
        if self.fps:
            time.sleep(max(0,self.next-time.monotonic()));self.next+=1/self.fps
        im=self.images[self.seq%len(self.images)];self.seq+=1
        stages['capture_wait_ms'].append((time.monotonic()-t)*1000)
        return True,im
    def set(self,*args):return True
    def release(self):pass

def quant(v):
    if not v:return None
    return dict(n=len(v),mean=float(np.mean(v)),p50=float(np.median(v)),p95=float(np.percentile(v,95)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['sync','async'],default='async');p.add_argument('--scale',type=float,choices=[.5,1.],default=.5)
    p.add_argument('--seconds',type=float,default=20);p.add_argument('--frames',type=int,default=68);p.add_argument('--warmup',type=int,default=17);p.add_argument('--fps',type=float,default=30)
    a=p.parse_args();verify();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'conditions.json').read_text());params=cfg['ranging']
    params.update(cache_display_frame=False,enable_display=False,render_on_device=False)
    files=Path(a.corpus).read_text().splitlines();images=[measured('png_decode_ms',cv2.imread,f) for f in files]
    assert all(im is not None and im.shape==(720,2560,3) for im in images)
    decode_metrics=quant(stages.pop('png_decode_ms'));service=RangingService(**params);rows=[]
    start=time.monotonic();warm_end=None;first=None
    if a.mode=='async':
        cap=Replay(images,a.fps);service._open_camera_capture=lambda:(0,cap,images[0]);service.camera_active=True
        def sink(event,details):
            nonlocal first,warm_end
            now=time.monotonic();raw=details.get('raw');source=details.get('source_camera_seq')
            row=dict(frame=event.frame_id,input_index=(details['camera_seq']-1)%len(images),camera_seq=details['camera_seq'],
                     detections=event.detections,raw=raw,published=event.distance_m,
                     status='new' if raw is not None else 'held' if event.distance_m is not None else 'invalid',
                     source=source,attempt=details.get('attempt_frame'),discarded=details.get('discarded_attempt',False),
                     arrival=details['capture_monotonic_s'],published_at=now,
                     latency_ms=(now-details['source_time'])*1000 if source is not None else None,
                     worker=details.get('attempt',{}).get('_profile'))
            if first is None:first=now
            # Five seconds includes spawn/NPU warmup; exclude all startup stages.
            if now-first<5:stages.clear();return
            if warm_end is None:warm_end=now;stages.clear()
            rows.append(row)
            if now-warm_end>=a.seconds:service.camera_active=False
        pipeline.run(service,sink,params,a.scale,display=False)
    else:
        # Calibration import may already have created OpenCV worker threads.
        # Bind the whole existing thread group, not just the calling thread.
        for tid in os.listdir('/proc/self/task'):
            try:os.sched_setaffinity(int(tid),{4,5,6,7})
            except ProcessLookupError:pass
        cv2.setNumThreads(3)
        model=Detector();buffer=bytearray(2*720*1280*3);worker=Worker(buffer,params,a.scale,False)
        value=previous=None
        for i in range(a.warmup+a.frames):
            stages.clear()
            im=images[i%len(images)];t=time.monotonic()
            left=measured('left_remap_ms',cv2.remap,im[:,:1280],calib.left_map1,calib.left_map2,cv2.INTER_LINEAR)
            detections=pipeline.detect(left,model);best=max(detections,key=lambda d:d['score'],default=None)
            box=tuple(best[k] for k in ['left','top','right','bottom']) if best else None
            if not pipeline.same_target(previous,box):value=None
            previous=box
            dp={k:v[-1] for k,v in stages.items() if v};stages.clear()
            raw=audit=None
            if best:
                worker.images[0]=left;worker.images[1]=im[:,1280:]
                raw,audit,_=worker.measure(box)
                value=raw if raw is None or value is None else .9*raw+.1*value
            else:value=None
            now=time.monotonic()
            row=dict(frame=i-a.warmup,input_index=i%len(images),detections=detections,raw=raw,published=value,status='new' if raw is not None else 'invalid',source=i-a.warmup if raw is not None else None,
                     arrival=t,published_at=now,latency_ms=(now-t)*1000,detector=dp,worker=audit['_profile'] if audit else None)
            if i>=a.warmup:rows.append(row)
        model.release()
    (out/'frames.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    new=[r for r in rows if r['status']=='new'];duration=rows[-1]['published_at']-rows[0]['arrival']
    report=dict(mode=a.mode,scale=a.scale,opencv=cv2.__version__,opencv_file=cv2.__file__,input=f'{len(images)} recorded SBS pairs repeated; NOT fresh camera',
                frames=len(rows),duration=duration,output_fps=len(rows)/duration,new=len(new),new_hz=len(new)/duration,
                fresh_latency_ms=quant([r['latency_ms'] for r in new]),detected=sum(bool(r['detections']) for r in rows),png_decode_ms=decode_metrics,
                stages={k:quant(v) for k,v in stages.items()})
    worker=[r['worker'] for r in rows if r['worker']]
    report['worker_stages']={k:quant([w[k] for w in worker]) for k in worker[0] if k.endswith('_ms')} if worker else {}
    report['result_pickup_ms']=quant([(r['published_at']-r['worker']['worker_end'])*1000 for r in rows if r['worker']])
    if a.mode=='sync':report['detector_stages']={k:quant([r['detector'][k] for r in rows]) for k in rows[0]['detector']}
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
