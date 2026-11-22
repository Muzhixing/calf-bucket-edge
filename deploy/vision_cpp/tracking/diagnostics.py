"""Bounded, advisory startup observations. No sleeps, commands or readiness gates."""
from collections import Counter, defaultdict, deque
import math
from pathlib import Path
import json
import shutil
import threading
import time


def distribution(values, threshold):
    values=sorted(values);n=len(values)
    return {'samples':n,'p95_ms':values[max(0,math.ceil(n*.95)-1)] if n else None,
            'max_ms':values[-1] if n else None,'long_frames':sum(v>threshold for v in values),'long_threshold_ms':threshold}


class StartupDiagnostics:
    thresholds={'capture_fetch':100,'inference':50,'tracking':50,'preview':50,'native_post_read':100,
                'host_to_tracking':100,'host_to_preview':100,'camera_interval':100,'serial_batch_interval':375}
    def __init__(self,seconds,now=None):
        self.started=time.monotonic() if now is None else now;self.deadline=self.started+seconds
        self.lock=threading.RLock();self.result=None;self.values=defaultdict(lambda:deque(maxlen=4096));self.errors=Counter()
        self.frames=0;self.candidates=0;self.last_frame=None;self.last_frame_host_at=None;self.last_image_sample=-math.inf
        self.serial_frames=0;self.serial_batches=0;self.last_serial_at=None;self.last_serial=None;self.uart_initial=None
        self.uart_new=0;self.uart_growth_batches=0;self.uart_ambiguous=0;self.counter_wraps=0;self.reset_signals=0
        self.parser_first=None;self.parser_last={};self.coalesced=0;self.baseline={'status':'样本不足','reason':'baseline_pending'}

    def open(self,now):return self.result is None and self.started<=now<self.deadline
    def metric(self,name,ms,now=None):
        now=time.monotonic() if now is None else now
        with self.lock:
            if self.open(now) and isinstance(ms,(int,float)) and math.isfinite(ms) and ms>=0:self.values[name].append(float(ms))
    def error(self,source,now=None):
        now=time.monotonic() if now is None else now
        with self.lock:
            if self.open(now):self.errors[source]+=1
    def frame(self,meta,image=None,now=None):
        now=time.monotonic() if now is None else now
        with self.lock:
            if not self.open(now):return
            seq,stamp=meta.get('frame_id'),meta.get('arrival')
            if not isinstance(seq,int) or not isinstance(stamp,(int,float)) or not math.isfinite(stamp):
                self.errors['invalid_frame_timestamp']+=1;return
            if self.last_frame:
                old_seq,old_stamp=self.last_frame
                if seq==old_seq:self.errors['duplicate_frame']+=1;return
                if seq<old_seq or stamp<=old_stamp:self.errors['frame_clock_or_sequence_reset']+=1
                else:
                    self.metric('camera_interval',(stamp-old_stamp)*1000,now)
                    # Latest-only pipeline skips are not proof that the camera lost frames.
                    self.errors['consumer_skipped_ids']+=max(0,seq-old_seq-1)
            self.last_frame=(seq,stamp);self.last_frame_host_at=now;self.frames+=1;self.candidates+=bool(meta.get('detections') or meta.get('tracking_candidates'))
            self.metric('inference',meta.get('detector',{}).get('npu_run_ms'),now)
            published=meta.get('published_at')
            if isinstance(published,(int,float)) and math.isfinite(published) and published>=stamp:
                self.metric('native_post_read',(published-stamp)*1000,now)  # Two native clock stamps only.
            if image is not None and now-self.last_image_sample>=.25:
                self.last_image_sample=now
                try:
                    import numpy as np
                    small=image[::12,::12].astype(np.float32).mean(axis=2)
                    self.errors['image_quality_samples']+=1
                    if float((small<8).mean())>.95:self.errors['mostly_dark_hint']+=1
                    if float((small>247).mean())>.95:self.errors['mostly_bright_hint']+=1
                    gradient=float(np.abs(np.diff(small,axis=0)).mean()+np.abs(np.diff(small,axis=1)).mean())
                    if gradient<2:self.errors['low_detail_or_blur_hint']+=1
                except (AttributeError,TypeError,ValueError):pass

    def serial(self,frames,stats,now=None):
        now=time.monotonic() if now is None else now
        with self.lock:
            if not self.open(now):return
            # Parser is new for each startup owner: include its first read errors.
            if self.parser_first is None:self.parser_first={k:0 for k in stats}
            self.parser_last=dict(stats)
            if not frames:return
            if self.last_serial_at is not None:self.metric('serial_batch_interval',(now-self.last_serial_at)*1000,now)
            self.last_serial_at=now;self.serial_batches+=1;self.serial_frames+=len(frames);self.coalesced+=max(0,len(frames)-1)
            grew=False
            for s in frames:
                count=s.get('uart_error_count_mod256')
                if not isinstance(count,int):continue
                if self.uart_initial is None:self.uart_initial=count
                prev=self.last_serial
                if prev is not None:
                    reset=any(prev.get(k) and not s.get(k) for k in ('lift_zero_confirmed','slide_zero_confirmed'))
                    if reset:self.reset_signals+=1
                    old=prev['uart_error_count_mod256'];delta=(count-old)%256
                    if delta:
                        if reset or delta>127:self.uart_ambiguous+=1
                        elif count<old:
                            # 255->0 may be wrap OR restart: count only as uncertain.
                            self.counter_wraps+=1;self.uart_ambiguous+=1
                        else:self.uart_new+=delta;grew=True
                self.last_serial=s
            self.uart_growth_batches+=grew

    def probe_async(self,folder,limits,expected):
        def probe():
            result={'status':'通过','unavailable':[]}
            try:
                result['disk_free_bytes']=shutil.disk_usage(folder).free
                mem=Path('/proc/meminfo')
                if mem.exists():result['mem_available_kib']=next((int(x.split()[1]) for x in mem.read_text().splitlines() if x.startswith('MemAvailable:')),None)
                else:result['unavailable'].append('memory')
                result['temperatures_c']={p.parent.name:int(p.read_text())/1000 for p in list(Path('/sys/class/thermal').glob('thermal_zone*/temp'))[:16]}
                result['cpu_frequency_khz']={p.parent.name:int(p.read_text()) for p in list(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq'))[:16]}
                if not result['temperatures_c']:result['unavailable'].append('temperature')
                if not result['cpu_frequency_khz']:result['unavailable'].append('cpu_frequency')
                result['throttling_status']='unavailable; current frequency alone does not prove throttling'
                binding=json.loads((Path(folder)/'binding.json').read_text())
                result['binding']=binding
                comparisons={k:binding.get(k)==v for k,v in expected.items() if v is not None}
                result['binding_matches']=comparisons
                result['model_classes']='native single-class tensor validation; semantic label not encoded in binding'
                limits.check([75,0,limits.c]);result['motion_config_valid']=True
                result['status']='异常' if any(not v for v in comparisons.values()) else '通过' if comparisons else '样本不足'
                if not comparisons:result['unavailable'].append('expected_binding')
            except Exception as exc:result.update(status='样本不足',reason=str(exc))
            with self.lock:
                if self.open(time.monotonic()):self.baseline=result
        threading.Thread(target=probe,name='startup-baseline',daemon=True).start()

    def finish(self,now=None,force=False):
        now=time.monotonic() if now is None else now
        with self.lock:
            if self.result is not None:return self.result
            if now<self.deadline and not force:return None
            metrics={k:distribution(self.values[k],v) for k,v in self.thresholds.items()}
            parser={k:max(0,v-(self.parser_first or {}).get(k,0)) for k,v in self.parser_last.items()}
            issues=[k for k,v in self.errors.items() if v and k not in ('consumer_skipped_ids','image_quality_samples','low_detail_or_blur_hint')]
            issues += [k+':long_frames' for k,v in metrics.items() if v['long_frames']]
            issues += [k for k,v in parser.items() if v and k not in ('valid_frames',)]
            if self.uart_new:issues.append('uart_new_errors')
            if self.uart_ambiguous:issues.append('uart_counter_wrap_or_reset_ambiguous')
            if self.reset_signals:issues.append('origin_state_lost_or_mcu_reset')
            end=min(now,self.deadline)
            camera_tail=(end-self.last_frame_host_at)*1000 if self.last_frame_host_at is not None else None
            serial_tail=(end-self.last_serial_at)*1000 if self.last_serial_at is not None else None
            if camera_tail is not None and camera_tail>400:issues.append('camera:no_recent_frame_at_window_end')
            if serial_tail is not None and serial_tail>700:issues.append('serial:no_recent_frame_at_window_end')
            if self.baseline.get('status')=='异常':issues.append('configuration_mismatch')
            insufficient=[k for k in ('capture_fetch','inference','tracking','preview','camera_interval','serial_batch_interval') if metrics[k]['samples']<5]
            state='异常' if issues else '样本不足' if insufficient else '通过'
            self.result={'status':state,'window_seconds':self.deadline-self.started,'partial':now<self.deadline,
                         'metrics':metrics,'issues':issues,'insufficient':insufficient,'camera':{'frames':self.frames,'observations':dict(self.errors),
                         'candidate_frames':self.candidates,'tail_gap_ms':camera_tail,'native_decode_errors':'unavailable: native backend exposes no decoder counter',
                         'continuity_scope':'delivered native frame timestamps; consumer skips are not camera drops; texture is not continuity'},
                         'serial':{'tail_gap_ms':serial_tail,'frames':self.serial_frames,'batches':self.serial_batches,'coalesced_frames':self.coalesced,'parser':parser,
                         'uart_initial':self.uart_initial,'uart_new_unambiguous':self.uart_new,'uart_growth_batches':self.uart_growth_batches,'uart_repeated_growth':self.uart_growth_batches>=2,
                         'uart_wrap_or_reset_ambiguous':self.uart_ambiguous,'wrap_candidates':self.counter_wraps,'reset_signals':self.reset_signals},
                         'baseline':self.baseline,'clock_basis':{'capture_fetch':'host HTTP fetch duration, includes wait/copy; not sensor exposure',
                         'native_post_read':'native read-completion to publication in native steady clock',
                         'host_to_tracking':'host BGR receipt to tracking completion, same frame/host monotonic clock',
                         'host_to_preview':'host BGR receipt to JPEG ready, excludes browser/network display',
                         'serial_batch_interval':'host read batch arrival, not per-frame MCU timing'},
                         'unavailable':['sensor exposure/decode stage duration','native decoder error counter','browser glass-to-glass latency','reliable throttling flag']}
            return self.result
