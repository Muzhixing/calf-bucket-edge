"""Local HTTP preview of the frozen V2 vision modules, without controller imports."""
import argparse, collections, json, os, platform, signal, statistics, sys, threading, time
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from support import ROOT, save, sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8788);ap.add_argument('--seconds',type=float,default=3600);ap.add_argument('--id',required=True);ap.add_argument('--mode', choices=['async','sync'], default='async')
    ap.add_argument('--cores', choices=['auto','0_1_2'], default='auto')
    ap.add_argument('--depth-scale', type=float, choices=[1.0,0.5], help='async default: 0.5 (fast); 1.0: full resolution')
    a=ap.parse_args()
    if a.depth_scale is None:
        a.depth_scale = 1.0 if a.mode == 'sync' else 0.5
    if a.mode == 'sync' and a.depth_scale != 1:
        ap.error('--depth-scale applies to async preview only')
    import re
    assert re.fullmatch('[A-Za-z0-9_-]+',a.id) and 0<a.seconds<=43200
    assert platform.machine()=='aarch64' and 'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_text()
    from verify import verify
    verify()
    for p,h in json.loads((ROOT/'config/viewer_tool_hashes.json').read_text()).items():assert sha(ROOT/p)==h
    page=(ROOT/'tools/preview.html').read_bytes()
    os.environ['BUCKET_CALIBRATION_PROFILE']='v2_radial2';sys.path.insert(0,str(ROOT))
    import cv2, numpy as np
    from app.infrastructure import detector, ranging_service as ranging, camera_config as calib
    from diagnostics import range_audit
    from bucket_decode import decode
    cv2.setNumThreads(4)
    cfg=json.loads((ROOT/'conditions.json').read_text());model=json.loads((ROOT/'models.json').read_text())[0];mp=(ROOT/model['path']).resolve()
    assert sha(mp)==model['sha256'];os.environ['RKNN_MODEL_PATH']=str(mp);os.environ['RKNN_CORE_MASK']=a.cores
    folder=ROOT/'logs'/a.id;folder.mkdir(parents=True,exist_ok=False)
    save(folder/'viewer_tools_build.json',json.loads((ROOT/'config/viewer_tool_hashes.json').read_text()))
    params=dict(cfg['ranging']);params.update(cache_display_frame=False,enable_display=False,render_on_device=False)
    params['camera_device']=str(Path(params['camera_device']).resolve(strict=True))
    service=ranging.RangingService(**params)
    from guidance import VisionGuidance
    guidance = VisionGuidance(a.id, cfg['controller']['center_tolerance'],
                              cfg['reference_contract']['vertical_tolerance_fraction_of_half_height'],
                              calib.P1[0, 0], calib.P1[1, 1])
    original_open=service._open_camera_capture
    def checked_open():
        idx,cap,im=original_open()
        assert cap is not None and im is not None and im.shape==(720,2560,3)
        assert idx==int(Path(params['camera_device']).name.removeprefix('video'))
        receipt['actual_camera_fps']=cap.get(cv2.CAP_PROP_FPS);receipt['actual_sbs_size']=[im.shape[1],im.shape[0]]
        save(folder/'environment.json',receipt)
        return idx,cap,im
    service._open_camera_capture=checked_open
    lock=threading.Condition();state={};image=[None];times=array('d');new_times=array('d');infer_ms=array('d');npu_ms=array('d');pending={};last_sample=[None]
    recent_depth = collections.deque()
    counts = collections.Counter()
    capture_ms, fresh_age_ms = array('d'), array('d')
    first_row = {}
    logfile=(folder/'frames.jsonl').open('x',buffering=1)
    base=detector.RknnDetector;raw_infer=detector.infer;measure=service._calculate_distance_for_box
    def timed_raw(rknn,inp):
        start=time.perf_counter();raw=raw_infer(rknn,inp);npu_ms.append((time.perf_counter()-start)*1000)
        assert len(raw)==9 and all(list(x.shape)==e['shape'] for x,e in zip(raw,model['expected_outputs']))
        return raw
    class ExplicitDetector(base):
        def __init__(self,*args,**kwargs):
            super().__init__(model_path=str(mp));assert Path(self.model_path).resolve()==mp
            receipt['effective_core_mask'] = self.core_mask_name
            save(folder/'environment.json',receipt)
            print('V2_MODEL_LOADED='+str(mp)+' SHA256='+sha(mp),flush=True)
        def infer(self,im):
            start = time.perf_counter()
            inputs = detector.resize_image(im, detector.MODEL_SIZE, True)[None]
            answer, _ = decode(detector, timed_raw(self.rknn, inputs))
            infer_ms.append((time.perf_counter() - start) * 1000)
            return answer
    def measured(*args):
        value=measure(*args);audit=range_audit(service,calib,*args);assert value==audit['value_m']
        pending.update(raw=value,attempt=audit);return value
    class Sink:
        def publish(self,event,details=None):
            if details is not None:pending.update(details)
            now=time.monotonic();times.append(now);md=service.latest_metadata
            fresh=pending.get('raw') is not None
            status='new' if fresh else 'held' if md['distance_held'] and event.distance_m is not None else 'invalid'
            if fresh:last_sample[0]=pending.get('source_frame',event.frame_id);new_times.append(now)
            if fresh:
                recent_depth.append(now)
            while recent_depth and recent_depth[0] <= now-3:
                recent_depth.popleft()
            recent=times[-30:];fps=(len(recent)-1)/(recent[-1]-recent[0]) if len(recent)>1 else None
            selected=max(event.detections,key=lambda x:x['score']) if event.detections else None
            row=dict(frame_id=event.frame_id,monotonic_s=now,detected=event.detected,selected=selected,distance_m=event.distance_m,
                raw_distance_m=pending.get('raw'),measurement_status=status,distance_source=md['distance_source'],distance_source_frame=last_sample[0] if event.distance_m is not None else None,
                depth_scale=a.depth_scale,fresh_distance_hz=len(recent_depth)/3 if now-times[0]>=3 else None,
                processing_fps=fps,npu_call_ms=npu_ms[-1],detect_with_prepost_ms=infer_ms[-1],profile_ms=md['profile_ms'])
            if details is not None:
                row.update(camera_seq=details['camera_seq'],capture_monotonic_s=details['capture_monotonic_s'],
                           capture_to_detection_ms=(now-details['capture_monotonic_s'])*1000,
                           distance_source_camera_seq=details.get('source_camera_seq'),
                           distance_age_ms=(now-details['source_time'])*1000 if 'source_time' in details else None,
                           attempt_frame=details.get('attempt_frame'),discarded_attempt=details.get('discarded_attempt',False),
                           attempt_reason=details.get('attempt_reason'))
                if status=='invalid' and not row['attempt_reason']:
                    row['attempt_reason']='waiting_for_depth' if event.detected else 'no_detected_bucket'
            height, width = event.left_frame.shape[:2]
            guide = guidance.update(row, width, height)
            row['guidance'] = guide
            if not first_row:
                first_row.update(row)
            counts[status] += 1
            counts['detected'] += int(event.detected)
            if details is not None:
                capture_ms.append(row['capture_to_detection_ms'])
                if fresh:
                    fresh_age_ms.append(row['distance_age_ms'])
            logfile.write(json.dumps(row,allow_nan=False)+'\n')
            im=event.left_frame.copy()
            center = (width // 2, height // 2)
            cv2.drawMarker(im, center, (32, 32, 32), cv2.MARKER_CROSS, 23, 4)
            cv2.drawMarker(im, center, (255, 255, 255), cv2.MARKER_CROSS, 23, 2)
            if event.detected and selected:
                corner = (selected['left'], selected['top'])
                opposite = (selected['right'], selected['bottom'])
                cv2.rectangle(im, corner, opposite, (24, 44, 34), 4)
                color = (138, 236, 90) if guide['aligned_xy'] else (76, 193, 252)
                cv2.rectangle(im, corner, opposite, color, 2)
                target = tuple(round(v) for v in guide['target_center_px'])
                cv2.line(im, center, target, color, 1, cv2.LINE_AA)
                cv2.circle(im, target, 4, color, -1, cv2.LINE_AA)
            if event.frame_id==1:assert cv2.imwrite(str(folder/'first_preview.jpg'),im)
            with lock:
                state.update(row)
                image[0] = im
                lock.notify_all()
            pending.clear()
            if now-times[0]>=a.seconds:service.camera_active=False
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def stream_frames(self):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            previous = 0
            while service.camera_active:
                with lock:
                    ready = lock.wait_for(lambda: state.get('frame_id', 0) > previous, 1)
                    if not ready:
                        continue
                    previous, frame = state['frame_id'], image[0]
                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    header = f'--frame\r\nContent-Type: image/jpeg\r\nX-Frame-Id: {previous}\r\nContent-Length: {jpeg.size}\r\n\r\n'
                    try:
                        self.wfile.write(header.encode() + jpeg.tobytes() + b'\r\n')
                    except (BrokenPipeError, ConnectionResetError):
                        return

        def do_GET(self):
            path=self.path.split('?')[0];ctype='text/html; charset=utf-8';code=200
            if path == '/stream.mjpg':
                return self.stream_frames()
            if path=='/':body=page
            elif path=='/status':
                with lock:data=dict(state)
                data['frame_age_s']=time.monotonic()-data['monotonic_s'] if data else None
                body=json.dumps(data).encode();ctype='application/json'
            elif path=='/frame.jpg':
                with lock:im=image[0]
                if im is None:body=b'Warming up';code=503;ctype='text/plain'
                else:
                    ok,jpeg=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,85]);assert ok;body=jpeg.tobytes();ctype='image/jpeg'
            else:body=b'Not found';code=404;ctype='text/plain'
            self.send_response(code);self.send_header('Content-Type',ctype);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
    # Bind before touching the camera, so an occupied port cannot strand a capture.
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    service.event_bus=Sink();service._calculate_distance_for_box=measured;detector.infer=timed_raw;detector.RknnDetector=ExplicitDetector
    signal.signal(signal.SIGINT,lambda *_:setattr(service,'camera_active',False));signal.signal(signal.SIGTERM,lambda *_:setattr(service,'camera_active',False))
    signal.signal(signal.SIGHUP,lambda *_:setattr(service,'camera_active',False))
    receipt=dict(model_absolute_path=str(mp),model_sha256=sha(mp),calibration_path=calib.CALIBRATION_FILE,calibration_sha256=calib.CALIBRATION_SHA256,
        camera_device=params['camera_device'],eye_size=[1280,720],camera_requested_fps=30,actuation=False,serial=False,mode=a.mode,core_mask=a.cores,preview_depth_scale=a.depth_scale,stereo_schedule='single_flight' if a.mode=='async' else 'sync_interval',ranging=params)
    if a.mode == 'async':
        receipt['preview_pipeline'] = dict(camera_buffers=2, camera_cpu=4, rectification_cpus=[5,2],
            detection_cpu=6, stereo_cpus=[4,6,7], stereo_nice=5, detection_cv_threads=1, stereo_cv_threads=3,
            stereo_num_disparities=round(params['sgbm_num_disparities']*a.depth_scale/16)*16,
            wls_enabled=params['use_wls'] and a.depth_scale == 1 and
                        hasattr(getattr(cv2,'ximgproc',None),'createRightMatcher'),
            association_iou=0.5, clear_on_detection_loss=True)
    save(folder/'environment.json',receipt);print(f'PREVIEW_HTTP http://127.0.0.1:{a.port}/ SERIAL=ABSENT',flush=True)
    service.camera_active=True;success=False
    try:
        if a.mode == 'async':
            from preview_pipeline import run
            run(service, service.event_bus.publish, params, a.depth_scale)
        else:
            service._camera_loop()
        assert times and len(times)==len(infer_ms)==len(npu_ms)
        assert all(x not in sys.modules for x in ['serial','app.infrastructure.serial_port','app.infrastructure.bucket_tracking','app.infrastructure.task_client','main'])
        success=True
    finally:
        service.camera_active=False
        if service.cap is not None:service.cap.release()
        server.shutdown();server.server_close();logfile.close();detector.infer=raw_infer;detector.RknnDetector=base
        duration=times[-1]-times[0] if len(times)>1 else None
        report=dict(passed=success,frames=len(times),detected=counts['detected'],measurement_states={s:counts[s] for s in ['new','held','invalid']},
            processing_fps=(len(times)-1)/duration if duration else None,npu_call_mean_ms=statistics.mean(npu_ms) if npu_ms else None,
            detector_with_prepost_mean_ms=statistics.mean(infer_ms) if infer_ms else None,new_distance_samples=len(new_times),new_distance_interval_hz=(len(new_times)-1)/(new_times[-1]-new_times[0]) if len(new_times)>1 else None,
            no_actuation=True,log_mode='preview only; not raw capture acceptance data')
        if times and a.mode == 'async':
            arrivals = state['capture_monotonic_s'] - first_row['capture_monotonic_s']
            frames = state['camera_seq'] - first_row['camera_seq']
            report.update(camera_read_fps=frames/arrivals if arrivals else None,
                          skipped_pairs=frames+1-len(times))
            for field, samples in [('capture_to_detection_ms',capture_ms), ('distance_age_ms',fresh_age_ms)]:
                if samples:
                    report[field+'_p50_p95'] = np.percentile(samples,[50,95]).tolist()
        save(folder/'completion.json',report);print('PREVIEW_STOPPED',json.dumps(report),flush=True)
if __name__=='__main__':main()
