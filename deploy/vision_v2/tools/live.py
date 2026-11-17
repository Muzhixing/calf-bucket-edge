import argparse,importlib.metadata,json,os,platform,re,signal,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from support import ROOT,sha,save,traced_detections
from bucket_decode import decode
from diagnostics import range_audit,overlay


def main():
    ap=argparse.ArgumentParser(description='V2 camera -> original detector/ranging -> files only')
    ap.add_argument('--id',required=True);ap.add_argument('--seconds',type=float,default=15);ap.add_argument('--max-frames',type=int,default=0);ap.add_argument('--sequence',type=Path);ap.add_argument('--output-tag');ap.add_argument('--calibration-profile',choices=['v1_original','v2_radial2'],default='v2_radial2')
    a=ap.parse_args();assert a.output_tag is None or re.fullmatch('[A-Za-z0-9_-]+',a.output_tag)
    assert re.fullmatch('[A-Za-z0-9_-]+',a.id);assert a.seconds>0 and a.max_frames>=0
    assert platform.machine()=='aarch64' and 'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_text()
    if a.sequence is None:assert a.calibration_profile=='v2_radial2','old calibration is replay-only in this V2 entry'
    os.environ['BUCKET_CALIBRATION_PROFILE']=a.calibration_profile
    from verify import verify
    build=verify()
    sys.path.insert(0,str(ROOT))
    import cv2,numpy as np
    from app.infrastructure import detector,ranging_service as ranging,camera_config as calib
    assert Path(detector.__file__).resolve()==ROOT/'app/infrastructure/detector.py'
    cv2.setNumThreads(4)
    cfg=json.loads((ROOT/'conditions.json').read_text());model=json.loads((ROOT/'models.json').read_text())[0];mp=(ROOT/model['path']).resolve()
    assert sha(mp)==model['sha256']
    os.environ['RKNN_MODEL_PATH']=str(mp);os.environ['RKNN_CORE_MASK']=cfg['core_mask']
    print(f'V2_MODEL_ABSOLUTE_PATH={mp}\nV2_MODEL_SHA256={sha(mp)}',flush=True)
    print('V2 app='+str(ROOT)+' device='+cfg['ranging']['camera_device']+' input=BGR score=0.536 NMS=0.45 SERIAL=ABSENT',flush=True)
    assert detector.OBJ_THRESH==detector.DETECT_SCORE_MIN==cfg['score_threshold'] and detector.NMS_THRESH==cfg['nms_threshold']
    seq=ROOT/'captures'/a.id;label=f'{a.id}__{a.calibration_profile}' if a.output_tag is None else f'{a.id}__{a.output_tag}__{a.calibration_profile}';out=ROOT/'results'/label;folder=out/model['name']
    assert not out.exists(),'Use new --id, never overwrite evidence'
    if a.sequence is None:assert not seq.exists(),'capture ID already exists'
    else:
        seq=a.sequence.resolve();assert seq.is_dir();source_meta=json.loads((seq/'sequence.json').read_text());source_rows=[json.loads(x) for x in (seq/'frames.jsonl').read_text().splitlines()]
        assert source_meta['capture_status']=='RECORDED_NEEDS_QA' and len(source_rows)==source_meta['frames'] and source_rows
        assert sha(seq/'frames.jsonl')==source_meta['frames_log_sha256']
        assert all(b['monotonic_ns']>x['monotonic_ns'] for x,b in zip(source_rows,source_rows[1:]))
        for row in source_rows:assert sha(seq/row['path'])==row['sha256']
    if a.sequence is None:seq.mkdir();(seq/'frames').mkdir()
    folder.mkdir(parents=True)
    for d in ['raw','disparity','rectified_left','rectified_right','overlays']:(folder/d).mkdir()
    runtime_cfg=dict(cfg['ranging'])
    device=Path(runtime_cfg['camera_device']).resolve(strict=True) if a.sequence is None else Path('/dev/video21')
    assert re.fullmatch(r'/dev/video[0-9]+',str(device)), 'unexpected stable camera link target'
    runtime_cfg['camera_device']=str(device)
    service=ranging.RangingService(**runtime_cfg)
    assert not any([service.enable_push,service.enable_webrtc,service.async_capture,service.async_distance])
    stop_reason=['time_limit'];signal.signal(signal.SIGINT,lambda *_:(setattr(service,'camera_active',False),stop_reason.__setitem__(0,'operator_SIGINT')))
    signal.signal(signal.SIGTERM,lambda *_:(setattr(service,'camera_active',False),stop_reason.__setitem__(0,'operator_SIGTERM')))
    current={};capture_rows=[];result_rows=[];last_sample=[None];model_loaded=[False];inferences=[0];start=time.monotonic();base=detector.RknnDetector
    capture_log=(seq/'frames.jsonl').open('x',buffering=1) if a.sequence is None else None;result_log=(folder/'frames.jsonl').open('x',buffering=1)
    receipt=dict(host=platform.node(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
       model_absolute_path=str(mp),model_sha256=sha(mp),conditions=cfg,conditions_sha256=sha(ROOT/'conditions.json'),build_sha256=sha(ROOT/'config/runtime_hashes.json'),
       runtime_sha256=sha('/usr/lib/librknnrt.so'),opencv=cv2.__version__,numpy=np.__version__,
       rknn_lite=importlib.metadata.version('rknn-toolkit-lite2'),camera_device=str(Path(service.camera_device).resolve()),
       calibration_loader_sha256=sha(ROOT/'app/infrastructure/camera_config.py'),calibration_file=calib.CALIBRATION_FILE,calibration_sha256=calib.CALIBRATION_SHA256,calibration_profile=calib.CALIBRATION_PROFILE,
       source_sequence=str(seq),source_sequence_sha256=sha(seq/'frames.jsonl') if a.sequence else None,P1=calib.P1.tolist(),P2=calib.P2.tolist(),Q=calib.Q.tolist(),
       R1=calib.R1.tolist(),T_mm=calib.T.tolist(),baseline_m=float(np.linalg.norm(calib.T))/1000,
       wls_right_matcher_api=hasattr(ranging.ximgproc,'createRightMatcher'),wls_filter_api=hasattr(ranging.ximgproc,'createDisparityWLSFilter'),
       timing_mode='live_real_clock' if a.sequence is None else 'replay_recorded_arrival_clock',output_tag=a.output_tag,actuation=False,entry='vision_v2.py',calibration_physical_identity='physical left/right confirmed; user-authorized V2 radial2 after accepted single-point rough screen check',
       camera_loop='original app/infrastructure/ranging_service.py; real clock, synchronous logging overhead included')
    save(out/'environment.json',receipt)
    save(out/'build.json',build)
    from calibration_io import raw_to_rectified
    receipt['coordinate_axes']=dict(image_midpoint=[640,360],rectified_principal_point=[float(calib.P1[0,2]),float(calib.P1[1,2])],raw_sensor_midpoint_rectified=raw_to_rectified([640,360],a.calibration_profile)[0].tolist(),mechanical_axis='not independently marked; 640 is only image midpoint')
    save(out/'environment.json',receipt)
    meta=dict(target_bucket='UNDESIGNATED',technical_readiness_only=True,independent_capture=False,actuator_output='absent',
      capture_status='RECORDING',device=service.camera_device,timestamp_kind='host monotonic read arrival; NOT exposure timestamp',
      start_wall_ns=time.time_ns(),stereo_sync='one SBS buffer; exposure sync and physical order need confirmation')
    if a.sequence is None:save(seq/'sequence.json',meta)
    # Bound memory to the current frame. Publish its receipt only after all writes succeed.
    writers = ThreadPoolExecutor(max_workers=3)
    def write_sbs(path, image):
        if not cv2.imwrite(str(path), image):
            raise IOError(f'Failed to save {path}')
        return sha(path), time.monotonic_ns()

    class Capture:
        def __init__(self,cap):self.cap=cap
        def read(self):
            before=time.monotonic_ns();ok,im=self.cap.read();after=time.monotonic_ns()
            assert ok and im is not None and im.shape==(720,2560,3),'camera read/size failure'
            fid=len(capture_rows)+1;p=seq/'frames'/f'{fid-1:06d}.png'
            current['sbs_write'] = writers.submit(write_sbs, p, im)
            row=dict(frame_id=fid,path=str(p.relative_to(seq)),read_start_ns=before,monotonic_ns=after,
              wall_ns=time.time_ns(),left_timestamp_ns=after,right_timestamp_ns=after,same_sbs_buffer=True)
            current.update(input=row,sbs=im,trace=[],raw_distance=None,range_attempt=None)
            return True,im
        def get(self,prop):return self.cap.get(prop)
        def release(self):self.cap.release()
    original_open=service._open_camera_capture
    def open_camera():
        idx,cap,im=original_open();assert idx==int(device.name.removeprefix('video')), 'camera device differs from pinned by-id target'
        receipt['opened_camera_index']=idx;receipt['requested_camera_by_id']=cfg['ranging']['camera_device'];save(out/'environment.json',receipt)
        assert cap is not None and im is not None and im.shape==(720,2560,3),'camera initialization/size failure'
        assert cv2.imwrite(str(seq/'preview_sbs.jpg'),im)
        for side,s in [('left',0),('right',1280)]:assert cv2.imwrite(str(seq/f'preview_{side}.jpg'),im[:,s:s+1280])
        return idx,Capture(cap),im
    service._open_camera_capture=open_camera
    class TracedDetector(base):
        def __init__(self,*args,**kwargs):
            super().__init__(model_path=str(mp));assert Path(self.model_path).resolve()==mp
            model_loaded[0]=True
            print(f'V2_MODEL_LOADED={self.model_path} SHA256={sha(self.model_path)}',flush=True)
        def infer(self,im):
            inferences[0]+=1
            raw=detector.infer(self.rknn,detector.resize_image(im,detector.MODEL_SIZE,True)[None])
            assert len(raw)==9 and all(list(v.shape)==e['shape'] for v,e in zip(raw,model['expected_outputs']))
            current['tensor_write'] = writers.submit(np.savez_compressed, folder/'raw'/f"{current['input']['frame_id']:06d}.npz",**{f'output_{i}':v for i,v in enumerate(raw)})
            decoded, anchors = decode(detector, raw)
            current['trace'] = traced_detections(detector, decoded, anchors, im.shape)
            return decoded
    orig_measure=service._calculate_distance_for_box
    def measure(*args):
        value=orig_measure(*args);audit=range_audit(service,calib,*args)
        assert value==audit['value_m'],'diagnostic differs from original'
        current['raw_distance']=value;current['range_attempt']=audit
        np.save(folder/'disparity'/f"{current['input']['frame_id']:06d}.npy",args[0])
        if value is not None:last_sample[0]=current['input']['frame_id']
        return value
    service._calculate_distance_for_box=measure
    class Sink:
        def publish(self,event):
            fid=current['input']['frame_id'];assert event.frame_id==fid
            if capture_log is not None:
                source = current['input']
                source['sha256'], source['save_done_ns'] = current['sbs_write'].result()
                capture_rows.append(source)
                capture_log.write(json.dumps(source) + '\n')
            trace=current['trace'];s=max(trace,key=lambda d:d['score']) if trace else None;md=service.latest_metadata;z=event.distance_m
            status='new' if current['raw_distance'] is not None else 'held' if md['distance_held'] else 'invalid'
            attempted=current['range_attempt'] is not None
            row=dict(timing_mode='live_real_clock' if a.sequence is None else 'replay_recorded_arrival_clock',output_tag=a.output_tag,calibration_profile=calib.CALIBRATION_PROFILE,calibration_sha256=calib.CALIBRATION_SHA256,localization_method='original_bucket_box_midpoint',range_method='original_expanded_bbox_roi',sequence=a.id,frame_id=fid,model=model['name'],selected=s,detections=trace,input_frame_sha256=current['input']['sha256'],
              monotonic_ns=current['input']['monotonic_ns'],elapsed_s=time.monotonic()-start,
              raw_distance_m=current['raw_distance'],distance_m=z,distance_valid=z is not None,
              distance_held=md['distance_held'],distance_source=md['distance_source'],distance_sample_frame=last_sample[0] if z is not None else None,
              measurement_status=status,range_attempt=current['range_attempt'],
              measurement_reason='successful_ROI_sample' if status=='new' else ('sample_failed_using_previous' if attempted else 'interval_skip_using_previous') if status=='held' else 'ROI_sample_failed' if attempted else 'no_detection' if s is None else 'interval_skipped_previous_expired' if last_sample[0] is not None else 'no_previous_valid_distance',
              center_x_px=s['center_x'] if s else None,center_y_px=(s['top']+s['bottom'])/2 if s else None,
              centered=abs(s['center_x']-640)<=51.2 if s else None,close_enough=z<=.5 if z is not None else None,
              rectified_left=f'rectified_left/{fid:06d}.png',real_actuation=False)
            left_write = writers.submit(cv2.imwrite, str(folder/row['rectified_left']), event.left_frame)
            right=cv2.remap(current['sbs'][:,1280:],calib.right_map1,calib.right_map2,cv2.INTER_LINEAR)
            right_write = writers.submit(cv2.imwrite, str(folder/'rectified_right'/f'{fid:06d}.png'), right)
            assert cv2.imwrite(str(folder/'overlays'/f'{fid:06d}.jpg'),overlay(event.left_frame,row))
            current['tensor_write'].result()
            if not left_write.result() or not right_write.result():
                raise IOError('Could not save rectified pair')
            result_rows.append(row);result_log.write(json.dumps(row,allow_nan=False)+'\n')
            print(f"profile={a.calibration_profile} frame={fid} detected={s is not None} box_cx={row['center_x_px']} Z={z} status={status} source={row['distance_source']} reason={row['measurement_reason']}",flush=True)
            if a.sequence is not None and fid>=len(source_rows):
                stop_reason[0]='end_of_recorded_sequence';service.camera_active=False
            elif a.max_frames and fid>=a.max_frames:
                stop_reason[0]='frame_limit';service.camera_active=False
            elif a.sequence is None and time.monotonic()-start>=a.seconds:
                stop_reason[0]='time_limit';service.camera_active=False
    if a.sequence is not None:
        class ReplaySource:
            def __init__(self):self.index=0
            def read(self):
                row=source_rows[self.index];self.index+=1;im=cv2.imread(str(seq/row['path']));assert im.shape==(720,2560,3)
                capture_rows.append(row);current.update(input=row,sbs=im,trace=[],raw_distance=None,range_attempt=None)
                return True,im
            def get(self,prop):return float(cv2.VideoWriter_fourcc(*'MJPG')) if prop==cv2.CAP_PROP_FOURCC else 0.0
            def release(self):pass
        real_perf=time.perf_counter
        class RecordedClock:
            def time(self):return (current.get('input') or source_rows[0])['monotonic_ns']/1e9
            monotonic=time
            perf_counter=staticmethod(real_perf)
            sleep=staticmethod(lambda n:None)
        replay=ReplaySource();service._open_camera_capture=lambda:(int(device.name.removeprefix('video')),replay,cv2.imread(str(seq/source_rows[0]['path'])))
        ranging.time=RecordedClock()
    service.event_bus=Sink();detector.RknnDetector=TracedDetector;service.camera_active=True;success=False
    try:
        service._camera_loop();assert result_rows and len(result_rows)==len(capture_rows)
        assert model_loaded[0] and inferences[0]==len(result_rows),'model not loaded or inference count differs'
        forbidden=['serial','app.infrastructure.serial_port','app.infrastructure.task_client','app.infrastructure.bucket_tracking','main']
        assert all(m not in sys.modules for m in forbidden)
        receipt['loaded_app_modules']={k:getattr(v,'__file__',None) for k,v in sys.modules.items() if k.startswith('app.')}
        receipt['forbidden_modules_absent']=forbidden;save(out/'environment.json',receipt);success=True
    finally:
        service.camera_active=False
        if service.cap is not None:service.cap.release()
        writers.shutdown(wait=True)
        detector.RknnDetector=base
        if capture_log is not None:capture_log.close()
        result_log.close()
        meta.update(capture_status='RECORDED_NEEDS_QA' if success else 'ABORTED',frames=len(capture_rows),frames_log_sha256=sha(seq/'frames.jsonl'),end_wall_ns=time.time_ns())
        if a.sequence is None:save(seq/'sequence.json',meta)
        save(out/'completion.json',dict(passed=success,frames=len(result_rows),real_camera=a.sequence is None,calibration_profile=a.calibration_profile,model_loaded=model_loaded[0],inferences=inferences[0],no_actuation=True,stop_reason=stop_reason[0]))
    from trial import qa,evaluate
    if a.sequence is None:qa(seq)
    evaluate(folder)
    print('V2_STOPPED camera released; outputs='+str(out),flush=True)
if __name__=='__main__':main()
