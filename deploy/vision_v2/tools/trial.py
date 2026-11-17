"""Capture integrity and evaluation of same-image independent references."""
import csv,json
from pathlib import Path
from support import ROOT,save,sha

def qa(sequence):
 import cv2,numpy as np
 meta=json.loads((sequence/'sequence.json').read_text());rows=[json.loads(x) for x in (sequence/'frames.jsonl').read_text().splitlines()]
 assert meta['capture_status']=='RECORDED_NEEDS_QA' and len(rows)==meta['frames'] and rows
 assert sha(sequence/'frames.jsonl')==meta['frames_log_sha256'] and all(b['monotonic_ns']>a['monotonic_ns'] for a,b in zip(rows,rows[1:]))
 halves=[]
 for side in ['left','right']:(sequence/side).mkdir(exist_ok=True)
 for row in rows:
  p=(sequence/row['path']).resolve();assert p.is_relative_to(sequence.resolve()) and sha(p)==row['sha256']
  im=cv2.imread(str(p));assert im is not None and im.shape==(720,2560,3) and row['same_sbs_buffer']
  item={'frame_id':row['frame_id']}
  for side,start in [('left',0),('right',1280)]:
   out=sequence/side/f"{row['frame_id']:06d}.png";assert cv2.imwrite(str(out),im[:,start:start+1280]);assert np.array_equal(cv2.imread(str(out)),im[:,start:start+1280]);item[side]={'path':str(out.relative_to(sequence)),'sha256':sha(out)}
  halves.append(item)
 save(sequence/'halves.json',halves);save(sequence/'capture_qa.json',dict(passed=True,frames=len(rows),all_SBS_hashes=True,strict_monotonic=True,left_right_lossless_match=True,physical_order='confirmed by 20261116 cover/uncover evidence',optical_exposure_sync='not independently timestamped'))
 print('CAPTURE_QA_PASS',len(rows),'left_right_lossless_match=true',flush=True)

def surface_diagnostic(folder,row,ref):
 import cv2,numpy as np
 result={'status':'not_requested','automatic_sampler_changed':False}
 body=ref.get('body_polygon_rectified');fg=ref.get('bucket_polygon_rectified');path=folder/'disparity'/f"{row['frame_id']:06d}.npy"
 if not path.exists():return dict(status='not_sampled_this_frame',automatic_sampler_changed=False)
 disp=np.load(path);result.update(disparity_source_frame=row['frame_id'],disparity_sha256=sha(path))
 if fg:
  overlaps=[]
  mask=np.zeros(disp.shape,np.uint8);cv2.fillPoly(mask,[np.round(fg).astype(np.int32)],1)
  for a in (row.get('range_attempt') or {}).get('attempts',[]):
   x1,y1,x2,y2=a['sample_xyxy'];region=mask[y1:y2,x1:x2];a['outside_independent_bucket_silhouette_fraction']=float((region==0).mean()) if region.size else None
   overlaps.append(dict(branch=a['branch'],sample_xyxy=a['sample_xyxy'],outside_fraction=a['outside_independent_bucket_silhouette_fraction']))
  result['expanded_roi_overlap']=overlaps
  result['status']='roi_overlap_diagnostic_only'
 if not body:return result
 mask=np.zeros(disp.shape,np.uint8);cv2.fillPoly(mask,[np.round(body).astype(np.int32)],1);mask=cv2.erode(mask,np.ones((3,3),np.uint8),iterations=1)
 valid=disp[(mask>0)&(disp>0)&np.isfinite(disp)];n=int(mask.sum());result.update(status='invalid',physical_surface='independently selected visible body patch; NOT mouth plane',requested_pixels=n,valid_pixels=len(valid),manual_mask=True,automatic_sampler_changed=False)
 if n==0 or len(valid)<max(30,n*.18):result['reason']='insufficient_body_patch_disparity';return result
 lo,hi=np.percentile(valid,[10,90]);v=valid[(valid>=lo)&(valid<=hi)];median=float(np.median(v));mad=float(np.median(abs(v-median)))+1e-6;v=v[abs(v-median)<=3*1.4826*mad]
 if len(v)<max(20,n*.1):result['reason']='insufficient_body_patch_inliers';return result
 from calibration_io import load
 m=load(row['calibration_profile'])['matrices'];z=float(m['P1'][0,0])*float(np.linalg.norm(m['T']))/1000/float(np.median(v))
 result.update(status='new_diagnostic_only',forward_Z_m=z,inlier_pixels=len(v),median_disparity_px=float(np.median(v)),sample_polygon_xy=body,reason='manual_body_mask_not_an_autonomous_algorithm')
 return result

def evaluate(folder):
 import cv2,numpy as np
 from diagnostics import overlay
 from calibration_io import reference_for_profile
 rows=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()];assert rows
 seq=rows[0]['sequence'];profile=rows[0]['calibration_profile'];trialpath=ROOT/'captures'/seq/'trial.json';trial=json.loads(trialpath.read_text()) if trialpath.exists() else {}
 refpath=ROOT/'captures'/seq/'independent_references.json';refs=json.loads(refpath.read_text()) if refpath.exists() else {};evaluated=[]
 for row in rows:
  source=refs.get(str(row['frame_id']));ref=reference_for_profile(source,profile) if source else None
  s=row['selected'];x=row['center_x_px'];z=row['distance_m']
  item=dict(placement_id=seq,calibration_profile=profile,calibration_sha256=row['calibration_sha256'],frame_id=row['frame_id'],detected=s is not None,box_center_x_px=x,box_center_y_px=row['center_y_px'],reference_mouth_x_px=ref['mouth_x_px'] if ref else None,reference_mouth_y_px=ref['mouth_y_px'] if ref else None,reference_raw_left_x_px=source['raw_left_x_px'] if source else None,reference_raw_left_y_px=source['raw_left_y_px'] if source else None,horizontal_error_px=x-ref['mouth_x_px'] if x is not None and ref else None,box_centered=row['centered'],reference_centered=abs(ref['mouth_x_px']-640)<=51.2 if ref else None,false_centered=bool(row['centered'] and abs(ref['mouth_x_px']-640)>51.2) if ref and s else None,reference_outside_side=('left' if ref['mouth_x_px']<588.8 else 'right' if ref['mouth_x_px']>691.2 else 'within') if ref else None,distance_m=z,raw_distance_m=row['raw_distance_m'],measurement_status=row['measurement_status'],measurement_reason=row['measurement_reason'],distance_source_frame=row['distance_sample_frame'],close_enough=row['close_enough'],mouth_datum_m=trial.get('mouth_datum_m'),body_datum_m=trial.get('body_datum_m'),optical_origin_offset_mm=trial.get('origin_offset_mm'),independent_mouth_optical_Z_m=trial.get('mouth_optical_Z_m'),distance_minus_mouth_datum_m=z-trial['mouth_datum_m'] if z is not None and trial.get('mouth_datum_m') is not None else None,distance_minus_body_datum_m=z-trial['body_datum_m'] if z is not None and trial.get('body_datum_m') is not None else None,formal_mouth_distance_error_m=None)
  # Datum differences stay differences until both optical origin and forward direction are independently verified.
  if z is not None and trial.get('geometry_verified') and trial.get('mouth_optical_Z_m') is not None:item['formal_mouth_distance_error_m']=z-trial['mouth_optical_Z_m']
  if ref:
   item['surface_diagnostic']=surface_diagnostic(folder,row,ref)
   (folder/'annotated').mkdir(exist_ok=True);im=cv2.imread(str(folder/row['rectified_left']));im=overlay(im,row,ref)
   for key,color in [('bucket_polygon_rectified',(255,180,0)),('body_polygon_rectified',(255,0,0))]:
    if ref.get(key):cv2.polylines(im,[np.round(ref[key]).astype(np.int32)],True,color,2)
   assert cv2.imwrite(str(folder/'annotated'/f"{row['frame_id']:06d}.jpg"),im)
  evaluated.append(item)
 with (folder/'evaluated_frames.jsonl').open('w') as f:
  for row in evaluated:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
 fields=[k for k in evaluated[0] if k!='surface_diagnostic']
 with (folder/'evaluated_frames.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(evaluated)
 def stats(values):
  values=[v for v in values if v is not None]
  return dict(n=len(values),min=min(values),max=max(values),mean=float(np.mean(values)),std=float(np.std(values))) if values else None
 new=[r for r in evaluated if r['measurement_status']=='new'];valid=[r for r in evaluated if r['distance_m'] is not None]
 pairs=[(a,b) for a,b in zip(evaluated,evaluated[1:]) if a['close_enough'] is not None and b['close_enough'] is not None]
 summary=dict(output_id=folder.parent.name,timing_mode=rows[0].get('timing_mode','legacy_see_environment'),placement_id=seq,calibration_profile=profile,frames=len(rows),detected=sum(r['detected'] for r in evaluated),new=len(new),held=sum(r['measurement_status']=='held' for r in evaluated),invalid=len(evaluated)-len(valid),reference_frames=sum(r['reference_mouth_x_px'] is not None for r in evaluated),horizontal_error_px=stats(r['horizontal_error_px'] for r in evaluated),box_center_x_px=stats(r['box_center_x_px'] for r in evaluated),new_raw_Z_m=stats(r['raw_distance_m'] for r in new),published_Z_m=stats(r['distance_m'] for r in valid),false_centered_count=sum(r['false_centered'] is True for r in evaluated),false_center_eval_frames=sum(r['false_centered'] is not None for r in evaluated),adjacent_valid_close_boundary_flips=sum(a['close_enough']!=b['close_enough'] for a,b in pairs),boundary_valid_pairs=len(pairs),trial=trial,reference_file_sha256=sha(refpath) if refpath.exists() else None,accepted='NO_OVERALL_ERROR_LIMITS_GIVEN',automatic_localizer='original_box_midpoint',automatic_sampler='original_expanded_bbox_roi',coordinate_policy='all errors computed within own calibration; independent points transferred via original-left pixels')
 save(folder/'summary.json',summary)
 allrows=[]
 for p in sorted((ROOT/'results').glob('*/seed2_REG_FP16/summary.json')):
  d=json.loads(p.read_text());allrows.append(dict(output_id=d.get('output_id',p.parent.parent.name),timing_mode=d.get('timing_mode','legacy_see_environment'),placement_id=d['placement_id'],calibration_profile=d['calibration_profile'],frames=d['frames'],detected=d['detected'],new=d['new'],held=d['held'],invalid=d['invalid'],reference_frames=d['reference_frames'],horizontal_error_min_px=(d['horizontal_error_px'] or {}).get('min'),horizontal_error_max_px=(d['horizontal_error_px'] or {}).get('max'),false_centered=d['false_centered_count'],false_center_eval_frames=d['false_center_eval_frames'],distance_boundary_flips=d['adjacent_valid_close_boundary_flips']))
 with (ROOT/'results/placements.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 print('V2_EVALUATED',json.dumps(summary,ensure_ascii=False),flush=True)
