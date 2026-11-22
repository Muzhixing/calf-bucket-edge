"""Body-only capture/evaluation on the unchanged vision baseline, no actuators."""
import argparse,csv,json,re,subprocess,sys
from pathlib import Path
from support import ROOT,save,sha

def evaluate(ident):
    record_path=ROOT/'trial_records'/f'{ident}.json'
    record=json.loads(record_path.read_text())
    folder=ROOT/'results'/f'{ident}__printed_30mm_20261122'/'seed2_REG_FP16'
    rows=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()]
    items=[]
    for r in rows:
        z=r['distance_m'];datum=record.get('body_datum_m')
        verified=record['geometry_verified']
        optical=datum-record['origin_offset_mm']/1000 if verified else None
        items.append(dict(placement_id=ident,frame_id=r['frame_id'],detected=r['selected'] is not None,
            score=(r['selected'] or {}).get('score'),box_center_x_px=r['center_x_px'],
            box_centered=r['centered'],distance_m=z,raw_distance_m=r['raw_distance_m'],
            measurement_status=r['measurement_status'],measurement_reason=r['measurement_reason'],
            distance_source_frame=r['distance_sample_frame'],body_datum_m=datum,
            datum_direction=record['direction'],uncalibrated_difference_to_datum_m=z-datum if z is not None and datum is not None else None,
            formal_body_Z_error_m=z-optical if z is not None and optical is not None else None,
            baseline_sampler='original_expanded_bbox_roi_NOT_body_specific',
            mouth_accuracy_requested=False,travel_to_enter_bucket_m=None))
    out=folder/'body_assessment';out.mkdir(exist_ok=True)
    with (out/'frames.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    with (out/'frames.jsonl').open('w') as f:
        for r in items:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    summary=dict(placement_id=ident,frames=len(items),detected=sum(x['detected'] for x in items),
        states={s:sum(x['measurement_status']==s for x in items) for s in ['new','held','invalid']},
        new_raw_Z_m=[x['raw_distance_m'] for x in items if x['measurement_status']=='new'],
        body_datum_m=record.get('body_datum_m'),geometry_verified=record['geometry_verified'],
        mouth_accuracy_requested=False,body_surface_correctness_verified=False,
        travel_to_enter_bucket_m=None,travel_reason='work-part geometry and independent range check required',
        accepted='MEASURED_ERROR_LIMIT_NOT_YET_ESTABLISHED',trial_sha256=sha(record_path))
    save(out/'summary.json',summary)
    placements=[]
    for p in sorted((ROOT/'results').glob('*/seed2_REG_FP16/body_assessment/summary.json')):
        s=json.loads(p.read_text());v=s['new_raw_Z_m']
        placements.append(dict(placement_id=s['placement_id'],frames=s['frames'],detected=s['detected'],
            new=s['states']['new'],held=s['states']['held'],invalid=s['states']['invalid'],
            body_datum_m=s['body_datum_m'],new_Z_min_m=min(v) if v else None,new_Z_max_m=max(v) if v else None,
            geometry_verified=s['geometry_verified'],body_surface_verified=s['body_surface_correctness_verified'],
            travel_to_enter_bucket_m=s['travel_to_enter_bucket_m']))
    with (ROOT/'results/body_placements.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(placements[0]));w.writeheader();w.writerows(placements)
    print('BODY_RANGE_EVALUATED',json.dumps(summary,ensure_ascii=False),flush=True)

def main():
    from verify import verify
    verify()
    for p,h in json.loads((ROOT/'config/body_tool_hashes.json').read_text()).items():
        assert sha(ROOT/p)==h,('body tool changed',p)
    ap=argparse.ArgumentParser(description='Body range baseline only; no mouth input or hardware commands')
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('capture');p.add_argument('id');p.add_argument('--body-datum-m',type=float)
    p.add_argument('--direction',choices=['horizontal','rectified_axis','slant','unverified'],default='horizontal')
    p.add_argument('--origin-offset-mm',type=float);p.add_argument('--geometry-verified',action='store_true')
    p.add_argument('--reference-note',default='D left camera housing front mark to B visible front body surface; operator reading')
    p.add_argument('--seconds',type=float,default=8);p.add_argument('--max-frames',type=int,default=12)
    p=sub.add_parser('evaluate');p.add_argument('id')
    a=ap.parse_args();assert re.fullmatch('[A-Za-z0-9_-]+',a.id)
    if a.command=='evaluate':evaluate(a.id);return
    assert a.body_datum_m is None or 0<a.body_datum_m<3
    assert 0<a.seconds<=30 and a.max_frames>0
    if a.geometry_verified:
        assert a.body_datum_m is not None and a.direction=='rectified_axis' and a.origin_offset_mm is not None
    record=dict(placement_id=a.id,phase='body_range_only',mouth_accuracy_requested=False,
        body_datum_m=a.body_datum_m,direction=a.direction,origin_offset_mm=a.origin_offset_mm,
        geometry_verified=a.geometry_verified,reference_note=a.reference_note,
        datum='D: left camera housing front mark; not assumed optical centre',
        body_endpoint='B: visible body front surface point at a repeatable marked height',
        camera_pose_instruction='keep current full stereo rig pose fixed; operator must report changes',
        actuation=False,baseline_sampler='original expanded bbox ROI; surface validity pending',
        work_part_geometry=None,allowed_error_m=None)
    reservation=ROOT/'trial_records'/f'{a.id}.json'
    assert not reservation.exists() and not (ROOT/'captures'/a.id).exists() and not (ROOT/'results'/f'{a.id}__printed_30mm_20261122').exists(),'Use a new ID'
    reservation.parent.mkdir(exist_ok=True);save(reservation,record)
    save(ROOT/'trial_records'/f'{a.id}_body_tools_build.json',json.loads((ROOT/'config/body_tool_hashes.json').read_text()))
    print('BODY_REFERENCE',record,flush=True)
    try:
        subprocess.run([sys.executable,'-u',str(ROOT/'vision_v2.py'),'--id',a.id,'--seconds',str(a.seconds),'--max-frames',str(a.max_frames)],check=True)
    finally:
        cap=ROOT/'captures'/a.id
        if cap.exists():save(cap/'trial.json',record)
    evaluate(a.id)
if __name__=='__main__':main()
