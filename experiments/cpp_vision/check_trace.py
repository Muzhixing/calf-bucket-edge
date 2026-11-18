"""Validate asynchronous measurements against the source pair, including loss."""
import argparse,json
from pathlib import Path
from compare import read
def main():
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('trace');p.add_argument('--out',required=True);a=p.parse_args()
    ref={r['input_index']:r for r in read(a.reference)};rows=read(a.trace);errors=[];previous=None
    for row in rows:
        reasons=[];expected=ref[row['input_index']]
        if len(row['detections'])!=len(expected['detections']):reasons.append('detection_count')
        for x,y in zip(row['detections'],expected['detections']):
            if any(x[k]!=y[k] for k in ('left','top','right','bottom')) or abs(x['score']-y['score'])>1e-7:reasons.append('box_or_score')
        if not row['detections'] and row['status']!='invalid':reasons.append('target_loss_retained_value')
        if row['status']=='new':
            source=row['source']
            if source is None or source>row['camera_seq']:reasons.append('source_order')
            else:
                z=ref[(source-1)%len(ref)]['raw']
                if z is None or row['raw'] is None or abs(z-row['raw'])>1e-6:reasons.append('raw_vs_source')
            if previous and previous['source']==source:reasons.append('reused_marked_new')
        elif row['raw'] is not None:reasons.append('fresh_value_label')
        if row['status']=='held' and previous:
            if row['source']!=previous['source'] or row['published']!=previous['published']:reasons.append('held_changed')
        if row['status']=='invalid' and (row['published'] is not None or row['source'] is not None):reasons.append('invalid_with_value')
        if row['published'] is not None and row['latency_ms']>1505:reasons.append('expired_value')
        if reasons:errors.append(dict(reasons=reasons,row=row))
        previous=row
    result=dict(frames=len(rows),failures=len(errors),states={s:sum(r['status']==s for r in rows) for s in ('new','held','invalid')},discarded=sum(r.get('discarded',False) for r in rows),errors=errors)
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='errors'}))
if __name__=='__main__':main()
