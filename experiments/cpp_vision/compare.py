"""Compare ordered per-pair outputs; timing summaries retain input-mode boundaries."""
import argparse,json,statistics
from pathlib import Path

def read(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def quant(v):
    if not v:return None
    v=sorted(v)
    def p(q):
        i=(len(v)-1)*q;a=int(i);b=min(a+1,len(v)-1)
        return v[a]+(v[b]-v[a])*(i-a)
    return dict(n=len(v),mean=statistics.mean(v),p50=p(.5),p95=p(.95),max=max(v))
def summary(path):
    rows=read(path);new=[r for r in rows if r['status']=='new'];dt=rows[-1]['published_at']-rows[0]['arrival']
    result=dict(frames=len(rows),detected=sum(bool(r['detections']) for r in rows),duration=dt,output_fps=len(rows)/dt,new=len(new),new_hz=len(new)/dt,
                fresh_latency_ms=quant([r['latency_ms'] for r in new]),status_counts={s:sum(r['status']==s for r in rows) for s in ('new','held','invalid')})
    for field in ('detector','worker'):
        values=[r[field] for r in rows if r.get(field)]
        result[field]={k:quant([v[k] for v in values if k in v]) for k in sorted({k for v in values for k in v if k.endswith('_ms')})}
    result['result_pickup_ms']=quant([(r['published_at']-r['worker']['worker_end'])*1000 for r in rows if r.get('worker')])
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('--out',required=True);a=p.parse_args()
    ref,got=read(a.reference),read(a.candidate);errors=[];coords=[];scores=[];distances=[];published=[]
    assert len(ref)==len(got),'different frame count'
    for r,g in zip(ref,got):
        reasons=[]
        if r['input_index']!=g['input_index']:reasons.append('input_index')
        if len(r['detections'])!=len(g['detections']):reasons.append('detection_count')
        for x,y in zip(r['detections'],g['detections']):
            c=max(abs(x[k]-y[k]) for k in ('left','top','right','bottom'));s=abs(x['score']-y['score']);coords.append(c);scores.append(s)
            if c or s>1e-7:reasons.append('box_or_score')
        for key,values in [('raw',distances),('published',published)]:
            if (r[key] is None)!=(g[key] is None):reasons.append(key+'_validity')
            elif r[key] is not None:
                d=abs(r[key]-g[key]);values.append(d)
                if d>1e-6:reasons.append(key+'_distance')
        for key in ('status','source'):
            if r[key]!=g[key]:reasons.append(key)
        if reasons:errors.append(dict(frame=r['frame'],reasons=reasons,reference=r,candidate=g))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    report=dict(pairs=len(ref),mismatches=len(errors),box_max_px=max(coords,default=0),score_max_abs=max(scores,default=0),raw_max_m=max(distances,default=0),published_max_m=max(published,default=0),reference=summary(a.reference),candidate=summary(a.candidate))
    (out/'comparison.json').write_text(json.dumps(report,indent=2));(out/'mismatches.json').write_text(json.dumps(errors,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('reference','candidate')}))
if __name__=='__main__':main()
