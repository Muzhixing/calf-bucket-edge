"""Aggregate measured windows, process-tree PSS/RSS and source-aware parity."""
import argparse,csv,json,statistics
from pathlib import Path
from compare import read,quant

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results');p.add_argument('--tag',default='repeated');a=p.parse_args();root=Path(a.results)
    table=[];errors=[]
    for folder in sorted(root.glob(a.tag+'_*')):
        if not folder.is_dir() or not (folder/'common_summary.json').exists():continue
        s=json.loads((folder/'common_summary.json').read_text());rows=read(folder/'frames.jsonl');receipt=json.loads((root/(folder.name+'_receipt.json')).read_text())['args']
        mem=json.loads((root/(folder.name+'_memory.json')).read_text())['samples']
        mem=[m for m in mem if rows[0]['arrival']<=m['time']<=rows[-1]['published_at']]
        record=dict(id=folder.name,implementation=receipt['impl'],scale=receipt['scale'],frames=s['frames'],new=s['new'],fps=s['output_fps'],new_hz=s['new_hz'],latency_p50_ms=s['fresh_latency_ms']['p50'],latency_p95_ms=s['fresh_latency_ms']['p95'],
                    pss_median_mib=statistics.median([m['pss_kib']/1024 for m in mem]) if mem else None,
                    pss_peak_mib=max([m['pss_kib']/1024 for m in mem],default=None),rss_median_mib=statistics.median([m['rss_kib']/1024 for m in mem]) if mem else None,
                    max_temp_c=max([m['temp_mC']/1000 for m in mem if 'temp_mC' in m],default=None))
        table.append(record)
        ref=read(root/('parity_stock_fast' if receipt['scale']==.5 else 'parity_stock_full')/'frames.jsonl')
        by_index={r['input_index']:r for r in ref}
        previous_source=previous_value=None
        for row in rows:
            expected=by_index[row['input_index']];reason=[]
            if len(row['detections'])!=len(expected['detections']):reason.append('detection_count')
            for box,reference in zip(row['detections'],expected['detections']):
                if any(box[k]!=reference[k] for k in ('left','top','right','bottom')) or abs(box['score']-reference['score'])>1e-7:reason.append('detector')
            if row['status']=='new':
                if row['source'] is None or row['source']>row['camera_seq'] or row['source']==previous_source:reason.append('new_source')
                else:
                    raw=by_index[(row['source']-1)%len(by_index)]['raw']
                    if raw is None or abs(row['raw']-raw)>1e-6:reason.append('raw_vs_source_pair')
                if previous_value is not None and abs(row['published']-(.9*row['raw']+.1*previous_value))>1e-6:reason.append('smoothing')
            elif row['raw'] is not None:reason.append('not_new_but_has_raw')
            if row['status']=='invalid' and (row['published'] is not None or row['source'] is not None):reason.append('invalid_with_value')
            if row['status']=='held' and previous_source is not None and row['source']!=previous_source:reason.append('held_changed_source')
            if row['status']=='held' and previous_value is not None and abs(row['published']-previous_value)>1e-6:reason.append('held_changed_value')
            previous_source=row['source']
            previous_value=row['published']
            if reason:errors.append(dict(run=folder.name,reason=reason,row=row))
    if table:
        with (root/'benchmark_table.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    (root/'source_parity_failures.json').write_text(json.dumps(errors,indent=2))
    groups=[]
    for scale in (.5,1.):
        for impl in ('python-stock','python-matched','cpp','cpp-pipeline'):
            group=[r for r in table if r['implementation']==impl and r['scale']==scale]
            if group:groups.append(dict(scale=scale,implementation=impl,repeats=len(group),metrics={k:quant([r[k] for r in group if r[k] is not None]) for k in ('fps','new_hz','latency_p50_ms','latency_p95_ms','pss_median_mib','pss_peak_mib','max_temp_c')}))
    report=dict(runs=len(table),source_parity_failures=len(errors),groups=groups)
    (root/'aggregate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
