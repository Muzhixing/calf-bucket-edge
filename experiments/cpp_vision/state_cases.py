"""Build a deterministic present/lost/reacquired replay, not an accuracy dataset."""
import json
from pathlib import Path
from compare import read
def main():
    base=Path(__file__).resolve().parent;binding=json.loads((base/'prepared/binding.json').read_text())
    reference={r['input_index']:r for r in read(base/'results/final_sync_python-stock_fast/frames.jsonl')}
    blank=binding['qa_frames'][-1];cases=[];rows=[]
    for index,count in [(0,30),(None,15),(3,30),(None,15)]:
        record=blank if index is None else binding['frames'][index]
        for _ in range(count):
            row=dict(reference[index]) if index is not None else dict(detections=[],raw=None,published=None,status='invalid',source=None)
            row['input_index']=len(cases);cases.append(record);rows.append(row)
    out=base/'results/state_cases';out.mkdir(exist_ok=True)
    (out/'corpus.txt').write_text(''.join(r['path']+'\n' for r in cases))
    (out/'reference.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'binding.json').write_text(json.dumps(cases,indent=2))
if __name__=='__main__':main()
