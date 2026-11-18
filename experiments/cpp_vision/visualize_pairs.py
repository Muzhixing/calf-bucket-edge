"""Render recorded paired predictions; images are never used as accuracy truth."""
import json,sys
from pathlib import Path
import cv2,numpy as np
from compare import read
def main():
    base=Path(__file__).resolve().parent;binding=json.loads((base/'prepared/binding.json').read_text());root=Path(binding['root']);sys.path[:0]=[str(root),str(root/'tools')]
    from app.infrastructure import camera_config as c
    out=base/'results/representatives';out.mkdir(exist_ok=True)
    def panel(record,row,label,w=640):
        raw=cv2.imread(record['path'])[:,:1280];im=cv2.remap(raw,c.left_map1,c.left_map2,cv2.INTER_LINEAR)
        cv2.drawMarker(im,(640,360),(255,255,255),cv2.MARKER_CROSS,25,2)
        for b in row['detections']:cv2.rectangle(im,(b['left'],b['top']),(b['right'],b['bottom']),(0,220,0),2)
        im=cv2.resize(im,(w,int(w*720/1280)));im=cv2.copyMakeBorder(im,38,0,0,0,cv2.BORDER_CONSTANT)
        text=f'{label} | raw Z={row["raw"]:.6f}m' if row['raw'] is not None else f'{label} | invalid distance'
        cv2.putText(im,text,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55 if w==640 else .4,(255,255,255),1,cv2.LINE_AA)
        return im
    for scale in ('fast','full'):
        py=read(base/f'results/final_sync_python-stock_{scale}/frames.jsonl')[0]
        cpp=read(base/f'results/final_sync_cpp_{scale}/frames.jsonl')[0]
        r=binding['frames'][py['input_index']]
        assert cv2.imwrite(str(out/f'{scale}_python_cpp.png'),np.hstack([panel(r,py,'Python'),panel(r,cpp,'C++')]))
    qa={r['input_index']:r for r in read(base/'results/qa_final_cpp_fast/frames.jsonl')}
    panels=[panel(r,qa[i],f'QA {i:02d}',400) for i,r in enumerate(binding['qa_frames'])]
    while len(panels)%4:panels.append(np.zeros_like(panels[0]))
    assert cv2.imwrite(str(out/'qa_montage.png'),np.vstack([np.hstack(panels[i:i+4]) for i in range(0,len(panels),4)]))
if __name__=='__main__':main()
