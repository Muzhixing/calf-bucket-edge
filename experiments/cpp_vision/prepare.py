"""Prepare an immutable replay list and OpenCV matrices from the deployed V2."""
import argparse, hashlib, json, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.root).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(root/'tools'),str(root)]
    from verify import verify
    verify()
    from app.infrastructure import camera_config as c
    import cv2
    f=cv2.FileStorage(str(out/'calibration.yml'),cv2.FILE_STORAGE_WRITE)
    for key,mat in [('K1',c.left_camera_matrix),('D1',c.left_distortion),('K2',c.right_camera_matrix),('D2',c.right_distortion),('R1',c.R1),('R2',c.R2),('P1',c.P1),('P2',c.P2),('T',c.T)]: f.write(key,mat)
    f.release()
    records=[]
    for name in ['engineering_capture_20261117_01','perf_record_final_01']:
        folder=root/'captures'/name
        for row in map(json.loads,(folder/'frames.jsonl').read_text().splitlines()):
            path=folder/row['path'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==row['sha256']
            records.append(dict(path=str(path),sha256=digest,original_monotonic_ns=row['monotonic_ns'],sequence=name,frame_id=row['frame_id']))
    (out/'corpus.txt').write_text(''.join(r['path']+'\n' for r in records))
    qa=[]
    history=Path('/home/elf/research/rk3588_field_validation_20261116/placed/approach_A_20261116_113702')
    if (history/'frames.jsonl').exists():
        old=list(map(json.loads,(history/'frames.jsonl').read_text().splitlines()))
        for i in sorted(set(range(0,len(old),20))|{len(old)-1}):
            row=old[i];path=history/row['path'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==row['sha256']
            qa.append(dict(path=str(path),sha256=digest,original_monotonic_ns=row['monotonic_ns'],sequence=history.name,frame_id=row['frame_id']))
    import numpy as np
    blank=out/'synthetic_blank.png';assert cv2.imwrite(str(blank),np.zeros((720,2560,3),np.uint8))
    qa.append(dict(path=str(blank),sha256=hashlib.sha256(blank.read_bytes()).hexdigest(),sequence='synthetic_blank_for_invalid_state',frame_id=0))
    (out/'qa_corpus.txt').write_text(''.join(r['path']+'\n' for r in qa))
    receipt=dict(root=str(root),model=json.loads((root/'models.json').read_text())[0],calibration_sha256=c.CALIBRATION_SHA256,
                 exported_calibration_sha256=hashlib.sha256((out/'calibration.yml').read_bytes()).hexdigest(),
                 conditions_sha256=hashlib.sha256((root/'conditions.json').read_bytes()).hexdigest(),
                 opencv_version=cv2.__version__,frames=records,qa_frames=qa,timing='original timestamps retained; paced replay separately reschedules these 17 frames at 30 Hz')
    (out/'binding.json').write_text(json.dumps(receipt,indent=2))
    (out/'opencv_stock_build.txt').write_text(cv2.getBuildInformation())
    print('CORPUS',len(records),out)
if __name__=='__main__':main()
