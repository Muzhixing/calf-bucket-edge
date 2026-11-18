"""Bounded real camera acquisition/decode probe, separate from replay benchmarks."""
import argparse,json,time
from pathlib import Path
import cv2
from compare import quant

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='/dev/v4l/by-id/usb-USB_Camera_USB_Camera_01.00.00-video-index0');p.add_argument('--out',required=True);p.add_argument('--frames',type=int,default=300);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cap=cv2.VideoCapture(a.device,cv2.CAP_V4L2)
    assert cap.isOpened()
    for key,value in [(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')),(cv2.CAP_PROP_FRAME_WIDTH,2560),(cv2.CAP_PROP_FRAME_HEIGHT,720),(cv2.CAP_PROP_FPS,30),(cv2.CAP_PROP_BUFFERSIZE,2)]:assert cap.set(key,value)
    rows=[]
    try:
        for i in range(30+a.frames):
            t=time.monotonic();assert cap.grab();grab=time.monotonic();ok,im=cap.retrieve();end=time.monotonic();assert ok and im.shape==(720,2560,3)
            if i>=30:rows.append(dict(grab_ms=(grab-t)*1000,retrieve_decode_ms=(end-grab)*1000,start=t,end=end))
        assert cv2.imwrite(str(out/'last_sbs.jpg'),im)
        report=dict(frames=len(rows),fps=len(rows)/(rows[-1]['end']-rows[0]['start']),grab_ms=quant([r['grab_ms'] for r in rows]),retrieve_decode_ms=quant([r['retrieve_decode_ms'] for r in rows]),meaning='V4L2 grab includes pacing/driver wait; retrieve includes MJPEG decode and BGR conversion, not exposure time')
        (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    finally:cap.release()
if __name__=='__main__':main()
