"""Sequential paired repeats. Never run this concurrently with a build or other benchmark."""
import argparse,json,subprocess,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--seconds',type=float,default=20);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--controlled',action='store_true',help='match starting temperature and use the separated C++ pipeline')
    a=p.parse_args()
    for scale,name in [(.5,'fast'),(1.,'full')]:
        for repeat in range(a.repeats):
            impls=['python-stock','python-matched','cpp-pipeline' if a.controlled else 'cpp']
            # Rotate order to expose sustained temperature and order effects.
            impls=impls[repeat%3:]+impls[:repeat%3]
            for impl in impls:
                ident=f'{a.tag}_{name}_{impl}_{repeat+1}'
                print('START',ident,flush=True)
                cmd=[sys.executable,'run.py','--impl',impl,'--mode','async','--scale',str(scale),'--seconds',str(a.seconds),'--id',ident]
                if a.controlled:cmd+=['--max-start-temp-c','60']
                subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
                result=json.loads((Path('results')/ident/'common_summary.json').read_text())
                print('DONE',ident,'fps',round(result['output_fps'],2),'new_hz',round(result['new_hz'],2),'latency_p50',round(result['fresh_latency_ms']['p50'],1),flush=True)
if __name__=='__main__':main()
