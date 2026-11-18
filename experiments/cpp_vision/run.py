"""Verified, isolated board benchmark launcher. Never opens a device or serial port."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--impl',choices=['python-stock','python-matched','cpp','cpp-pipeline'],required=True)
    p.add_argument('--mode',choices=['sync','async'],default='async');p.add_argument('--scale',choices=[.5,1.],type=float,default=.5)
    p.add_argument('--seconds',type=float,default=20);p.add_argument('--frames',type=int,default=68);p.add_argument('--warmup',type=int,default=17);p.add_argument('--id',required=True)
    p.add_argument('--qa',action='store_true',help='historical moving-bucket subset plus a synthetic blank, numerical tests only')
    p.add_argument('--max-start-temp-c',type=float,help='wait for this SoC temperature before starting; never changes clocks or cooling')
    a=p.parse_args();base=Path(__file__).resolve().parent;os.chdir(base)
    import re
    assert re.fullmatch('[A-Za-z0-9_-]+',a.id)
    binding=json.loads((base/'prepared/binding.json').read_text());root=Path(binding['root']);model=(root/binding['model']['path']).resolve()
    for path,want in [(model,binding['model']['sha256']),(root/'calibrations/v2_radial2.json',binding['calibration_sha256']),
                      (base/'prepared/calibration.yml',binding['exported_calibration_sha256']),
                      (root/'conditions.json',binding['conditions_sha256'])]:
        actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==want
        print('VERIFIED',path,actual,flush=True)
    for record in binding['qa_frames'] if a.qa else binding['frames']:assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()==record['sha256']
    folder=base/'results'/a.id;assert not folder.exists(),'refusing to overwrite results'
    thermal=Path('/sys/class/thermal/thermal_zone0/temp')
    if a.max_start_temp_c is not None:
        wait_start=time.monotonic()
        while int(thermal.read_text())/1000>a.max_start_temp_c:
            if time.monotonic()-wait_start>300:raise TimeoutError('board did not cool within five minutes')
            time.sleep(1)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',BUCKET_V2_ROOT=str(root))
    if a.impl=='python-matched':env['PYTHONPATH']=str(base/'deps/install/python')
    corpus=base/('prepared/qa_corpus.txt' if a.qa else 'prepared/corpus.txt')
    if a.impl.startswith('cpp'):
        folder.mkdir();mode='pipeline' if a.impl=='cpp-pipeline' and a.mode=='async' else a.mode
        cmd=[str(base/'vision'),str(model),str(base/'prepared/calibration.yml'),str(corpus),str(folder/'frames.jsonl'),mode,str(a.scale),str(a.seconds),str(a.frames),str(a.warmup)]
    else:cmd=[sys.executable,'python_bench.py','--corpus',str(corpus),'--out',str(folder),'--mode',a.mode,'--scale',str(a.scale),'--seconds',str(a.seconds),'--frames',str(a.frames),'--warmup',str(a.warmup)]
    receipt=dict(args=vars(a),command=cmd,model=str(model),model_sha256=binding['model']['sha256'],calibration_sha256=binding['calibration_sha256'],display=False,save_images=False,log='buffered JSONL after timed window',serial=False,start_temp_c=int(thermal.read_text())/1000)
    receipt['runtime_sha256']=hashlib.sha256(Path('/usr/lib/librknnrt.so').read_bytes()).hexdigest()
    receipt['benchmark_sources']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in base.iterdir() if f.suffix in ('.py','.cpp','.sh')}
    (base/'results'/f'{a.id}_receipt.json').write_text(json.dumps(receipt,indent=2))
    with (base/'results'/f'{a.id}.log').open('x') as log:
        subprocess.run([sys.executable,'monitor.py','--out',str(base/'results'/f'{a.id}_memory.json'),*cmd],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    from compare import summary
    report=summary(folder/'frames.jsonl');(folder/'common_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
