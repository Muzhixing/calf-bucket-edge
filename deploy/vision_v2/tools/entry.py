"""Mac launcher for the engineering-only RK3588 vision package."""
import argparse,datetime,os,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/elf/deploy/vision_v2'
HOST=os.environ.get('BUCKET_V2_HOST','elf@192.168.31.175')
SOCK=os.environ.get('BUCKET_SSH_SOCKET','/tmp/bucket-v1-20261116.sock')
def call(args,tty=False):
    command=f'cd {REMOTE} && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(args)
    subprocess.run(['ssh',*(['-tt'] if tty else []),'-S',SOCK,HOST,command],check=True)
def pull(directory):
    subprocess.run(['rsync','-a','-e',f'ssh -S {SOCK}',f'{HOST}:{REMOTE}/{directory}/',str(ROOT/directory)+'/'],check=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['capture','evaluate','preview','pull','verify','live']);a,rest=ap.parse_known_args()
    if subprocess.run(['ssh','-S',SOCK,'-O','check',HOST],capture_output=True).returncode:
        subprocess.run(['ssh','-M','-S',SOCK,'-o','ControlPersist=2h','-Nf',HOST],check=True)
    if a.action=='pull':
        for d in ['captures','results','logs','trial_records']:pull(d)
    elif a.action=='verify':call(['python3','tools/verify.py'])
    elif a.action in ['capture','evaluate']:call(['python3','-u','tools/body_range.py',a.action,*rest],tty=a.action=='capture')
    elif a.action=='live':call(['python3','-u','vision_v2.py',*rest],tty=True)
    else:
        name='preview_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        call(['python3','-u','vision_v2.py','--id',name,'--seconds','15','--max-frames','1'])
        pull('results');subprocess.run(['open',str(ROOT/'results'/f'{name}__v2_radial2/seed2_REG_FP16/overlays/000001.jpg')],check=True)
if __name__=='__main__':main()
