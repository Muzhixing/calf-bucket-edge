"""Open a localhost SSH tunnel and the isolated board-side visual preview."""
import argparse,datetime,shlex,subprocess,threading,webbrowser
from entry import SOCK,HOST,REMOTE
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=3600);ap.add_argument('--port',type=int,default=8788);ap.add_argument('--id');ap.add_argument('--mode', choices=['sync','async'], default='async')
    ap.add_argument('--cores', choices=['auto','0_1_2'], default='auto')
    ap.add_argument('--depth-scale', type=float, choices=[1.0,0.5], default=1.0)
    a=ap.parse_args()
    ident=a.id or 'preview_live_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    command=f'cd {REMOTE} && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(['python3','-u','tools/live_viewer.py','--id',ident,'--seconds',str(a.seconds),'--port',str(a.port),'--mode',a.mode,'--cores',a.cores,'--depth-scale',str(a.depth_scale)])
    if subprocess.run(['ssh','-S',SOCK,'-O','check',HOST],capture_output=True).returncode:
        subprocess.run(['ssh','-M','-S',SOCK,'-o','ControlPersist=2h','-Nf',HOST],check=True)
    url=f'http://127.0.0.1:{a.port}/';print(url+' ; Ctrl+C stops preview and releases camera',flush=True)
    threading.Timer(3,lambda:webbrowser.open(url)).start()
    subprocess.run(['ssh','-tt','-S',SOCK,'-o','ExitOnForwardFailure=yes','-L',f'{a.port}:127.0.0.1:{a.port}',HOST,command],check=True)
if __name__=='__main__':main()
