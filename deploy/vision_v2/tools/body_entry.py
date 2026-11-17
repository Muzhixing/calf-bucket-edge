"""Mac body-only workflow; arguments are shell quoted with shlex.join."""
import argparse,subprocess,shlex
from entry import call,ROOT,SOCK,HOST,pull
def main():
    ap=argparse.ArgumentParser(description='capture ID [--body-datum-m M] | evaluate ID | preview | pull')
    ap.add_argument('action',choices=['capture','evaluate','preview','pull']);a,rest=ap.parse_known_args()
    if a.action in ['preview','pull']:
        subprocess.run(['python3',str(ROOT/'tools/entry.py'),a.action,*rest],check=True);return
    if subprocess.run(['ssh','-S',SOCK,'-O','check',HOST],capture_output=True).returncode:
        subprocess.run(['ssh','-M','-S',SOCK,'-o','ControlPersist=2h','-Nf',HOST],check=True)
    call(['python3','-u','tools/body_range.py',a.action,*rest],tty=a.action=='capture')
if __name__=='__main__':main()
