"""Run a benchmark and sample the complete process tree's PSS and RSS (KiB)."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('cmd',nargs=argparse.REMAINDER)
    a=p.parse_args();child=subprocess.Popen(a.cmd,start_new_session=True);rows=[]
    def stop(signum,frame):
        if child.poll() is None:os.killpg(child.pid,signal.SIGINT)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    while child.poll() is None:
        parents={}
        for f in Path('/proc').glob('[0-9]*/status'):
            try:
                d=dict(line.split(':',1) for line in f.read_text().splitlines());parents[int(f.parent.name)]=int(d['PPid'])
            except (OSError,ValueError,KeyError):pass
        ids={child.pid}
        while True:
            expanded=ids|{i for i,pid in parents.items() if pid in ids}
            if expanded==ids:break
            ids=expanded
        row=dict(time=time.monotonic(),pids=sorted(ids),pss_kib=0,rss_kib=0)
        for pid in ids:
            try:
                d=dict(line.split(':',1) for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines()[1:])
                row['pss_kib']+=int(d['Pss'].split()[0]);row['rss_kib']+=int(d['Rss'].split()[0])
            except (OSError,KeyError):pass
        try:row['temp_mC']=int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())
        except OSError:pass
        for key,path in [('cpu4_khz','/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq'),('cpu6_khz','/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq'),('npu_hz','/sys/class/devfreq/fdab0000.npu/cur_freq')]:
            try:row[key]=int(Path(path).read_text())
            except OSError:pass
        rows.append(row);time.sleep(.5)
    Path(a.out).write_text(json.dumps(dict(returncode=child.returncode,samples=rows),indent=2))
    raise SystemExit(child.returncode)
if __name__=='__main__':main()
