"""Plot measured PWM and encoder deltas; no motor model or RPM assumptions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]/'outputs/chassis_debug_20261123'
RUN = 'verified_speed10_20261123_01'
events = [json.loads(l) for l in (ROOT/'serial'/RUN/'events.jsonl').read_text().splitlines()]
tx = [e for e in events if e['kind']=='serial_tx']
start,end = tx[0]['monotonic'],tx[-1]['monotonic']
rows = [json.loads(l) for l in (ROOT/'registers'/RUN/'registers.jsonl').read_text().splitlines()]
rows = [r for r in rows if start+1<r['start_monotonic']<end-.1]
fig,axes = plt.subplots(2,2,figsize=(12,6),sharex=True,layout='constrained')
for col,(wheel,title) in enumerate([('lf','A channel (encoder: physical right rear)'),('lb','B channel (encoder: physical right front)')]):
    t=[r['start_monotonic']-start for r in rows]
    pwm=[r['pwm_'+wheel]/10 for r in rows]
    d=[-((b['encoder_'+wheel]-a['encoder_'+wheel]+32768)%65536-32768)
       for a,b in zip(rows,rows[1:])]
    axes[0,col].plot(t,pwm,color='#2457A7',lw=1)
    axes[0,col].set(title=title,ylabel='PWM duty (%)',ylim=(-1,31))
    axes[1,col].plot(t[1:],d,color='#2457A7',lw=.7,alpha=.7)
    axes[1,col].scatter([x for x,v in zip(t[1:],d) if v<0],
                        [v for v in d if v<0],s=15,color='#C34343',zorder=3)
    axes[1,col].axhline(0,color='#555555',lw=.7)
    axes[1,col].set(ylabel='Encoder counts per ~40 ms',xlabel='Seconds after command')
    for ax in axes[:,col]: ax.grid(alpha=.2)
fig.suptitle('Suspended chassis: constant backward command at 10%\n'
             'Negative count changes oppose the commanded direction',fontsize=13)
fig.savefig(ROOT/'wheel_feedback.png',dpi=160)
