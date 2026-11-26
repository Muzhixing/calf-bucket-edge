"""Plot the reversible B input-filter comparison; retain raw data separately."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2]/'outputs/chassis_debug_20261123'
fig,axes=plt.subplots(2,3,figsize=(13,6),sharey='row',layout='constrained')
for col,(phase,title) in enumerate([
    ('baseline6','Original filter = 6'),('candidate15','Stronger filter = 15'),
    ('return6','Restored filter = 6')]):
    run='b_filter_'+phase+'_20261123_02'
    events=[json.loads(l) for l in (root/'serial'/run/'events.jsonl').read_text().splitlines()]
    tx=[e for e in events if e['kind']=='serial_tx'];begin,end=tx[0]['monotonic'],tx[-1]['monotonic']
    rows=[json.loads(l) for l in (root/'registers'/run/'registers.jsonl').read_text().splitlines()]
    rows=[r for r in rows if begin+1<r['start_monotonic']<end-.1]
    t=[r['start_monotonic']-begin for r in rows]
    pwm=[r['pwm_lb']/10 for r in rows]
    d=[-((b['encoder_lb']-a['encoder_lb']+32768)%65536-32768) for a,b in zip(rows,rows[1:])]
    # One isolated zero read creates two huge, physically implausible deltas.
    # Leave gaps rather than interpolating or claiming they were actual speeds.
    d=[v if abs(v)<=1000 else float('nan') for v in d]
    axes[0,col].plot(t,pwm,lw=1,color='#2457A7');axes[0,col].set_title(title)
    axes[1,col].plot(t[1:],d,lw=.8,color='#2457A7')
    axes[1,col].scatter([x for x,v in zip(t[1:],d) if v<0],
                         [v for v in d if v<0],color='#C34343',s=10)
    axes[1,col].axhline(0,color='#666666',lw=.6)
    axes[1,col].set_xlabel('Seconds after command')
    for ax in axes[:,col]:ax.grid(alpha=.2)
axes[0,0].set_ylabel('B PWM duty (%)');axes[0,0].set_ylim(-1,31)
axes[1,0].set_ylabel('B encoder counts per ~40 ms')
fig.suptitle('B / right-front encoder: same 10% backward command, only input filter changed\n'
             'Negative count changes oppose command; isolated invalid reads appear as gaps',fontsize=12)
fig.savefig(root/'b_filter_comparison.png',dpi=160)
