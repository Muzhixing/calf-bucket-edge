#!/usr/bin/env python3
"""Summarize UART evidence for bounded chassis tests, without inferring motion."""
import argparse,json,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
results=[]
for direction in ['forward','backward']:
 folder=args.run/direction
 if not (folder/'summary.json').exists():continue
 status=json.loads((folder/'summary.json').read_text())
 rows=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
 tx=[x for x in rows if x['kind']=='serial_tx']
 received=[(x['monotonic'],s) for x in rows if x['kind']=='serial_rx' for s in x['data']]
 frames=[s for _,s in received];active=[s for s in frames if s['enabled'] and s['state']==3]
 gaps=[(b[0]-a[0])*1000 for a,b in zip(received,received[1:])]
 result={'direction':direction,'speed_field':status['speed_field'],'duration_between_motion_disable_s':tx[1]['monotonic']-tx[0]['monotonic'] if len(tx)==2 else None,'serial_tx_count':len(tx),'serial_tx_hex':[x['data']['hex'] for x in tx],'telemetry_frames':len(frames),'driving_frames':len(active),'system_faults':sorted({s['system_fault'] for s in frames}),'motor_faults':sorted({s['motor_fault'] for s in frames}),'uart_error_counts':sorted({s['uart_error_count_mod256'] for s in frames}),'command_counts':sorted({s['command_count_mod256'] for s in frames}),'arm_poses':sorted({(s['servo_software_deg'],s['lift_position_mm'],s['slide_position_mm']) for s in frames}),'yaw_range': [min(s['yaw_deg'] for s in frames if s['yaw_deg'] is not None),max(s['yaw_deg'] for s in frames if s['yaw_deg'] is not None)] if any(s['yaw_deg'] is not None for s in frames) else None,'telemetry_max_gap_ms':max(gaps) if gaps else None,'telemetry_median_gap_ms':statistics.median(gaps) if gaps else None,'final_state':status.get('final_state'),'error':status['error'],'stop_error':status['stop_error'],'operator_stopped':status.get('operator_stopped',False)}
 results.append(result)
report={'scope':'Suspended chassis, original UART protocol, no running SWD samples; per-wheel smoothness requires user observation','trials':results}
(args.run/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
