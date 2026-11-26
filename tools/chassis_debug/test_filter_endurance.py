#!/usr/bin/env python3
"""Authorized suspended 8% trials, 120 seconds each, original UART controls."""
import json,signal,sys,time
from pathlib import Path
sys.path.insert(0,'/home/elf/deploy/serial_rx')
from chassis_probe import ChassisPort,HeldArmLimits
ROOT=Path('/home/elf/deploy/chassis_debug/all_encoder_filter_20261123')
RUN=ROOT/'endurance8_120s'

def trial(direction):
 out=RUN/direction;out.mkdir()
 p=None
 result={'direction':direction,'speed_field':8,'requested_seconds':120,'error':None,'stop_error':None,'drive_observed':False}
 with (out/'events.jsonl').open('x',buffering=1) as f:
  def log(k,d):f.write(json.dumps({'utc_epoch':time.time(),'monotonic':time.monotonic(),'kind':k,'data':d})+'\n')
  try:
   p=ChassisPort(HeldArmLimits('/home/elf/deploy/serial_rx/arm_reference.json'),log)
   result['initial_state']=p.state
   print('PREFLIGHT_OK',direction,json.dumps(p.state),flush=True)
   p.start_pulse(direction,8,1.2,continuous=True)
   # ChassisPort.tick handles this finite deadline with the normal disable path.
   p.deadline=p.started+120
   log('bounded_deadline',{'seconds':120,'deadline':p.deadline})
   next_report=0
   while p.phase!='idle':
    if (RUN/'STOP').exists() or (out/'STOP').exists():
     result['operator_stopped']=True;p.stop()
    p.tick()
    elapsed=time.monotonic()-p.started
    if elapsed>=next_report and p.drive_observed:
     print('RUNNING',direction,round(elapsed,2),'s',json.dumps({'enabled':p.state['enabled'],'state':p.state['state'],'system_fault':p.state['system_fault'],'motor_fault':p.state['motor_fault'],'command_count':p.state['command_count_mod256']}),flush=True)
     next_report+=30
    time.sleep(.01)
  except BaseException as exc:
   result['error']=repr(exc)
  finally:
   if p is not None:
    result['drive_observed']=p.drive_observed
    result['stop_error']=p.shutdown()
    result['final_state']=p.state
   (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print('TRIAL_FINISHED',json.dumps(result),flush=True)
 if result['error'] or result['stop_error'] or not result['drive_observed'] or result.get('operator_stopped'):
  raise RuntimeError('trial_failed_or_interrupted_no_next_motion')
 s=result['final_state']
 if s['enabled'] or s['state']!=0 or s['system_fault'] or s['motor_fault']:
  raise RuntimeError('parking_not_healthy_no_next_motion')
 return result

def main():
 RUN.mkdir(exist_ok=False)
 def interrupted(signum,frame):raise KeyboardInterrupt('signal_'+str(signum))
 for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:signal.signal(sig,interrupted)
 results=[]
 try:
  for direction in ['forward','backward']:
   if (RUN/'STOP').exists():raise RuntimeError('operator_stop')
   print('STARTING',direction,flush=True)
   results.append(trial(direction))
   if direction=='forward':
    print('PARKED_BREAK_8_SECONDS',flush=True);time.sleep(8)
  (RUN/'result.json').write_text(json.dumps({'success':True,'trials':results,'physical_observation':'await_user'},indent=2)+'\n')
 except BaseException as exc:
  (RUN/'result.json').write_text(json.dumps({'success':False,'trials':results,'error':repr(exc)},indent=2)+'\n')
  raise
if __name__=='__main__':main()
