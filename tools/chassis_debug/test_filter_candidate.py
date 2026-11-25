#!/usr/bin/env python3
"""Bounded UART tests; running SWD readback proved unreliable, so omit it."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/elf/deploy/chassis_debug');SERIAL=Path('/home/elf/deploy/serial_rx')
RUN=ROOT/'all_encoder_filter_20261123/motion_test'
def main():
 assert json.loads((ROOT/'all_encoder_filter_20261123/flash_run/result.json').read_text())['success']
 RUN.mkdir(exist_ok=False)
 for direction in ['backward','forward']:
  out=RUN/direction;serial=out/'serial';out.mkdir()
  motor=None
  try:
   print('STARTING',direction,'10 percent; about 10 seconds',flush=True)
   motor=subprocess.Popen(['python3','-u',str(SERIAL/'chassis_probe.py'),'--direction',direction,'--speed','10','--continuous','--out',str(serial)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   deadline=time.monotonic()+11.5
   while motor.poll() is None and time.monotonic()<deadline:time.sleep(.05)
  finally:
   if motor is not None:
    if serial.is_dir():(serial/'STOP').touch()
    try:message=motor.communicate(timeout=4)[0]
    except subprocess.TimeoutExpired:
     motor.terminate();message=motor.communicate(timeout=4)[0]
    (out/'motor.log').write_text(message)
    print(message,flush=True)
  assert motor is not None and motor.returncode==0,'trial_failed'
  result=json.loads((serial/'summary.json').read_text());state=result['final_state']
  assert not state['enabled'] and state['state']==0 and not state['system_fault'] and not state['motor_fault']
  print('STOPPED_VERIFIED',direction,flush=True)
  time.sleep(3)
 (RUN/'result.json').write_text(json.dumps({'success':True,'directions':['backward','forward'],'speed_field':10,'stopped':True,'faults':0,'running_swd_sampling':False,'physical_smoothness':'await_user_observation'},indent=2)+'\n')
if __name__=='__main__':main()
