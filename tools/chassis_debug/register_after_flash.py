#!/usr/bin/env python3
"""Register physically confirmed B/C origins while disabled after this flash."""
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,'/home/elf/deploy/serial_rx')
from chassis_probe import ArmPort,HeldArmLimits,pose
ROOT=Path('/home/elf/deploy/chassis_debug/all_encoder_filter_20261123')
class RegistrationPort(ArmPort):
 @staticmethod
 def check_state(s):
  if s['system_fault'] or s['motor_fault'] or not s['angle_valid'] or not s['heading_reference_established']:
   raise RuntimeError('fault_or_heading_unready')

def main():
 assert json.loads((ROOT/'flash_run/result.json').read_text())['success']
 out=ROOT/'register_origins';out.mkdir(exist_ok=False)
 f=(out/'events.jsonl').open('x',buffering=1)
 def log(k,d):f.write(json.dumps({'time':time.time(),'kind':k,'data':d})+'\n')
 p=RegistrationPort(HeldArmLimits('/home/elf/deploy/serial_rx/arm_reference.json'),log)
 try:
  original=pose(p.state)
  assert original==[90,0,0] and not p.state['enabled']
  assert not p.state['lift_zero_confirmed'] and not p.state['slide_zero_confirmed']
  log('physical_confirmation',{'source':'user confirmed suspended chassis, B top, C fully retracted and arm unobstructed before authorized flash'})
  for flags in [6,0]:
   before=p.state['command_count_mod256'];expected=(before+1)%256
   frame=bytearray(p.frame(original,False));frame[8]=flags
   assert p.gateway.write_control_frame(bytes(frame),repeats=1)
   acknowledged=False;deadline=time.monotonic()+2
   while time.monotonic()<deadline:
    for s in p.poll():
     p.check_state(s)
     assert not s['enabled'] and s['state']==0 and pose(s)==original
     assert s['command_count_mod256'] in [before,expected]
     if s['command_count_mod256']==expected:
      assert s['result']==0 and s['lift_zero_confirmed'] and s['slide_zero_confirmed']
      acknowledged=True
    if acknowledged:break
    time.sleep(.01)
   assert acknowledged,'zero_ack_timeout'
  (out/'result.json').write_text(json.dumps(p.state,indent=2)+'\n')
  print(json.dumps(p.state),flush=True)
 finally:
  os.close(p.fd);p.fd=None;f.close()
if __name__=='__main__':main()
