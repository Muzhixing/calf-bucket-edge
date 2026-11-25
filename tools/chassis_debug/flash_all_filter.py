#!/usr/bin/env python3
"""One explicitly authorized candidate flash with exclusive UART and readback."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/elf/deploy/chassis_debug')
REL=ROOT/'all_encoder_filter_20261123'
RUN=REL/'flash_run'
sys.path.insert(0,'/home/elf/deploy/serial_rx')
from chassis_probe import ChassisPort,HeldArmLimits

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):(RUN/name).write_text(json.dumps(obj,indent=2)+'\n')

def main():
 RUN.mkdir(exist_ok=True)
 events=(RUN/'serial_events_program.jsonl').open('x',buffering=1)
 def log(k,d):events.write(json.dumps({'time':time.time(),'kind':k,'data':d})+'\n')
 port=None
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(ROOT/'runtime/usr/lib/aarch64-linux-gnu')+':'+str(ROOT/'runtime/lib/aarch64-linux-gnu')
 base=[str(ROOT/'runtime/usr/bin/openocd'),'-s',str(ROOT/'runtime/usr/share/openocd/scripts')]
 def run(name,args):
  p=subprocess.run(base+args,env=env,capture_output=True,text=True,timeout=55)
  (RUN/(name+'.log')).write_text(p.stdout+p.stderr)
  print(name,p.returncode,p.stdout+p.stderr,flush=True)
  if p.returncode:raise RuntimeError(name+'_failed')
 ro=(ROOT/'attach_readonly.cfg').read_text().rsplit('shutdown',1)[0]
 guards='''
if {[stm32f4x.cpu curstate] ne "running"} {error "cpu_not_running"}
foreach addr {0x40014038 0x40010038 0x4001003c 0x40010040} {
 mem2array pwm 32 $addr 1
 if {$pwm(0) != 0} {error "PWM_not_zero"}
}
'''
 try:
  image=REL/'build/f407_all_encoder_filter.hex';binary=REL/'build/f407_all_encoder_filter.bin'
  assert sha(image)=='e371d9328a9f3803074a822461106586c481245dde8828a03e3196bec528d584'
  assert sha(binary)=='c55dd404078189c1cad35c5ef671315577f26c2d80e8330970c08febff9d4e00'
  port=ChassisPort(HeldArmLimits('/home/elf/deploy/serial_rx/arm_reference.json'),log)
  save('preflash_state.json',port.state)
  assert not port.state['enabled'] and port.state['state']==0
  backup=RUN/'halted1_512k.bin'
  assert len(backup.read_bytes())==524288
  assert backup.read_bytes()==(RUN/'halted2_512k.bin').read_bytes()
  assert sha(backup)=='7841f900fe332a5511b37610e0c53879caca482cd889f376837220c07abee6eb'
  cfg=RUN/'preprogram.cfg';cfg.write_text(ro+guards+'\nshutdown\n')
  run('preprogram',['-f',str(cfg)])
  print('BACKUP_VERIFIED',sha(backup),flush=True)
  readback=RUN/'after_flash_image.bin'
  run('program',['-f','interface/stlink.cfg','-c','transport select hla_swd','-f','target/stm32f4x.cfg','-c','reset_config none','-c','adapter speed 1000','-c','program '+str(image)+' verify','-c',f'dump_image {readback} 0x08000000 {len(binary.read_bytes())}','-c','verify_image '+str(binary)+' 0x08000000 bin','-c','reset run','-c','shutdown'])
  assert readback.read_bytes()==binary.read_bytes(),'halted_flash_byte_mismatch'
  # Drop startup/buffered history by observation, then require fresh stable reports.
  deadline=time.monotonic()+3
  while time.monotonic()<deadline:port.poll();time.sleep(.02)
  frames=[];deadline=time.monotonic()+3
  while time.monotonic()<deadline:frames.extend(port.poll());time.sleep(.02)
  save('postflash_frames.json',frames)
  assert len(frames)>=8,'missing_postflash_telemetry'
  for s in frames:
   assert not s['enabled'] and s['state']==0 and s['system_fault']==0 and s['motor_fault']==0,s
   assert s['command_count_mod256']==0 and not s['lift_zero_confirmed'] and not s['slide_zero_confirmed'],s
   assert s['angle_valid'] and s['heading_reference_established'],s
   assert [s['servo_software_deg'],s['lift_position_mm'],s['slide_position_mm']]==[90,0,0],s
  readback=RUN/'after_flash_image.bin'
  cfg=RUN/'after.cfg'
  cfg.write_text(ro+guards+'''
foreach addr {0x40000018 0x40000418 0x40000818 0x40010418} {
 mem2array filter 32 $addr 1
 echo "FILTER $addr $filter(0)"
 if {$filter(0) != 0xf1f1} {error "filter_mismatch"}
}
'''+ '\nshutdown\n')
  run('after',['-f',str(cfg)])
  assert readback.read_bytes()==binary.read_bytes(),'flash_byte_mismatch'
  save('result.json',{'success':True,'old_flash_sha256':sha(backup),'new_flash_image_sha256':sha(readback),'new_hex_sha256':sha(image),'all_filters':'0xf1f1','pwm':[0,0,0,0],'postflash_state':frames[-1],'zero_registration':False,'motion_commands':0})
  print('FLASH_AND_STARTUP_VERIFIED',flush=True)
 except BaseException as exc:
  save('error.json',{'error':repr(exc)})
  raise
 finally:
  if port is not None and port.fd is not None:os.close(port.fd);port.fd=None
  events.close()
if __name__=='__main__':main()
