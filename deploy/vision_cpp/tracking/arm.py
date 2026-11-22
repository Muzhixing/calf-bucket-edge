"""Exclusive original-protocol serial owner and bounded image-space policy."""
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import select
import sys
import termios
import time
from collections import deque

SERIAL_ROOT = Path(__file__).resolve().parents[2]/'serial_rx'
sys.path.insert(0,str(SERIAL_ROOT))
from receive import Parser, DEFAULT_PORT, holders


class Limits:
    def __init__(self, reference):
        config=json.loads(Path(reference).read_text())
        self.b_min,self.b_max=config['operating_envelope']['B_command_mm']
        self.c=config['operating_envelope']['C_trial_hold_mm']
        self.a_min,self.a_max=config['operating_envelope'].get('A_command_degrees',[70,100])
        self.sender=config['control_policy']['original_sender_module_on_rk3588']
        self.startup_policy=config.get('power_cycle_startup',{})
        self.c_bounds=config['operating_envelope'].get('C_command_mm',[0,30])
        if not (all(isinstance(v,(int,float)) and math.isfinite(v) and v==round(v)
                    for v in (self.a_min,self.a_max))
                and 0<=self.a_min<self.a_max<=180
                and 0<=self.b_min<=self.b_max<=96 and 0<=self.c<=30):
            raise RuntimeError('invalid_arm_reference')

    def check(self,q):
        if len(q)!=3 or not all(math.isfinite(float(v)) for v in q):
            raise RuntimeError('invalid_pose')
        if not (self.a_min<=q[0]<=self.a_max and self.b_min<=q[1]<=self.b_max and q[2]==self.c):
            raise RuntimeError('pose_outside_working_range')


def pose(s):
    return [s['servo_software_deg'],s['lift_position_mm'],s['slide_position_mm']]


def objective(error):
    return (error[0]/20)**2+(error[1]/15)**2


def should_interrupt(current, target, error, origin=None):
    """Replan when the remaining motion no longer helps the latest image.

    Error growth alone is not a fault: the vehicle/target may be moving.
    Use only the remaining commanded displacement, never a stale target error.
    """
    # The original protocol accepts whole mm. Finish a bounded sub-mm
    # normalization after an interrupted move before image-space replanning;
    # otherwise a large image error can cancel it before the first pulse.
    if origin is not None and target[0]==origin[0] and 0<abs(target[1]-origin[1])<1:
        return False
    remaining=[target[k]-current[k] for k in range(2)]
    axis=0 if origin is not None and target[0]!=origin[0] else (0 if abs(remaining[0])>.01 else 1)
    if axis==1 and abs(error[0])>35:return True
    if abs(remaining[axis])<.01:return False
    tolerance,gain=(20,20.) if axis==0 else (15,-1.)
    if abs(error[axis])<=tolerance:return True
    predicted=error[axis]+gain*remaining[axis]
    return abs(predicted)>abs(error[axis])+2


def plan(q,error,limits):
    """Horizontal priority; each axis corrects its own error within its limits.

    A cannot be used to compensate for unreachable vertical error. B is
    interruptible when horizontal tracking needs it to release the serial owner.
    """
    limits.check(q)
    if abs(error[0])<=20 and abs(error[1])<=15:return None
    if abs(q[1]-round(q[1]))>.01:
        normalized=[int(q[0]),round(q[1]),int(q[2])]
        limits.check(normalized);return normalized
    for axis,tolerance,gain,cap,low,high in ((0,20,20.,4,limits.a_min,limits.a_max),
                                          (1,15,-1.,10,limits.b_min,limits.b_max)):
        if abs(error[axis])<=tolerance:continue
        delta=max(-cap,min(cap,round(-error[axis]/gain)))
        target=max(low,min(high,q[axis]+delta))
        if abs(target-q[axis])<.01:continue
        new=[int(v) for v in q];new[axis]=int(target);limits.check(new)
        return new
    return None


class ArmPort:
    def __init__(self,limits,log,port=DEFAULT_PORT,sender_factory=None,check_holders=True,defer_preflight=False):
        self.limits,self.log,self.port=limits,log,port
        self.fd=None;self.parser=Parser();self.state=None;self.rx_at=0.
        self.phase='idle';self.expected=None;self.target=None;self.origin=None
        self.gateway=None;self.attempted=False;self.deadline=0.;self.ready_at=0.;self.arrived_at=None
        if check_holders:holders(port,allow_partial=True)
        try:
            self.fd=os.open(port,os.O_RDWR|os.O_NOCTTY|os.O_NONBLOCK)
            fcntl.ioctl(self.fd,termios.TIOCEXCL)
            a=termios.tcgetattr(self.fd);a[0]=a[1]=a[3]=0
            a[2]=termios.CS8|termios.CREAD|termios.CLOCAL;a[4]=a[5]=termios.B115200
            a[6][termios.VMIN]=a[6][termios.VTIME]=0
            termios.tcsetattr(self.fd,termios.TCSANOW,a)
            if not defer_preflight:
                # Discard buffered history; require fresh periodic reports thereafter.
                until=time.monotonic()+.4
                while time.monotonic()<until:self.poll();time.sleep(.01)
                frames=[];until=time.monotonic()+1.2
                while time.monotonic()<until:
                    frames.extend(self.poll());time.sleep(.01)
                self.validate_preflight(frames)
            if sender_factory is None:
                module=Path(self.limits.sender)
                if not module.is_file():raise RuntimeError('original_sender_missing')
                sys.path.insert(0,str(module.parents[2]))
                sender_factory=importlib.import_module('app.infrastructure.serial_port').SerialPortGateway
            owner=self
            class ExistingFD:
                def write(self,data):
                    n=os.write(owner.fd,data)
                    owner.log('serial_tx',{'hex':data.hex(' '),'host_bytes':n})
                    if n!=len(data):raise RuntimeError('partial_write_no_retry')
                    return n
            self.gateway=sender_factory(port=port,serial_factory=lambda port,baud:ExistingFD(),command_repeats=1)
            self.gateway._open_serial()
            self.log('arm_serial_open_readonly' if defer_preflight else 'arm_ready',self.state)
        except BaseException:
            if self.fd is not None:os.close(self.fd);self.fd=None
            raise

    def validate_preflight(self,frames):
        if len(frames)<3:raise RuntimeError('no_fresh_telemetry')
        start=pose(frames[-1]);counter=frames[-1]['command_count_mod256']
        for s in frames:
            self.preflight_state(s)
            if (not self.preflight_disabled(s) or pose(s)!=start or s['command_count_mod256']!=counter
                    or s.get('uart_error_count_mod256')!=frames[-1].get('uart_error_count_mod256')):
                raise RuntimeError('arm_not_stopped_or_another_controller')
        self.preflight_pose(start);self.expected=counter

    def preflight_state(self,s):self.check_state(s)

    def preflight_disabled(self,s):return not s['enabled'] and s['state']==0

    def preflight_pose(self,q):self.limits.check(q)

    @staticmethod
    def check_state(s):
        if s['system_fault'] or s['motor_fault']:raise RuntimeError('firmware_fault')
        if not all(s[k] for k in ('lift_zero_confirmed','slide_zero_confirmed','angle_valid','heading_reference_established')):
            raise RuntimeError('origins_or_heading_not_ready')
        if s['yaw_deg'] is None or not math.isfinite(s['yaw_deg']):raise RuntimeError('heading_invalid')

    def poll(self):
        frames=[]
        while self.fd is not None and select.select([self.fd],[],[],0)[0]:
            data=os.read(self.fd,4096)
            if not data:raise RuntimeError('serial_disconnected')
            decoded=self.parser.feed(data);frames.extend(decoded)
            self.log('serial_observation',{'frames':decoded,'parser':dict(self.parser.stats)})
        if frames:
            self.state=frames[-1];self.rx_at=time.monotonic()
            self.log('serial_rx',frames)
        return frames

    def frame(self,q,enable):
        yaw=self.state['yaw_deg']
        heading=round((yaw or 0)/2)%180
        return bytes([0xB3,0,0,heading,*map(int,q),0,int(enable),0xB4])

    def move(self,q):
        if self.phase!='idle' or time.monotonic()<self.ready_at:raise RuntimeError('arm_busy')
        self.check_state(self.state);self.limits.check(q)
        if time.monotonic()-self.rx_at>.7:raise RuntimeError('stale_telemetry')
        start=pose(self.state);self.limits.check(start)
        delta=[abs(q[k]-start[k]) for k in range(3)]
        if sum(d>.01 for d in delta)!=1 or delta[0]>4 or delta[1]>20 or delta[2]!=0:
            raise RuntimeError('invalid_tracking_step')
        self.origin,self.target=start,q;self.arrived_at=None
        self.move_counter=(self.expected+1)%256
        self.phase='moving';self.attempted=True;self.started=time.monotonic()
        self.deadline=self.started+6+delta[1]/5
        if not self.gateway.write_control_frame(self.frame(q,True),repeats=1):
            raise RuntimeError('move_write_failed')

    def stop(self):
        if self.state is None and not self.attempted:return
        if self.phase=='stopping':return
        if not self.attempted and not self.state['enabled']:
            return
        # Preserve the issued absolute targets; disabled frames hold PWM and stop pulses.
        q=self.target or [round(v) for v in pose(self.state)]
        self.stop_counter=(self.move_counter+1)%256 if self.attempted else (self.expected+1)%256
        self.phase='stopping';self.deadline=time.monotonic()+1.5
        if not self.gateway.write_control_frame(self.frame(q,False),repeats=1):
            raise RuntimeError('STOP_UNCONFIRMED_write_failed')

    def tick(self):
        frames=self.poll();now=time.monotonic()
        if self.phase=='stopping':
            for s in frames:
                # If the motion frame was rejected at the UART level, disable can be +1.
                if s['command_count_mod256'] in (self.stop_counter,(self.stop_counter-1)%256) and s['result']==0 and not s['enabled'] and s['state']==0:
                    self.expected=s['command_count_mod256'];self.phase='idle';self.attempted=False
                    self.target=None;self.ready_at=now+.1
                    self.log('disabled_confirmed',s)
                    return
            if now>self.deadline:raise RuntimeError('STOP_UNCONFIRMED_no_feedback')
            return
        if now-self.rx_at>.7:raise RuntimeError('telemetry_timeout')
        for s in frames:
            self.check_state(s)
            if self.phase=='idle':
                if s['command_count_mod256']!=self.expected or s['enabled']:
                    raise RuntimeError('unexpected_external_control')
                self.preflight_pose(pose(s))
            else:
                if s['command_count_mod256'] not in (self.expected,self.move_counter):
                    raise RuntimeError('unexpected_command_counter')
                if s['command_count_mod256']==self.move_counter:
                    if s['result']!=0:raise RuntimeError('firmware_rejected_command')
                    q=pose(s)
                    if any(not min(a,b)-.01<=v<=max(a,b)+.01 for v,a,b in zip(q,self.origin,self.target)):
                        raise RuntimeError('unexpected_mechanism_motion')
                    if s['enabled'] and q==self.target:
                        if self.arrived_at is None:self.arrived_at=now
                        if now-self.arrived_at>=.1 and now-self.started>=.5:
                            self.stop();return
                    else:self.arrived_at=None
        if self.phase=='moving' and now>self.deadline:raise RuntimeError('movement_timeout')

    def shutdown(self):
        error=None
        try:
            if self.gateway is not None:
                self.stop()
                until=time.monotonic()+1.7
                while self.phase=='stopping' and time.monotonic()<until:
                    self.tick();time.sleep(.01)
                if self.phase=='stopping':raise RuntimeError('STOP_UNCONFIRMED_shutdown')
        except BaseException as exc:
            error=str(exc);self.log('stop_unconfirmed',{'reason':error})
        finally:
            if self.fd is not None:os.close(self.fd);self.fd=None
        return error


class ContinuousArmPort(ArmPort):
    """Bounded target stream with cumulative acknowledgments, no A stop/start cycles.

    Original firmware applies Servo_SetTenths on every enabled packet, including
    while a finite B move is running. Preserve that B goal in A-only updates.
    """
    def __init__(self, *args, **kwargs):
        self.history = deque(maxlen=40)
        super().__init__(*args, **kwargs)
        self.pending_counter = None; self.pending = []; self.last_tx = 0.
        self.b_origin = self.state['lift_position_mm']; self.b_started = 0.

    def poll(self):
        frames = super().poll()
        if frames:
            self.history.append((self.rx_at, pose(frames[-1])[0]))
        return frames

    def angle_at(self, stamp):
        entries = list(self.history)
        if not entries:
            return pose(self.state)[0]
        for (t0,a0),(t1,a1) in zip(entries, entries[1:]):
            if t0 <= stamp <= t1 and t1 > t0:
                return a0+(a1-a0)*(stamp-t0)/(t1-t0)
        return entries[0][1] if stamp < entries[0][0] else entries[-1][1]

    def can_send(self):
        return (self.phase in ('idle','following') and len(self.pending)<3
                and time.monotonic()-self.last_tx >= .04 and time.monotonic() >= self.ready_at)

    def follow(self, q):
        if not self.can_send():
            return False
        self.check_state(self.state); self.limits.check(q)
        if time.monotonic()-self.rx_at > .35:
            raise RuntimeError('stale_telemetry')
        current = pose(self.state); self.limits.check(current)
        old = self.target or [round(v) for v in current]
        changes = [abs(q[k]-old[k]) for k in range(3)]
        if sum(d>.01 for d in changes)>1 or changes[0]>4 or changes[1]>8 or changes[2]!=0:
            raise RuntimeError('invalid_continuous_target')
        if changes[1] and abs(current[1]-old[1])>.11:
            raise RuntimeError('lift_busy_preserve_goal')
        if self.target == q and self.state['enabled']:
            return False
        if changes[1] or self.target is None:
            self.b_origin=current[1];self.b_started=time.monotonic()
        self.origin=current; self.target=list(q)
        self.move_counter=((self.move_counter if self.attempted else self.expected)+1)%256
        self.pending_counter=self.move_counter;self.pending.append((self.move_counter,time.monotonic()))
        self.phase='following';self.attempted=True;self.last_tx=time.monotonic()
        self.deadline=self.last_tx+.6
        if not self.gateway.write_control_frame(self.frame(q,True),repeats=1):
            raise RuntimeError('move_write_failed')
        return True

    def tick(self):
        if self.phase=='stopping':
            frames=self.poll();now=time.monotonic()
            allowed={(self.expected+i)%256 for i in range(1,len(self.pending)+2)}
            for s in frames:
                if s['command_count_mod256'] in allowed and s['result']==0 and not s['enabled'] and s['state']==0:
                    self.expected=s['command_count_mod256'];self.phase='idle';self.attempted=False
                    self.target=None;self.ready_at=now+.04;self.pending=[];self.pending_counter=None
                    self.log('disabled_confirmed',s);return
            if now>self.deadline:raise RuntimeError('STOP_UNCONFIRMED_no_feedback')
            return
        if self.phase=='idle':
            super().tick();return
        frames=self.poll();now=time.monotonic()
        if now-self.rx_at>.7:
            raise RuntimeError('telemetry_timeout')
        for s in frames:
            self.check_state(s);self.limits.check(pose(s))
            counter=s['command_count_mod256']
            pending_counts=[item[0] for item in self.pending]
            if counter!=self.expected and counter not in pending_counts:
                raise RuntimeError('unexpected_command_counter')
            if counter in pending_counts:
                if s['result']!=0 or not s['enabled']:
                    raise RuntimeError('firmware_rejected_command')
                self.expected=counter;self.pending=self.pending[pending_counts.index(counter)+1:]
                self.pending_counter=self.pending[-1][0] if self.pending else None
            elif not s['enabled']:
                # The disabled report preceding the first command may be in flight.
                if self.pending_counter is None:
                    raise RuntimeError('unexpected_external_control')
            b=s['lift_position_mm']
            if not min(self.b_origin,self.target[1])-.11 <= b <= max(self.b_origin,self.target[1])+.11:
                raise RuntimeError('unexpected_mechanism_motion')
        if self.pending and now-self.pending[0][1]>.6:
            raise RuntimeError('command_ack_timeout')
        if abs(self.state['lift_position_mm']-self.target[1])>.11 and now-self.b_started>8:
            raise RuntimeError('movement_timeout')
