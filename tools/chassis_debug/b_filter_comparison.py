#!/usr/bin/env python3
"""Reversible TIM3 input-filter experiment, with stopped-only guarded writes.

Only IC1F/IC2F in TIM3_CCMR1 may change. No flash, timer counter, polarity,
prescaler, PWM, or PID writes. Every trial uses the original UART motion path.
"""
import argparse
import json
import re
from pathlib import Path
import socket
import subprocess
import time

from sample_registers import rpc

CCMR1 = 0x40000418
FILTER_MASK = 0xF0F0
PWM_ADDRS = (0x40014038, 0x40010038, 0x4001003C, 0x40010040)
ROOT = Path('/home/elf/deploy/chassis_debug')
SERIAL = Path('/home/elf/deploy/serial_rx')


def filtered_value(original, setting):
    if setting not in (6, 15) or original not in (0x6161, 0xF1F1):
        raise RuntimeError('unexpected_filter_configuration')
    return (original & ~FILTER_MASK) | (setting << 4) | (setting << 12)


def read(sock, address):
    return int(rpc(sock, 'mem2array v 32 0x%x 1; set v(0)' % address), 0)


def write_filter(sock, expected, value, log):
    if value not in (0x6161,0xF1F1) or (expected ^ value) & ~FILTER_MASK:
        raise RuntimeError('non_filter_bits_changed')
    if rpc(sock,'stm32f4x.cpu curstate').strip() != 'running':
        raise RuntimeError('cpu_not_running')
    pwm = [read(sock,a) for a in PWM_ADDRS]
    if any(pwm):
        raise RuntimeError('refuse_filter_write_while_pwm_active')
    before = read(sock,CCMR1)
    if before != expected:
        raise RuntimeError('filter_changed_externally')
    log('filter_write_intent',{'address':hex(CCMR1),'before':hex(before),'after':hex(value),'pwm':pwm})
    rpc(sock,'mww 0x%x 0x%x' % (CCMR1,value))
    actual = read(sock,CCMR1)
    log('filter_readback',{'value':hex(actual)})
    if actual != value:
        raise RuntimeError('filter_readback_mismatch')


def trial(label, log):
    output = ROOT/'runs'/label
    serial = SERIAL/'runs'/label
    sampler = subprocess.Popen(['python3','-u',str(ROOT/'sample_registers.py'),
                                '--seconds','14','--out',str(output)])
    motor = None
    try:
        time.sleep(1)
        if sampler.poll() is not None:
            raise RuntimeError('sampler_failed_before_motion')
        motor = subprocess.Popen(['python3','-u',str(SERIAL/'chassis_probe.py'),
                                  '--direction','backward','--speed','10',
                                  '--continuous','--out',str(serial)],
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        deadline = time.monotonic()+10
        while motor.poll() is None and time.monotonic()<deadline:
            if sampler.poll() is not None:
                raise RuntimeError('sampler_failed_during_motion')
            time.sleep(.05)
    finally:
        if motor is not None:
            if serial.is_dir(): (serial/'STOP').touch()
            try: text = motor.communicate(timeout=4)[0]
            except subprocess.TimeoutExpired:
                motor.terminate(); text = motor.communicate(timeout=4)[0]
            log('motor_process',{'label':label,'exit_code':motor.returncode,'output':text})
        sampler.wait(timeout=16)
    if motor is None or motor.returncode or sampler.returncode:
        raise RuntimeError('trial_failed_no_next_motion')
    state = json.loads((serial/'summary.json').read_text())
    final = state['final_state']
    if final['enabled'] or final['state'] != 0 or final['system_fault'] or final['motor_fault']:
        raise RuntimeError('trial_not_disabled_and_healthy')
    log('trial_complete',{'label':label,'final_counter':final['command_count_mod256']})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',required=True)
    args = ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):ap.error('invalid run id')
    run = ROOT/('runs/b_filter_comparison_'+args.run_id)
    run.mkdir(parents=True,exist_ok=False)
    with (run/'events.jsonl').open('x',buffering=1) as f:
        def log(kind,data):
            f.write(json.dumps({'utc_epoch':time.time(),'monotonic':time.monotonic(),
                                'kind':kind,'data':data})+'\n')
            if kind in ('trial_complete','filter_readback','error','restored'):
                print(kind,json.dumps(data),flush=True)
        with socket.create_connection(('127.0.0.1',6667),timeout=2) as sock:
            original = read(sock,CCMR1)
            if original != 0x6161:
                raise RuntimeError('original_filter_not_expected')
            changed = False
            result = {'original':hex(original),'error':None,'restore_error':None}
            try:
                trial('b_filter_baseline6_'+args.run_id,log)
                changed = True  # restoration required even if write/readback raises
                write_filter(sock,original,filtered_value(original,15),log)
                time.sleep(.3)
                trial('b_filter_candidate15_'+args.run_id,log)
                write_filter(sock,0xF1F1,original,log)
                changed = False
                time.sleep(.3)
                trial('b_filter_return6_'+args.run_id,log)
            except BaseException as exc:
                result['error'] = repr(exc); log('error',{'reason':repr(exc)})
            finally:
                if changed:
                    try:
                        actual = read(sock,CCMR1)
                        write_filter(sock,actual,original,log)
                        log('restored',{'value':hex(original)})
                    except BaseException as exc:
                        result['restore_error'] = repr(exc)
                result['final_filter'] = hex(read(sock,CCMR1))
                result['final_pwm'] = [read(sock,a) for a in PWM_ADDRS]
                (run/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
            return 0 if not result['error'] and not result['restore_error'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
