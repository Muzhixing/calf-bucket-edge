#!/usr/bin/env python3
"""Read F407 wheel timer/GPIO registers through a localhost OpenOCD Tcl port.

No target writes, resets, halts or motion commands. Samples are sequential,
not atomic; each record includes the complete host read interval.
"""
import argparse
import json
from pathlib import Path
import socket
import time

# STM32F407 CMSIS register definitions, with wheel wiring from firmware 785123b.
REGISTERS = {
    'encoder_lf': 0x40000024, 'encoder_lb': 0x40000424,
    'encoder_rf': 0x40000824, 'encoder_rb': 0x40010424,
    'pwm_lf': 0x40014038, 'pwm_lb': 0x40010038,
    'pwm_rf': 0x4001003C, 'pwm_rb': 0x40010040,
    'gpiob_odr': 0x40020414, 'gpiod_odr': 0x40020C14,
    # Read the counters again after the PWM/GPIO reads. This brackets
    # short-term changes and exposes isolated debug-read artifacts.
    'encoder_lf_repeat': 0x40000024, 'encoder_lb_repeat': 0x40000424,
    'encoder_rf_repeat': 0x40000824, 'encoder_rb_repeat': 0x40010424,
}


def rpc(sock, command):
    sock.sendall(command.encode('ascii') + b'\x1a')
    data = bytearray()
    while b'\x1a' not in data:
        part = sock.recv(4096)
        if not part:
            raise RuntimeError('debug_connection_closed')
        data.extend(part)
        if len(data) > 65536:
            raise RuntimeError('oversized_debug_response')
    return bytes(data).split(b'\x1a')[0].decode('ascii')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=5)
    ap.add_argument('--hz', type=float, default=25)
    ap.add_argument('--port', type=int, default=6667)
    ap.add_argument('--require-stopped', action='store_true',
                    help='Abort observation if any wheel PWM register is nonzero')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if not 0 < a.seconds <= 120 or not 1 <= a.hz <= 50:
        ap.error('seconds must be 0..120 and hz 1..50')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'manifest.json').write_text(json.dumps({
        'registers': {k: hex(v) for k,v in REGISTERS.items()},
        'requested_hz': a.hz, 'seconds': a.seconds,
        'require_stopped': a.require_stopped,
        'wiring_basis': '785123b source; confirm using measured response',
        'atomic': False, 'target_write_commands': 0,
    }, indent=2)+'\n')
    command = '; '.join('mem2array r%d 32 0x%08x 1' % (i,addr)
                        for i,addr in enumerate(REGISTERS.values()))
    command += '; list ' + ' '.join('$r%d(0)' % i for i in range(len(REGISTERS)))
    count = 0
    with socket.create_connection(('127.0.0.1',a.port),timeout=2) as sock:
        if rpc(sock,'stm32f4x.cpu curstate').strip() != 'running':
            raise RuntimeError('target_not_running')
        start = time.monotonic()
        with (a.out/'registers.jsonl').open('x',buffering=1) as f:
            while time.monotonic()-start < a.seconds:
                before = time.monotonic()
                values = [int(v,0) for v in rpc(sock,command).split()]
                after = time.monotonic()
                if len(values) != len(REGISTERS):
                    raise RuntimeError('unexpected_register_count')
                row = dict(zip(REGISTERS,values))
                row.update(start_monotonic=before,end_monotonic=after,utc_epoch=time.time())
                f.write(json.dumps(row)+'\n'); count += 1
                if a.require_stopped and any(row['pwm_'+wheel] for wheel in ('lf','lb','rf','rb')):
                    raise RuntimeError('nonzero_pwm_during_manual_observation')
                time.sleep(max(0,1/a.hz-(time.monotonic()-before)))
        final = rpc(sock,'stm32f4x.cpu curstate').strip()
    print(json.dumps({'samples':count,'elapsed_s':time.monotonic()-start,'final_cpu_state':final}))
    if final != 'running':
        raise RuntimeError('target_not_running_at_end')


if __name__ == '__main__':
    main()
