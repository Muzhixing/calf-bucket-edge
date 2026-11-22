"""Open an SSH tunnel and the current native RK3588 viewer."""
import argparse
import datetime
import shlex
import subprocess
import threading
import webbrowser
from entry import SOCK, HOST, REMOTE


def main():
    parser = argparse.ArgumentParser(description='C++ fixed-camera preview')
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--port', type=int, default=8788)
    parser.add_argument('--id')
    parser.add_argument('--cores', choices=['auto', '0_1_2'], default='auto')
    parser.add_argument('--depth-scale', type=float, choices=[1.0, 0.5], default=0.5)
    parser.add_argument('--observe-only', action='store_true', help='禁用机构追踪入口，仅观察')
    args = parser.parse_args()
    ident = args.id or 'live_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    command = ['../vision_cpp/run.sh', '--id', ident, '--seconds', str(args.seconds),
               '--port', str(args.port), '--cores', args.cores, '--depth-scale', str(args.depth_scale)]
    if args.observe_only:
        command.append('--observe-only')
    if subprocess.run(['ssh', '-S', SOCK, '-O', 'check', HOST], capture_output=True).returncode:
        subprocess.run(['ssh', '-M', '-S', SOCK, '-o', 'ControlPersist=2h', '-Nf', HOST], check=True)
    url = f'http://127.0.0.1:{args.port}/'
    print('C++ preview: ' + url + ' ; Ctrl+C stops and releases the camera', flush=True)
    threading.Timer(3, lambda: webbrowser.open(url)).start()
    subprocess.run(['ssh', '-tt', '-S', SOCK, '-o', 'ExitOnForwardFailure=yes',
                    '-L', f'{args.port}:127.0.0.1:{args.port}', HOST,
                    f'cd {shlex.quote(REMOTE)} && ' + shlex.join(command)], check=True)


if __name__ == '__main__':
    main()
