"""Select the board viewer backend, then replace this short-lived process."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--backend', choices=['cpp', 'python'], default='cpp')
parser.add_argument('--mode', choices=['async', 'sync'], default='async')
args, rest = parser.parse_known_args()
os.chdir(ROOT)
if args.backend == 'python':
    os.execv(sys.executable, [sys.executable, '-u', 'tools/live_viewer.py', '--mode', args.mode, *rest])
if args.mode != 'async':
    parser.error('C++ uses async mode; use --backend python --mode sync')
entry = ROOT.parent / 'vision_cpp' / 'run.sh'
if not entry.is_file():
    raise SystemExit('C++ backend is not deployed. Run deploy.sh, or use --backend python.')
os.execv(str(entry), [str(entry), *rest])
