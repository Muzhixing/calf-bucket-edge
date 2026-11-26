"""Compatibility entry; implementation lives in the installed bucket-robot package."""
import importlib
import sys
_module = importlib.import_module('bucket_robot.motion.chassis')
if __name__ == "__main__":
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'tools/field/chassis_probe.py'),run_name='__main__')
sys.modules[__name__] = _module
