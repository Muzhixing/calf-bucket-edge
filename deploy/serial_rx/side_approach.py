"""Compatibility entry; implementation lives in the installed bucket-robot package."""
import importlib
import sys
_module = importlib.import_module('bucket_robot.missions.approach')
if __name__ == "__main__":
    raise SystemExit(_module.main())
sys.modules[__name__] = _module
