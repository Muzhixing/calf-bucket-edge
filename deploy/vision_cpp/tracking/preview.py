"""Compatibility entry; implementation lives in the installed bucket-robot package."""
import importlib
import sys
_module = importlib.import_module('bucket_robot.runtime.preview')
sys.modules[__name__] = _module
