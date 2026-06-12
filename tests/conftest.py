from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "source" / "isaac_fight"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

try:
    import gymnasium  # noqa: F401
except ModuleNotFoundError:
    class _Env:
        @property
        def unwrapped(self):
            return self

    class _Wrapper(_Env):
        def __init__(self, env):
            self.env = env

        @property
        def unwrapped(self):
            return self.env.unwrapped

    registry = {}

    def _register(id, **kwargs):  # noqa: ANN001
        registry[id] = kwargs

    sys.modules["gymnasium"] = types.SimpleNamespace(Env=_Env, Wrapper=_Wrapper, registry=registry, register=_register)
