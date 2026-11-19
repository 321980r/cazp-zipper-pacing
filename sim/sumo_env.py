"""
SUMO environment detection and import.

Priority for locating SUMO:
  1. USER_SUMO_HOME in config.py (None by default)
  2. environment variable SUMO_HOME
  3. the pip package `eclipse-sumo` (pip install eclipse-sumo==1.18.0 libsumo==1.18.0 traci==1.18.0 sumolib==1.18.0)
  4. common installation folders on Windows / Linux / macOS

The simulation kernel prefers `libsumo` (fast, in-process). If libsumo is not
available, it falls back to `traci` (socket based, slower but identical API).
With the GUI (core.simulation.run(..., gui=True)), `traci` and `sumo-gui` are used.
"""
from __future__ import annotations
import os
import sys
import shutil
import platform

_COMMON = [
    r"C:\Program Files (x86)\Eclipse\Sumo",
    r"C:\Program Files\Eclipse\Sumo",
    r"D:\Program Files (x86)\Eclipse\Sumo",
    r"D:\Program Files\Eclipse\Sumo",
    r"C:\Sumo", r"D:\Sumo",
    "/usr/share/sumo", "/usr/local/share/sumo",
    "/opt/homebrew/share/sumo", "/usr/local/opt/sumo/share/sumo",
]


def _exe(name: str) -> str:
    return name + (".exe" if platform.system() == "Windows" else "")


def find_sumo_home(user_home: str | None = None) -> str | None:
    cands = []
    if user_home:
        cands.append(user_home)
    if os.environ.get("SUMO_HOME"):
        cands.append(os.environ["SUMO_HOME"])
    try:  # pip package eclipse-sumo
        import sumo  # type: ignore
        cands.append(sumo.SUMO_HOME)
    except Exception:
        pass
    cands += _COMMON
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "bin", _exe("sumo"))):
            return os.path.abspath(c)
    return None


def setup(user_home: str | None = None, prefer_local_tools: bool = False) -> str:
    home = find_sumo_home(user_home)
    if home is None:
        raise RuntimeError(
            "SUMO not found. Either set USER_SUMO_HOME in config.py to your SUMO folder "
            "(the folder that contains bin/sumo), set the SUMO_HOME environment variable, "
            "or run: pip install eclipse-sumo==1.18.0 libsumo==1.18.0 traci==1.18.0 sumolib==1.18.0")
    os.environ["SUMO_HOME"] = home
    tools = os.path.join(home, "tools")
    if os.path.isdir(tools) and tools not in sys.path:
        if prefer_local_tools:
            sys.path.insert(0, tools)     # traci that matches the installed sumo-gui
        else:
            sys.path.append(tools)
    return home


def binary(name: str, user_home: str | None = None) -> str:
    home = setup(user_home)
    p = os.path.join(home, "bin", _exe(name))
    if os.path.isfile(p):
        return p
    w = shutil.which(name)
    if w:
        return w
    raise RuntimeError(f"Cannot find SUMO binary '{name}' under {home}")


def import_kernel(prefer_libsumo: bool = True):
    """Return (module, is_libsumo). Both expose the TraCI API."""
    if prefer_libsumo:
        try:
            import libsumo as kernel  # type: ignore
            return kernel, True
        except Exception:
            pass
    import traci as kernel  # type: ignore
    return kernel, False
