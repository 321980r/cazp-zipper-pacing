"""
Network lookup. The networks of the three closure configurations are built by closures.py (netconvert, from the
plain-XML files kept next to each network) and are only looked up here: a missing network is an error, never
rebuilt silently with a different lane configuration.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C          # noqa: E402


def net_name(geo) -> str:
    return os.path.join(C.NET_DIR, f"lanedrop_{geo.n_up}to{geo.n_down}_x{geo.x_drop:.0f}"
                                   f"_L{geo.L_down:.0f}_v{geo.v_base_kmh:.0f}.net.xml")


def ensure_network(geo) -> str:
    """Return the network path for this geometry (must exist in C.NET_DIR)."""
    p = net_name(geo)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"network {p} missing: build it with closures.py")
    return p
