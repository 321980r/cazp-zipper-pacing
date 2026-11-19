"""
Closure configurations of the paper (Fig. 1c) and their SUMO networks. Lane 0 is the shoulder-side lane and lane 3 the
median-side lane; every configuration has four upstream lanes and a 1-km downstream section.

    seq    configuration I, main test bed: successive closure with the spacing of MUTCD Typical Application 37. Lane 3
           ends at x = 4325 m and lane 2 at x = 5000 m. Taper L = W S / 1.6 = 3.6 m x 100 km/h / 1.6 = 225 m; the
           second taper begins 2L = 450 m after the end of the first one, so the lane ends are 675 m apart.
    base   configuration II: lanes 2 and 3 end together at x = 5000 m.
    l43    configuration III: four-to-three closure, lane 3 ends at x = 5000 m.

A closing lane is coupled to the lane into which it merges (ref) and receives lateral move-ins from lat_from; the other
lanes are open lanes. The internal codes seq, base and l43 are part of the stored run keys.

    python closures.py     rebuilds the three networks in a temporary folder and compares them with networks/
"""
from __future__ import annotations
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
NETWORKS = os.path.join(HERE, "networks")
V_MS = 100.0 / 3.6


def netconvert() -> str:
    exe = "netconvert" + (".exe" if platform.system() == "Windows" else "")
    home = os.environ.get("SUMO_HOME")
    if home and os.path.isfile(os.path.join(home, "bin", exe)):
        return os.path.join(home, "bin", exe)
    found = shutil.which("netconvert")
    if found:
        return found
    raise RuntimeError("netconvert not found: set SUMO_HOME to the SUMO installation folder")


@dataclass(frozen=True)
class Closure:
    name: str
    net: str                         # network folder
    n_down: int                      # lanes of the final (downstream) section
    chain: tuple                     # mainline ((edge, junction at its end), ...), "" for the last edge
    ends: tuple                      # ((closing lane, nominal x of its lane end), ...)
    ref: tuple                       # ((closing lane, lane into which it merges), ...)
    lat_from: tuple                  # ((closing lane, lane supplying lateral move-ins), ...)
    n_up: int = 4
    x_drop: float = 5000.0           # final lane end, where the final bottleneck begins
    L_down: float = 1000.0

    @property
    def closing(self) -> tuple:
        return tuple(sorted(dict(self.ends)))

    @property
    def open(self) -> tuple:
        return tuple(ln for ln in range(self.n_up) if ln not in dict(self.ends))

    @property
    def same_end(self) -> bool:
        """All closing lanes end at the same point (the alignment rule applies only then)."""
        return len(set(dict(self.ends).values())) == 1


SEQ = Closure("seq", os.path.join(NETWORKS, "seq"), 2, (("E_up", "m1"), ("E_mid", "n1"), ("E_down", "")),
              ends=((2, 5000.0), (3, 4325.0)), ref=((2, 1), (3, 2)), lat_from=((2, 1), (3, 2)))
BASE = Closure("base", os.path.join(NETWORKS, "base"), 2, (("E_up", "n1"), ("E_down", "")),
               ends=((2, 5000.0), (3, 5000.0)), ref=((2, 1), (3, 1)), lat_from=((2, 1), (3, 2)))
L43 = Closure("l43", os.path.join(NETWORKS, "l43"), 3, (("E_up", "n1"), ("E_down", "")),
              ends=((3, 5000.0),), ref=((3, 2),), lat_from=((3, 2),))
CLOSURES = {c.name: c for c in (SEQ, BASE, L43)}
NUMBER = {"seq": "I", "base": "II", "l43": "III"}


def plain_xml(c: Closure) -> dict:
    """Node, edge and connection files: one node per lane end, lanes continue straight on."""
    xs = sorted(set(dict(c.ends).values()))
    nodes = [("n0", 0.0)] + [(node, x) for (_, node), x in zip(c.chain[:-1], xs)] + [("n2", c.x_drop + c.L_down)]
    lanes = [c.n_up - sum(1 for x in dict(c.ends).values() if x <= xa) for _, xa in nodes[:-1]]
    nod = "<nodes>\n" + "".join(f'<node id="{n}" x="{x:.1f}" y="0.0" type="priority"/>\n' for n, x in nodes) + \
          "</nodes>\n"
    edg = "<edges>\n" + "".join(
        f'<edge id="{e}" from="{nodes[j][0]}" to="{nodes[j + 1][0]}" numLanes="{lanes[j]}" speed="{V_MS:.4f}"/>\n'
        for j, (e, _) in enumerate(c.chain)) + "</edges>\n"
    con = "<connections>\n" + "".join(
        f'<connection from="{e}" to="{c.chain[j + 1][0]}" fromLane="{i}" toLane="{i}"/>\n'
        for j, (e, _) in enumerate(c.chain[:-1]) for i in range(lanes[j + 1])) + "</connections>\n"
    return {"nod": nod, "edg": edg, "con": con}


def net_file(c: Closure, folder: str | None = None) -> str:
    return os.path.join(folder or c.net, f"lanedrop_{c.n_up}to{c.n_down}_x{c.x_drop:.0f}_L{c.L_down:.0f}_v100.net.xml")


def build(c: Closure, folder: str | None = None) -> str:
    folder = folder or c.net
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, "lanedrop")
    for ext, txt in plain_xml(c).items():
        with open(f"{base}.{ext}.xml", "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
    out = net_file(c, folder)
    r = subprocess.run([netconvert(), "--node-files", "lanedrop.nod.xml", "--edge-files", "lanedrop.edg.xml",
                        "--connection-files", "lanedrop.con.xml", "--output-file", os.path.basename(out),
                        "--no-turnarounds", "true", "--offset.disable-normalization", "true"],
                       cwd=folder, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError(r.stderr or "netconvert produced no output")
    return out


def body(path: str) -> str:
    """Network file without the header comment (it holds the netconvert options and the build date)."""
    return re.sub(r"<!--.*?-->", "", open(path, encoding="utf-8").read(), flags=re.S).strip()


if __name__ == "__main__":
    import tempfile
    tmp = tempfile.mkdtemp(prefix="closures_")
    for c in CLOSURES.values():
        same = body(build(c, os.path.join(tmp, c.name))) == body(net_file(c))
        print(f"configuration {NUMBER[c.name]} ({c.name}): rebuilt network identical to networks/{c.name}: {same}")
    shutil.rmtree(tmp, ignore_errors=True)
