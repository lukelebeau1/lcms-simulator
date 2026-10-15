"""Down at the C18 surface: one analyte molecule among bonded C18 chains, water and methanol. Illustration.

What is real: the analyte's 3D structure and its hydrophobic side (Crippen shares, app/molecules.py), and the
share of time it spends held in the C18 layer, k / (1 + k), with k from the retention model at the %B around the
band (lcms_sim.lc.column: band position and local composition through the run). The methanol / water count in the
scene is the mole fraction for that %B (volumes 40.7 and 18.05 mL/mol; mixing contraction ignored).

What is drawn freely: the chain conformations (a disordered brush, ~one chain per 50 A^2, tilted ~25 deg), the hop
timing (a two-state Markov chain with a fixed cycle: frequent short holds at moderate k, brief releases at high k;
the held share is k/(1+k) either way), the flow speed and the poses. Held, the molecule is partitioned into the
chain layer along its polar-to-oily axis: oily end among the chains, polar end at the chain tops facing the solvent. Real hops take nanoseconds to microseconds and happen millions of times per
run; the animation compresses time, and in "through the run" each frame is a representative moment.

Methanol also soaks into the top of the C18 layer (the stationary phase is chains plus sorbed organic solvent);
shown qualitatively by letting methanol, but not water, sit just inside the chain tops.
"""

from __future__ import annotations

import json
import math
from html import escape
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

import molecules

BOX_X, BOX_Y = 56.0, 28.0            # periodic patch of surface, A
C_STEP, ZIGZAG = 1.26, 0.42          # C-C rise along the chain axis, zig-zag offset
N_CHAIN_C = 18
TOP_Z = 1.6 + N_CHAIN_C * C_STEP * math.cos(math.radians(25))     # ~22 A: chain tops
CEIL_Z = 46.0
V_MEOH, V_H2O = 40.7, 18.05          # molar volumes, mL/mol


def methanol_mole_fraction(phi: float) -> float:
    """Methanol volume fraction -> mole fraction (no mixing contraction)."""
    phi = float(np.clip(phi, 0.0, 1.0))
    a, b = phi / V_MEOH, (1 - phi) / V_H2O
    return a / (a + b) if a + b > 0 else 0.0


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) or 1.0)
    x, y, z = axis
    c, s, C = math.cos(ang), math.sin(ang), 1 - math.cos(ang)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _align(u: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector u onto target."""
    u, target = u / np.linalg.norm(u), target / np.linalg.norm(target)
    axis = np.cross(u, target)
    s, c = np.linalg.norm(axis), float(np.dot(u, target))
    if s < 1e-9:
        return np.eye(3) if c > 0 else _rot(np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0]), math.pi)
    return _rot(axis, math.atan2(s, c))


def surface_xyz(rng: np.random.Generator) -> str:
    """Static part: silica surface (Si, O) and the grafted C18 chains (united-atom carbons)."""
    atoms: list[tuple[str, float, float, float]] = []
    for ix in range(int(BOX_X / 3.5)):
        for iy in range(int(BOX_Y / 3.5)):
            x, y = 1.75 + 3.5 * ix, 1.75 + 3.5 * iy
            atoms.append(("Si" if (ix + iy) % 2 == 0 else "O", x, y, 0.0))
    for ix in range(8):
        for iy in range(4):
            base = np.array([3.5 + 7.0 * ix + rng.normal(0, 0.6), 3.5 + 7.0 * iy + rng.normal(0, 0.6), 0.0])
            atoms.append(("Si", *base[:2], 0.9))
            th, ph = math.radians(rng.normal(25, 8)), rng.uniform(0, 2 * math.pi)
            d = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
            p = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.99 else [1.0, 0.0, 0.0])
            p /= np.linalg.norm(p)
            pos = base + np.array([0.0, 0.0, 2.4])
            for n in range(N_CHAIN_C):
                if n > 3 and rng.random() < 0.18:             # an occasional gauche kink: disorder near the top
                    d = _rot(p, rng.normal(0, 0.45)) @ d
                    d[2] = abs(d[2])
                    d /= np.linalg.norm(d)
                pos = pos + C_STEP * d
                q = pos + (ZIGZAG if n % 2 else -ZIGZAG) * p
                atoms.append(("C", q[0] % BOX_X, q[1] % BOX_Y, q[2]))
    return f"{len(atoms)}\nC18 surface\n" + "\n".join(f"{e} {x:.2f} {y:.2f} {z:.2f}" for e, x, y, z in atoms)


WATER = [("O", 0.0, 0.0, 0.0), ("H", 0.76, 0.59, 0.0), ("H", -0.76, 0.59, 0.0)]
METHANOL = [("C", -0.71, 0.0, 0.0), ("O", 0.71, 0.0, 0.0), ("H", 1.03, 0.9, 0.0), ("H", -1.07, 1.03, 0.0),
            ("H", -1.07, -0.51, 0.89), ("H", -1.07, -0.51, -0.89)]


@dataclass(frozen=True)
class Species:
    """One compound in the scene. Arrays are per frame."""
    mol: Chem.Mol
    name: str
    phi: np.ndarray            # %B (fraction) around this compound's own band
    k: np.ndarray              # its retention factor there
    presence: np.ndarray       # share of its molecules inside the window (1 = its band is here)
    ahead: np.ndarray          # +1 its band is downstream of the window, -1 upstream (where leavers go)
    note: list[str]            # readout fragment per frame (band position, separation)
    eluted: np.ndarray


@dataclass(frozen=True)
class Scene:
    surface: str               # xyz, static
    frames: list[str]          # xyz per frame: analyte copies first (in `present` order), then solvent
    present: list[list[int]]   # species index of each analyte copy drawn, in atom order
    n_analyte: list[int]       # analyte atoms per frame
    sp_colours: dict           # {"hyd": [[per atom] per species], "cmp": [...]}
    sp_radii: list[list[float]]
    tags: list[list[list]]     # per frame: [name, x, y, z] above each drawn copy
    info: list[str]
    held: list[bool]           # species 0, copy 0
    phi: list[float]
    k: list[float]
    colours: list[str]         # species 0, hydrophobicity (kept for single-compound callers)
    centers: list[list[list[float]]]   # per frame: [x, y, z] of each drawn copy, aligned with `present`
    names: list[str]           # species names, in species order
    extents: list[float]       # per species: farthest atom from its centroid, A


COMPOUND_HUES = ["#f2a33a", "#3fc7b0", "#c58af9"]
ELEMENT = {"N": "#4f74ff", "O": "#ff4d4d", "Cl": "#3ddc4a", "S": "#f2e14c", "P": "#ff9a3c", "F": "#9be36b",
           "Br": "#b5523b", "Na": "#ab5cf2"}


def amphiphilic_axis(m: Chem.Mol) -> np.ndarray:
    """Unit vector from the molecule's polar end to its oily end (Crippen shares as weights).

    Polar centroid: atoms with negative shares; oily centroid: atoms with positive shares. Using both ends, not
    the single most hydrophobic spot, keeps one strongly hydrophobic atom (an aryl Cl) from setting the pose.
    """
    xyz = m.GetConformer().GetPositions()
    xyz = xyz - xyz.mean(axis=0)
    c = molecules.contributions(m)
    pos, neg = np.clip(c, 0, None), np.clip(-c, 0, None)
    oily = (xyz * pos[:, None]).sum(axis=0) / pos.sum() if pos.sum() > 0 else np.zeros(3)
    polar = (xyz * neg[:, None]).sum(axis=0) / neg.sum() if neg.sum() > 0 else np.zeros(3)
    v = oily - polar
    return v / np.linalg.norm(v) if np.linalg.norm(v) > 0.3 else np.array([0.0, 0.0, -1.0])


def build_multi(species: list[Species], rng: np.random.Generator, labels: list[str] | None = None,
                n_solvent: int = 46, cycle_frames: float = 10.0, flow_step: float = 1.4,
                n_copies: int = 1) -> Scene:
    """Several compounds in one window that follows species 0's band.

    The solvent drawn is the composition at species 0's band. Every species holds and hops by its own k (at its
    own band). A species whose band moves away keeps round(n_copies x presence) molecules in the window; the
    others leave: forced free and carried out downstream (band ahead) or left behind upstream (band behind).
    Hop timing per copy: two-state Markov chain, mean free stretch max(1, cycle / (1 + k)) frames, mean held
    stretch k times that, so the held share is k / (1 + k) whatever k is.
    """
    n = len(species[0].phi)
    nsp = len(species)
    tot = nsp * n_copies

    xyz0, sink, extent, elems = [], [], [], []
    hyd_c, cmp_c, rad = [], [], []
    for s, sp in enumerate(species):
        x = sp.mol.GetConformer().GetPositions()
        xyz0.append(x - x.mean(axis=0))
        sink.append(_align(amphiphilic_axis(sp.mol), np.array([0.0, 0.0, -1.0])))
        extent.append(float(np.max(np.linalg.norm(xyz0[-1], axis=1))))
        el = [a.GetSymbol() for a in sp.mol.GetAtoms()]
        elems.append(el)
        hyd_c.append([molecules.colour_for(float(c)) for c in molecules.contributions(sp.mol)])
        cmp_c.append([COMPOUND_HUES[s % len(COMPOUND_HUES)] if e in ("C", "H") else ELEMENT.get(e, "#dddddd")
                      for e in el])
        rad.append([0.22 if e == "H" else 0.38 for e in el])

    rank = rng.random(n_solvent)
    sp_pos = np.c_[rng.uniform(0, BOX_X, n_solvent), rng.uniform(0, BOX_Y, n_solvent),
                   rng.uniform(TOP_Z + 1, CEIL_Z, n_solvent)]
    sp_rot = [_rot(rng.normal(size=3), rng.uniform(0, 2 * math.pi)) for _ in range(n_solvent)]

    who = np.repeat(np.arange(nsp), n_copies)                     # species of each copy
    p0 = np.array([species[w].k[0] / (1 + species[w].k[0]) for w in who])
    held = rng.random(tot) < p0
    was = np.zeros(tot, dtype=bool)
    lanes = (rng.permutation(tot) + 0.5) / tot
    cen = np.c_[rng.uniform(0, BOX_X, tot), 3 + (BOX_Y - 6) * lanes,
                np.array([TOP_Z + extent[w] + 4 for w in who])]
    cen[0, 0] = BOX_X * 0.3
    Rs = [_rot(rng.normal(size=3), rng.uniform(0, 2 * math.pi)) for _ in range(tot)]
    inside = np.ones(tot, dtype=bool)          # in the window
    leaving = np.zeros(tot, dtype=bool)

    frames, present, n_an, tags, info, held_l, centers = [], [], [], [], [], [], []
    sep = " &nbsp;|&nbsp; "
    for f in range(n):
        # Which copies should be in the window: the first round(n_copies x presence) of each species.
        for s, sp in enumerate(species):
            idx = np.flatnonzero(who == s)
            want = 0 if sp.eluted[f] else int(round(n_copies * float(np.clip(sp.presence[f], 0, 1))))
            have = idx[inside[idx] & ~leaving[idx]]
            if len(have) > want:                                    # send the free ones off first
                order = sorted(have, key=lambda c: (held[c], -c))
                leaving[order[:len(have) - want]] = True
            elif len(have) < want:
                back = [c for c in idx if not inside[c]][:want - len(have)]
                for c in back:                                       # re-enter from the side the band is on
                    inside[c], leaving[c] = True, False
                    cen[c] = [BOX_X - 1 if sp.ahead[f] > 0 else 1.0, cen[c, 1], TOP_Z + extent[s] + 4]
            if sp.eluted[f]:
                inside[idx], leaving[idx] = False, False
        lines, pres, tg, cn = [], [], [], []
        u = rng.random(tot)
        for c in range(tot):
            s = who[c]
            if not inside[c]:
                continue
            kf = float(species[s].k[f])
            free_mean = max(1.0, cycle_frames / (1 + kf))
            hold_mean = kf * free_mean
            if leaving[c]:
                held[c] = False
            elif held[c]:
                held[c] = u[c] >= 1.0 / max(hold_mean, 1.0)
            else:
                held[c] = u[c] < 1.0 / free_mean
            if held[c]:
                base = sink[s] if not was[c] else Rs[c]
                Rs[c] = _rot(rng.normal(size=3), rng.normal(0, 0.04)) @ base
                cen[c] = [cen[c, 0] + rng.normal(0, 0.08), cen[c, 1] + rng.normal(0, 0.08),
                          0.5 * cen[c, 2] + 0.5 * (TOP_Z - 0.45 * extent[s])]
            else:
                Rs[c] = _rot(rng.normal(size=3), rng.normal(0, 0.14)) @ Rs[c]
                step = flow_step
                if leaving[c]:
                    step = 2.5 * flow_step if species[s].ahead[f] > 0 else -2.5 * flow_step
                zc = float(np.clip(cen[c, 2] + rng.normal(0.8, 1.2), TOP_Z + extent[s] * 0.6 + 2, CEIL_Z - 2))
                cen[c] = [cen[c, 0] + step + rng.normal(0, 0.3),
                          float(np.clip(cen[c, 1] + rng.normal(0, 0.4), 3, BOX_Y - 3)), zc]
            if leaving[c] and not (0 <= cen[c, 0] <= BOX_X):
                inside[c], leaving[c] = False, False                 # gone from the window
                continue
            if not leaving[c]:
                cen[c, 0] %= BOX_X
            pos = (Rs[c] @ xyz0[s].T).T + cen[c]
            lines += [f"{e} {x:.2f} {y:.2f} {z:.2f}" for e, (x, y, z) in zip(elems[s], pos)]
            pres.append(int(s))
            cn.append([round(float(x), 2) for x in cen[c]])
            top = pos[:, 2].max() + 1.5
            tg.append([species[s].name, round(float(cen[c, 0]), 2), round(float(cen[c, 1]), 2), round(float(top), 2)])
        was = held.copy()
        chi = methanol_mole_fraction(float(species[0].phi[f]))
        for j in range(n_solvent):
            meoh = rank[j] < chi
            floor = TOP_Z - 4 if meoh else TOP_Z + 1.5           # methanol soaks into the chain tops, water does not
            speed = flow_step * (0.3 if sp_pos[j, 2] < TOP_Z + 2 else 1.0)
            sp_pos[j] = [(sp_pos[j, 0] + speed + rng.normal(0, 0.4)) % BOX_X, (sp_pos[j, 1] + rng.normal(0, 0.4)) % BOX_Y,
                         float(np.clip(sp_pos[j, 2] + rng.normal(0, 0.8), floor, CEIL_Z))]
            sp_rot[j] = _rot(rng.normal(size=3), rng.normal(0, 0.3)) @ sp_rot[j]
            for e, *v in (METHANOL if meoh else WATER):
                x, y, z = sp_rot[j] @ np.array(v) + sp_pos[j]
                lines.append(f"{e} {x:.2f} {y:.2f} {z:.2f}")
        frames.append(f"{len(lines)}\nframe {f}\n" + "\n".join(lines))
        present.append(pres)
        centers.append(cn)
        n_an.append(sum(len(elems[s]) for s in pres))
        tags.append(tg if nsp > 1 else [])
        head = (labels[f] + sep) if labels else ""
        head += f"window at the {species[0].name} band: %B {100 * species[0].phi[f]:.0f}% (methanol {100 * chi:.0f}% of molecules)"
        parts = []
        for s, sp in enumerate(species):
            kf = float(sp.k[f])
            idx = np.flatnonzero(who == s)
            n_in = int(np.sum(inside[idx] & ~leaving[idx]))
            n_held = int(np.sum(held[idx] & inside[idx] & ~leaving[idx]))
            if sp.eluted[f]:
                st_ = "band has left the column"
            elif n_copies == 1:
                st_ = ("not in this window" if n_in == 0 else "held in the C18 layer" if n_held else "moving with the mobile phase")
            else:
                st_ = f"{n_in} of {n_copies} in this window, {n_held} held"
            extra = (" " + sp.note[f]) if sp.note and sp.note[f] else ""
            parts.append(f"<b>{sp.name}</b>{extra}: k {kf:.3g}, held {100 * kf / (1 + kf):.1f}% of the time, "
                         f"average speed {100 / (1 + kf):.2g}% of the solvent; <b>{st_}</b>")
        info.append(head + "<br>" + "<br>".join(parts))
        held_l.append(bool(held[0] and inside[0] and not species[0].eluted[f]))
    return Scene(surface_xyz(rng), frames, present, n_an, {"hyd": hyd_c, "cmp": cmp_c}, rad, tags, info, held_l,
                 list(map(float, species[0].phi)), list(map(float, species[0].k)), hyd_c[0],
                 centers, [sp.name for sp in species], [round(float(e), 3) for e in extent])


def build(m: Chem.Mol, phi: np.ndarray, k: np.ndarray, rng: np.random.Generator, labels: list[str] | None = None,
          eluted: np.ndarray | None = None, n_solvent: int = 46, cycle_frames: float = 10.0,
          flow_step: float = 1.4, n_copies: int = 1, name: str = "") -> Scene:
    """One compound: the window is at its band (see build_multi)."""
    n = len(phi)
    el = np.zeros(n, dtype=bool) if eluted is None else np.asarray(eluted, dtype=bool)
    sp = Species(m, name, np.asarray(phi, float), np.asarray(k, float), np.ones(n), np.ones(n), [""] * n, el)
    return build_multi([sp], rng, labels, n_solvent, cycle_frames, flow_step, n_copies)


def html(scene: Scene, height: int = 520, frame_ms: int = 140, colour: str = "hyd") -> str:
    """Self-contained player: 3Dmol.js from its CDN, Play / Pause, a frame slider, a colour switch, and a camera
    that rotates and zooms but never pans.

    3Dmol's own mouse handling is off (nomouse) and the page sets the view itself. Rotation: turntable (drag
    sideways spins about the surface normal through the view centre, up and down tilts between 5 and 88 degrees
    above the surface, so the surface stays down), spin only (one plane), or free trackball. Zoom: the wheel,
    within 0.7 to 4x of the fitted view, in the whole-surface view; or "close on <compound>", which centres on
    that compound's first drawn molecule and follows it from frame to frame with a framing set by its size
    (wheel off), easing back to the whole surface while no molecule of it is in the window. The view array 3Dmol
    exposes is [minus the centre (x, y, z), camera z, quaternion (x, y, z, w)], the quaternion taking model axes
    to screen axes (x right, y up, z toward the viewer).
    """
    data = json.dumps({"surface": scene.surface, "frames": scene.frames, "present": scene.present,
                       "col": scene.sp_colours, "rad": scene.sp_radii, "tags": scene.tags, "info": scene.info,
                       "cen": scene.centers, "ext": scene.extents})
    views = "".join(f'<option value="{i}">View: close on {escape(nm)}</option>' for i, nm in enumerate(scene.names))
    return (_PLAYER.replace("__DATA__", data).replace("__NF__", str(len(scene.frames) - 1))
            .replace("__BOXH__", str(height - 96)).replace("__BG__", molecules.BG).replace("__COLOUR__", colour)
            .replace("__MS__", str(frame_ms)).replace("__VIEWS__", views))


_PLAYER = r"""
<div style="width:100%;background:__BG__;font:12px sans-serif;color:#e6e6e6">
  <div id="c18box" style="position:relative;width:100%;height:__BOXH__px;touch-action:none;cursor:grab"></div>
  <div style="display:flex;gap:8px;align-items:center;padding:6px 4px 2px 4px">
    <button id="c18play" style="padding:2px 10px">Play</button>
    <input id="c18slider" type="range" min="0" max="__NF__" value="0" style="flex:1">
    <select id="c18col" style="font-size:12px"><option value="hyd">Colour: hydrophobicity</option>
      <option value="cmp">Colour: by compound</option></select>
  </div>
  <div style="display:flex;gap:8px;align-items:center;padding:2px 4px 2px 4px;flex-wrap:wrap">
    <select id="c18view" style="font-size:12px" title="Close on a compound: the camera centres on one of its molecules and follows it">
      <option value="-1">View: whole surface</option>__VIEWS__</select>
    <select id="c18rot" style="font-size:12px" title="Turntable: spin about the surface normal, tilt between 5 and 88 degrees. Spin only: one plane. Free: trackball">
      <option value="turn">Rotate: turntable</option><option value="spin">Rotate: spin only</option>
      <option value="free">Rotate: free</option></select>
    <button id="c18reset" style="padding:2px 10px">Reset view</button>
    <span style="opacity:0.7">drag rotates, wheel zooms (whole surface only), nothing pans</span>
  </div>
  <div id="c18info" style="padding:2px 6px 6px 6px;line-height:1.5;min-height:36px"></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"></script>
<script>
const S = __DATA__;
const box = document.getElementById("c18box"), info = document.getElementById("c18info");
const slider = document.getElementById("c18slider"), btn = document.getElementById("c18play");
const sel = document.getElementById("c18col"); sel.value = "__COLOUR__";
const viewSel = document.getElementById("c18view"), rotSel = document.getElementById("c18rot");
const v = $3Dmol.createViewer(box, {backgroundColor: "__BG__", nomouse: true});
const surf = v.addModel(S.surface, "xyz");
v.setStyle({model: surf, elem: "C"}, {stick: {radius: 0.13, color: "#7d8594"}, sphere: {radius: 0.28, color: "#7d8594"}});
v.setStyle({model: surf, elem: "Si"}, {sphere: {radius: 0.85, color: "#c8b48c"}});
v.setStyle({model: surf, elem: "O"}, {sphere: {radius: 0.65, color: "#b04848"}});

// Camera state: q (model axes -> screen axes), zf (zoom relative to the fitted surface), cc (model point at the
// screen centre, also the rotation centre), follow (species index, -1 = whole surface).
v.zoomTo({model: surf});
const base = v.getView(), CZ = typeof v.CAMERA_Z === "number" ? v.CAMERA_Z : 150, fitD = CZ - base[3];
const surfC = [-base[0], -base[1], -base[2]];
let fitR = 0;
for (const a of surf.selectedAtoms({})) fitR = Math.max(fitR, Math.hypot(a.x - surfC[0], a.y - surfC[1], a.z - surfC[2]));
const DEG = Math.PI / 180, ZOOM0 = 1.15, EL0 = 18;
let q, zf, fz = 1, follow = -1, cc = surfC.slice(), dyn = null, k = 0, timer = null;
function qAxis(x, y, z, deg) { const s = Math.sin(deg * DEG / 2); return [x * s, y * s, z * s, Math.cos(deg * DEG / 2)]; }
function qMul(a, b) {
  const r = [a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1],
             a[1] * b[3] + a[3] * b[1] + a[2] * b[0] - a[0] * b[2],
             a[2] * b[3] + a[3] * b[2] + a[0] * b[1] - a[1] * b[0],
             a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
  const n = Math.hypot(r[0], r[1], r[2], r[3]); return r.map(x => x / n);
}
function qRot(p, u) {                                   // rotate vector u by unit quaternion p
  const t = [2 * (p[1] * u[2] - p[2] * u[1]), 2 * (p[2] * u[0] - p[0] * u[2]), 2 * (p[0] * u[1] - p[1] * u[0])];
  return [u[0] + p[3] * t[0] + p[1] * t[2] - p[2] * t[1], u[1] + p[3] * t[1] + p[2] * t[0] - p[0] * t[2],
          u[2] + p[3] * t[2] + p[0] * t[1] - p[1] * t[0]];
}
function elev() {                                       // camera height above the surface plane, degrees
  const n = qRot(q, [0, 0, 1]); return Math.asin(Math.max(-1, Math.min(1, n[2]))) / DEG;
}
function level() {                                      // back to a turntable pose: tilt about screen x after a spin about the normal
  const th = elev() - 90;
  const mx = qRot(qAxis(1, 0, 0, -th), qRot(q, [1, 0, 0]));
  q = qMul(qAxis(1, 0, 0, th), qAxis(0, 0, 1, Math.atan2(mx[1], mx[0]) / DEG));
}
function closeFactor() { return Math.min(6, Math.max(1.5, fitR / (2.2 * S.ext[follow]))); }
function cam(draw) {                                   // draw: redraw now (an interaction); else the caller renders
  const arr = [-cc[0], -cc[1], -cc[2], CZ - fitD / (zf * fz), q[0], q[1], q[2], q[3]];
  if (!draw && v.modelGroup && v.rotationGroup) {      // the fields setView writes, set without its draw (one draw per frame)
    v.modelGroup.position.set(arr[0], arr[1], arr[2]); v.rotationGroup.position.z = arr[3];
    v.rotationGroup.quaternion.set(arr[4], arr[5], arr[6], arr[7]);
  } else v.setView(arr);
}
function resetPose() { q = qAxis(1, 0, 0, EL0 - 90); zf = ZOOM0; }
function target(f) { const i = S.present[f].indexOf(follow); return i < 0 ? null : S.cen[f][i]; }

function show(f) {
  if (dyn) v.removeModel(dyn);
  v.removeAllLabels();
  dyn = v.addModel(S.frames[f], "xyz");
  v.setStyle({model: dyn}, {stick: {radius: 0.1, colorscheme: "Jmol"}, sphere: {scale: 0.2, colorscheme: "Jmol"}});
  let off = 0;
  for (const sp of S.present[f]) {
    const cols = S.col[sel.value][sp], rad = S.rad[sp];
    for (let j = 0; j < cols.length; j++)
      v.setStyle({model: dyn, index: off + j}, {stick: {radius: 0.17, color: cols[j]}, sphere: {radius: rad[j], color: cols[j]}});
    off += cols.length;
  }
  for (const t of S.tags[f])
    v.addLabel(t[0], {position: {x: t[1], y: t[2], z: t[3]}, fontSize: 10, fontColor: "white",
      backgroundColor: "black", backgroundOpacity: 0.4, showBackground: true, inFront: true});
  // Ease toward the followed molecule, or back to the whole surface while none of that compound is in the window.
  const t = follow >= 0 ? target(f) : null, goal = t || surfC, gf = t ? closeFactor() : 1;
  cc = cc.map((c, i) => c + 0.4 * (goal[i] - c)); fz += 0.4 * (gf - fz);
  cam(false); info.innerHTML = S.info[f]; slider.value = f; v.render();
}
resetPose(); show(0);

slider.addEventListener("input", e => { k = +e.target.value; show(k); });
sel.addEventListener("change", () => show(k));
btn.addEventListener("click", () => {
  if (timer) { clearInterval(timer); timer = null; btn.textContent = "Play"; return; }
  btn.textContent = "Pause";
  timer = setInterval(() => { k = (k + 1) % S.frames.length; show(k); }, __MS__);
});
viewSel.addEventListener("change", () => {
  follow = +viewSel.value; zf = ZOOM0;
  const t = follow >= 0 ? target(k) : null;
  cc = (t || surfC).slice(); fz = t ? closeFactor() : 1; cam(true);
});
rotSel.addEventListener("change", () => { if (rotSel.value !== "free") level(); cam(true); });
document.getElementById("c18reset").addEventListener("click", () => {
  resetPose(); viewSel.value = "-1"; follow = -1; cc = surfC.slice(); fz = 1; cam(true);
});
let drag = null;
box.addEventListener("pointerdown", e => {
  if (e.button !== 0) return;
  drag = [e.clientX, e.clientY]; box.setPointerCapture(e.pointerId); box.style.cursor = "grabbing";
});
box.addEventListener("pointermove", e => {
  if (!drag) return;
  const dx = 360 * (e.clientX - drag[0]) / box.clientWidth, dy = 360 * (e.clientY - drag[1]) / box.clientHeight;
  drag = [e.clientX, e.clientY];
  if (rotSel.value === "free") q = qMul(qMul(qAxis(0, 1, 0, dx), qAxis(1, 0, 0, dy)), q);
  else {
    q = qMul(q, qAxis(0, 0, 1, dx));                    // spin about the surface normal (model z)
    if (rotSel.value === "turn") {                      // tilt about screen x, camera kept 5 to 88 deg above the surface
      const el = elev(); q = qMul(qAxis(1, 0, 0, Math.max(5, Math.min(88, el + dy)) - el), q);
    }
  }
  cam(true);
});
const stop = () => { drag = null; box.style.cursor = "grab"; };
box.addEventListener("pointerup", stop); box.addEventListener("pointercancel", stop);
box.addEventListener("wheel", e => {
  e.preventDefault(); e.stopPropagation();
  if (follow >= 0) return;                              // close view: the framing is the preset
  const d = Math.max(-100, Math.min(100, e.deltaMode === 1 ? 16 * e.deltaY : e.deltaY));
  zf = Math.max(0.7, Math.min(4, zf * Math.exp(-d / 400))); cam(true);
}, {capture: true, passive: false});
</script>
"""
