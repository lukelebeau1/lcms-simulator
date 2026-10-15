"""Molecules and ions of the method in 3D, with each atom's share of hydrophobicity. Illustration, not a model.

Structures: data/reference/structures/*.sdf (scripts/fetch_structures.py: PubChem 3D conformers, RDKit-built where
PubChem has none). Hydrophobicity per atom: the Wildman-Crippen atomic contributions to log P (RDKit), which add
up to the molecule's Crippen log P; summed over functional groups they show which parts of the molecule hold it
on a C18 column (positive) and which pull it into the mobile phase (negative). Crippen log P is a different
estimate from the PubChem XLogP3 used by lcms_sim.lc.analogy; both are shown.

Ions: the precursor type is read from the method (precursor m/z minus the neutral's monoisotopic mass:
+1.007 [M+H]+, +18.034 [M+NH4]+, +22.989 [M+Na]+, 0 for a permanent cation). The added proton or adduct is placed
at the most likely site by a simple ranking (basic N, then carbonyl or P=O/P=S oxygen, then ether O): a sketch of
where the charge sits, not a computed gas-phase structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
REF = Path(__file__).resolve().parents[1] / "data" / "reference"
PROTON, NH4, NA, D_H = 1.007276, 18.033823, 22.989218, 1.006277   # D_H: deuterium minus hydrogen
HEAVY_ISO = 1.9975       # 37Cl - 35Cl (1.99705) and 81Br - 79Br (1.99795): the M+2 step
HF = 20.006229           # H + F, lost in the source by some fluorinated molecules (fludioxonil's positive-mode ion)

# Functional groups, in priority order: an atom belongs to the first group that matches it.
GROUPS: list[tuple[str, str]] = [
    ("quaternary ammonium (permanent +)", "[NX4+]"),
    ("phosphate / thiophosphate ester", "[PX4](=[O,S])([O,S,N])([O,S,N])"),
    ("phosphate / thiophosphate ester", "[PX4](=[O,S])"),
    ("phosphate / thiophosphate ester", "[O,S;$(*[PX4])]"),
    ("carbamate", "[NX3][CX3](=[OX1])[OX2]"),
    ("urea", "[NX3][CX3](=[OX1])[NX3]"),
    ("carboxylic acid", "[CX3](=O)[OX2H1]"),
    ("ester / lactone", "[CX3](=O)[OX2][#6]"),
    ("amide / hydrazide", "[NX3][CX3]=[OX1]"),
    ("nitro / nitroimine", "[NX3+](=O)[O-]"),
    ("nitro / nitroimine", "[NX2]=[NX3+](=O)[O-]"),
    ("nitrile", "[CX2]#[NX1]"),
    ("triazole ring", "[#7]1:[#7]:[#6]:[#7]:[#6]:1"),
    ("imidazole / pyrazole ring", "[#7]1:[#6]:[#7]:[#6]:[#6]:1"),
    ("imidazole / pyrazole ring", "[#7]1:[#7]:[#6]:[#6]:[#6]:1"),
    ("N-heteroaromatic ring (pyridine, pyrimidine...)", "[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1"),
    ("ketone / aldehyde", "[#6][CX3](=O)[#1,#6]"),
    ("hydroxyl (alcohol or phenol)", "[OX2H1][#6]"),
    ("thioether / sulfur", "[#16]"),
    ("ether / acetal oxygen", "[OX2]([#6])[#6]"),
    ("CF3 / CHF2 / OCF3", "[CX4](F)(F)"),
    ("halogen on a ring", "[F,Cl,Br,I;$(*a)]"),
    ("halogen on a chain", "[F,Cl,Br,I]"),
    ("basic amine / other N", "[#7]"),
    ("other heteroatom", "[!#6;!#1]"),
    ("aromatic ring (carbon)", "c"),
    ("carbon chain / skeleton", "[#6]"),
]


@dataclass(frozen=True)
class Group:
    name: str
    atoms: tuple[int, ...]        # heavy atoms and their hydrogens
    logp: float                   # summed Crippen contributions


def _ring_atoms(m: Chem.Mol, smarts: str) -> list[tuple[int, ...]]:
    """For the N-heteroaromatic pattern: only six-membered aromatic rings that contain an N."""
    out = []
    for ring in m.GetRingInfo().AtomRings():
        if len(ring) == 6 and all(m.GetAtomWithIdx(i).GetIsAromatic() for i in ring) and \
                any(m.GetAtomWithIdx(i).GetSymbol() == "N" for i in ring):
            out.append(tuple(ring))
    return out


@lru_cache(maxsize=None)
def table() -> pd.DataFrame:
    return pd.read_csv(REF / "structures.csv").set_index("name")


@lru_cache(maxsize=None)
def load(name: str) -> Chem.Mol:
    """The 3D structure with explicit hydrogens."""
    m = Chem.MolFromMolFile(str(REF / table().loc[name, "file"]), removeHs=False, sanitize=True)
    if m is None:
        raise ValueError(f"cannot read structure for {name}")
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    return max(frags, key=lambda f: f.GetNumHeavyAtoms())      # drop counter-ions (Imazalil-D5 is a sulfate salt)


def contributions(m: Chem.Mol) -> np.ndarray:
    """Wildman-Crippen log P contribution of every atom (hydrogens included); sums to Crippen.MolLogP."""
    return np.array([c[0] for c in rdMolDescriptors._CalcCrippenContribs(m)])


def groups(m: Chem.Mol) -> list[Group]:
    """Every heavy atom (with its hydrogens) assigned to one functional group; contributions summed per group."""
    contrib = contributions(m)
    owner: dict[int, str] = {}
    for name, smarts in GROUPS:
        matches = _ring_atoms(m, smarts) if name.startswith("N-heteroaromatic") else \
            m.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        for match in matches:
            for i in match:
                if m.GetAtomWithIdx(i).GetAtomicNum() > 1 and i not in owner:
                    owner[i] = name
    members: dict[str, list[int]] = {}
    for a in m.GetAtoms():
        i = a.GetIdx()
        if a.GetAtomicNum() == 1:
            nb = a.GetNeighbors()
            key = owner.get(nb[0].GetIdx(), "carbon chain / skeleton") if nb else "carbon chain / skeleton"
        else:
            key = owner.get(i, "carbon chain / skeleton")
        members.setdefault(key, []).append(i)
    out = [Group(k, tuple(v), float(contrib[v].sum())) for k, v in members.items()]
    return sorted(out, key=lambda g: g.logp, reverse=True)


def n_deuterium(m: Chem.Mol) -> int:
    return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 1 and a.GetIsotope() == 2)


def precursor_type(name: str, precursor_mz: float, polarity: str = "positive") -> tuple[str, int, int]:
    """Which ion the method selects: (type, deuterium missing from the drawn structure, heavy-isotope steps).

    Heavy-isotope steps: methods often select the M+2 or M+4 peak of molecules with several Cl or Br, because
    with 37Cl / 81Br those peaks are as tall as (or taller than) the all-light one.
    """
    m = load(name)
    mass = Descriptors.ExactMolWt(m)
    charged = Chem.GetFormalCharge(m) > 0
    n_halo = sum(1 for a in m.GetAtoms() if a.GetSymbol() in ("Cl", "Br"))
    missing = range(0, 14) if "parent" in str(table().loc[name, "note"]) else (0,)
    kinds = (("[M-H]-", -PROTON),) if polarity == "negative" else         (("[M]+", 0.0), ("[M+H]+", PROTON), ("[M+NH4]+", NH4), ("[M+Na]+", NA), ("[M+H-HF]+", PROTON - HF))
    for iso in range(0, min(n_halo, 4) + 1):
        for nd in missing:
            d = precursor_mz - (mass + nd * D_H + iso * HEAVY_ISO)
            for label, add in kinds:
                if (label == "[M]+") != charged and polarity == "positive":
                    continue
                if abs(d - add) < 0.35:
                    return label, nd, iso
    return "unknown", 0, 0


def _site(m: Chem.Mol) -> int | None:
    """Atom most likely to carry the added proton or adduct (simple ranking, see module notes)."""
    ranks = [
        "[NX3;H2,H1,H0;+0;!$(N-C=[O,S,N]);!$(N-a);!$(N-[S,P]);!$(N-N);!$(N~O);!$(N#*)]",   # aliphatic amine
        "[nX2;r5;$(n:c:n),$(n:n)]",                                                    # azole N with lone pair
        "[nX2;r6]",                                                                    # pyridine-type N
        "[NX2;$(N=C)]",                                                                # imine / amidine N
        "[OX1]=[CX3][NX3]",                                                            # amide carbonyl O
        "[OX1]=[PX4]", "[SX1]=[PX4]",
        "[OX1]=[CX3]",                                                                 # other carbonyl O
        "[OX2;H0]([#6])[#6]",                                                          # ether O
        "[O,N]",
    ]
    for s in ranks:
        hit = m.GetSubstructMatches(Chem.MolFromSmarts(s))
        if hit:
            return hit[0][0]
    return None


def _outward(m: Chem.Mol, i: int) -> np.ndarray:
    pos = m.GetConformer().GetPositions()
    nbs = [n.GetIdx() for n in m.GetAtomWithIdx(i).GetNeighbors()]
    v = pos[i] - (pos[nbs].mean(axis=0) if nbs else pos.mean(axis=0))
    if np.linalg.norm(v) < 1e-6:
        v = np.array([0.0, 0.0, 1.0])
    return v / np.linalg.norm(v)


def make_ion(m: Chem.Mol, kind: str) -> tuple[Chem.Mol, list[int]]:
    """The molecule with the method's proton or adduct added in 3D; returns it and the indices of the added atoms.
    An in-source loss ([M+H-HF]+) is drawn as the protonated molecule it comes from."""
    if kind == "[M+H-HF]+":
        kind = "[M+H]+"
    if kind in ("[M]+", "unknown"):
        return m, []
    if kind == "[M-H]-":
        acid = None
        for smarts in ("[OX2H1][CX3]=O", "[OX2H1]c", "[nH]", "[NX3H1]", "[NX3H2]", "[OX2H1]"):
            hit = m.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            if hit:
                acid = hit[0][0]
                break
        if acid is None:
            return m, []
        h = next(n.GetIdx() for n in m.GetAtomWithIdx(acid).GetNeighbors() if n.GetAtomicNum() == 1)
        rw = Chem.RWMol(m)
        rw.GetAtomWithIdx(acid).SetFormalCharge(-1)
        rw.GetAtomWithIdx(acid).SetNoImplicit(True)
        rw.RemoveAtom(h)
        ion = rw.GetMol()
        ion.UpdatePropertyCache(strict=False)
        return ion, [acid if acid < h else acid - 1]        # highlight the atom that lost its proton
    site = _site(m)
    if site is None:
        return m, []
    rw = Chem.RWMol(m)
    conf = rw.GetConformer()
    p0, u = np.array(conf.GetAtomPosition(site)), _outward(m, site)
    added = []

    def add(symbol: str, xyz, charge: int = 0) -> int:
        a = Chem.Atom(symbol)
        a.SetFormalCharge(charge)
        a.SetNoImplicit(True)
        k = rw.AddAtom(a)
        conf.SetAtomPosition(k, [float(c) for c in xyz])
        added.append(k)
        return k

    if kind == "[M+H]+":
        k = add("H", p0 + u * (1.02 if m.GetAtomWithIdx(site).GetSymbol() == "N" else 0.97))
        rw.AddBond(site, k, Chem.BondType.SINGLE)
        rw.GetAtomWithIdx(site).SetFormalCharge(1)
        rw.GetAtomWithIdx(site).SetNoImplicit(True)
    elif kind == "[M+Na]+":
        add("Na", p0 + u * 2.35, charge=1)
    elif kind == "[M+NH4]+":
        n_at = p0 + u * 2.85
        n = add("N", n_at, charge=1)
        # Tetrahedral hydrogens, one pointing back at the binding oxygen (a hydrogen bond).
        e = [-u]
        perp = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        for ang in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
            r = np.cos(ang) * perp + np.sin(ang) * np.cross(u, perp)
            e.append(u * (1 / 3) + r * np.sqrt(8 / 9))
        for v in e:
            h = add("H", n_at + 1.03 * v / np.linalg.norm(v))
            rw.AddBond(n, h, Chem.BondType.SINGLE)
    ion = rw.GetMol()
    ion.UpdatePropertyCache(strict=False)
    return ion, added


def crippen_logp(m: Chem.Mol) -> float:
    return float(Crippen.MolLogP(m))


# --- 3D viewer -----------------------------------------------------------------------------------

def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(255 * c)) for c in rgb)


BG = "#0e1117"                  # Streamlit's dark background
NEUTRAL, HOLD, PULL = np.array([0.93, 0.95, 0.98]), np.array([0.96, 0.49, 0.09]), np.array([0.20, 0.53, 0.96])
DEAD_ZONE = 0.12                # |share| below this reads as neutral: most C and H carry a small positive share


def colour_for(v: float, vmax: float = 0.8) -> str:
    """Diverging: blue = pulls toward the mobile phase, cool white = neutral, orange = holds on C18."""
    a = abs(v)
    t = 0.0 if a < DEAD_ZONE else min(1.0, (a - DEAD_ZONE) / (vmax - DEAD_ZONE))
    end = HOLD if v > 0 else PULL
    return _hex(NEUTRAL + (end - NEUTRAL) * t)


# Slower wheel zoom than 3Dmol's default: the wheel is caught on the viewer's own div before it reaches the canvas.
_ZOOM_JS = """
<script>
$3Dmolpromise.then(function() {
  var box = document.getElementById("3dmolviewer_UNIQUEID");
  box.addEventListener("wheel", function(e) {
    e.preventDefault(); e.stopPropagation();
    var v = viewer_UNIQUEID; if (!v) return;
    v.zoom(e.deltaY < 0 ? 1 + STEP : 1 / (1 + STEP)); v.render();
  }, {capture: true, passive: false});
});
</script>
"""


def viewer_html(m: Chem.Mol, colour: str = "hydrophobicity", added: list[int] | None = None, spin: bool = False,
                labels: bool = False, height: int = 420, contrib: np.ndarray | None = None,
                label_groups: list[Group] | None = None, atom_labels: bool = True, zoom_step: float = 0.25) -> str:
    """Self-contained HTML (3Dmol.js from its CDN) of a rotatable molecule on the app's dark background.

    For an ion, pass the neutral molecule's `contrib` and `label_groups`: the added atoms come last and are drawn
    in magenta. `atom_labels` marks every atom that is not C or H with its element. `zoom_step` is the zoom per
    wheel notch (3Dmol's own is much larger).
    """
    import py3Dmol

    added = added or []
    v = py3Dmol.view(width="100%", height=height)
    v.addModel(Chem.MolToMolBlock(m), "sdf")
    v.setBackgroundColor(BG)
    if colour == "hydrophobicity":
        contrib = contributions(m) if contrib is None else contrib
        for a in m.GetAtoms():
            i = a.GetIdx()
            c = "#e040e0" if i in added else colour_for(float(contrib[i]) if i < len(contrib) else 0.0)
            r = 0.22 if a.GetAtomicNum() == 1 else 0.33
            v.setStyle({"index": i}, {"stick": {"color": c, "radius": 0.16}, "sphere": {"color": c, "radius": r}})
    else:
        v.setStyle({}, {"stick": {"radius": 0.16, "colorscheme": "Jmol"}, "sphere": {"scale": 0.25, "colorscheme": "Jmol"}})
        for i in added:
            v.setStyle({"index": i}, {"stick": {"color": "#e040e0", "radius": 0.16}, "sphere": {"color": "#e040e0", "radius": 0.35}})
    pos = m.GetConformer().GetPositions()
    if atom_labels:
        for a in m.GetAtoms():
            if a.GetSymbol() in ("C", "H") or a.GetIdx() in added:
                continue
            x, y, z = pos[a.GetIdx()]
            q = a.GetFormalCharge()
            text = a.GetSymbol() + ("+" if q > 0 else "-" if q < 0 else "")
            v.addLabel(text, {"position": {"x": float(x), "y": float(y), "z": float(z)}, "fontSize": 12,
                              "fontColor": "white", "backgroundColor": "black", "backgroundOpacity": 0.45,
                              "showBackground": True, "alignment": "center", "inFront": True})
    if labels:
        for g in (label_groups if label_groups is not None else groups(m)):
            heavy = [i for i in g.atoms if i < m.GetNumAtoms() and m.GetAtomWithIdx(i).GetAtomicNum() > 1]
            if g.name.startswith("carbon chain") or not heavy:
                continue
            x, y, z = pos[heavy].mean(axis=0)
            v.addLabel(f"{g.name.split(' (')[0]} {g.logp:+.1f}",
                       {"position": {"x": float(x), "y": float(y) + 1.2, "z": float(z)}, "fontSize": 11,
                        "fontColor": "#ffe7a8", "backgroundColor": "#1b1f2a", "backgroundOpacity": 0.85,
                        "showBackground": True, "inFront": True})
    for i in added[:1]:
        x, y, z = pos[i]
        v.addLabel("+" if Chem.GetFormalCharge(m) >= 0 else "-",
                   {"position": {"x": float(x), "y": float(y), "z": float(z)}, "fontSize": 15,
                    "fontColor": "#e040e0", "showBackground": False, "inFront": True})
    v.zoomTo()
    if spin:
        v.spin("y", 0.6)
    html = v._make_html()
    return html + _ZOOM_JS.replace("UNIQUEID", v.uniqueid).replace("STEP", f"{zoom_step:g}")


def draw_2d(m: Chem.Mol, width: int = 460, height: int = 320) -> str:
    """The conventional skeletal diagram (SVG, dark style), hydrogens implicit, charges shown."""
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    flat = Chem.RemoveHs(Chem.Mol(m), sanitize=False)
    flat.UpdatePropertyCache(strict=False)
    AllChem.Compute2DCoords(flat)
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = d.drawOptions()
    rdMolDraw2D.SetDarkMode(opts)
    opts.setBackgroundColour((0.055, 0.067, 0.09, 1.0))
    opts.addStereoAnnotation = False
    opts.bondLineWidth = 2
    d.DrawMolecule(rdMolDraw2D.PrepareMolForDrawing(flat, kekulize=True, addChiralHs=False))
    d.FinishDrawing()
    return d.GetDrawingText()
