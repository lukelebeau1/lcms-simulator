"""Generate the bundled example input set, data/examples/.

    python scripts/make_example_data.py

`Method` and `write_set` take any method's tables; a lab builds its own set the same way from its own documents
(that builder stays with the lab). Sources of the bundled example (data/reference/):
  * agilent_5994-1734en_transitions.csv, parsed from Appendix A of Agilent application note 5994-1734EN by
    scripts/parse_agilent_appnote.py: transitions, fragmentor, collision energy, retention time and window,
    measured by Agilent on the note's column and gradient. Within a compound the first row is the quantifier.
  * The note's method text: Poroshell 120 Phenyl-Hexyl 3.0 x 100 mm, 2.7 um, 55 C; water / methanol with ammonium
    formate and formic acid at 0.5 mL/min; the gradient below; 10 uL injection; 1 g flower, 15 mL MeCN, C18 SPE,
    two 5 mL rinses, made to 25 mL, then 50 uL in 500 uL (250-fold overall); matrix-matched calibrators made in
    the final extract; recoveries spiked at 60 ppb (pesticides) and 4 ppb (mycotoxins).
  * mn_ocm_cta_v2_analytes.csv: the Minnesota OCM list and action limits. Its analytes the note does not cover
    are listed in GAPS.md.
Assumed, because the note gives none: internal standards (the note uses none; here nine isotope-labelled analogues
in 50 uL of the 450 uL diluent, each target paired with the nearest in retention time and polarity, transitions
shifted by the label where the label stays on the product), calibration and QC levels, isomer fractions, dwell
volume, S (from molecular weight; one gradient cannot separate S from log kw), response factors, ion ratios,
detector gain, the dMRM cycle, the matrix profile and the PT sample.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

from lcms_sim.io.schemas import AcquisitionMethod, GradientProgram, Transition
from lcms_sim.lc.gradient import LCConditions
from lcms_sim.lc.retention import estimate_S, fit_log_kw, retention_time
from lcms_sim.ms.acquisition import build_schedule
from lcms_sim.paths import EXAMPLES, REFERENCE

SEED = 20261007
TOTAL_POROSITY = 0.5
DWELL_ML = 0.6                    # assumed, 1260 binary pump G7112B (the note's pump)
D_SHIFT = 2.014102 - 1.007825     # m/z per deuterium for a singly charged ion

RESPONSE_FACTOR = 3.0e5           # placeholder area per ng on column
RESPONSE_FACTOR_MYCO = 3.0e6      # placeholder; aflatoxins ionise well
QUAL_RATIOS = (0.5, 0.25, 0.15)   # placeholder ion ratios for 1st, 2nd, 3rd qualifier
MYCOTOXINS = {"Aflatoxin B1", "Aflatoxin B2", "Aflatoxin G1", "Aflatoxin G2", "Ochratoxin A"}
ACID_BASE = {"Imazalil": ("base", 6.5), "Imazalil-D5": ("base", 6.5), "Ochratoxin A": ("acid", 4.4),
             "Spinosyn A": ("base", 8.1), "Spinosyn D": ("base", 7.9), "Chlormequat chloride": ("base", 14.0)}

# Regulated name of each component the MN list reports as a sum or under another name; others are their own.
REGULATED = {
    "Avermectin B1a": "Abamectin", "Avermectin B1b": "Abamectin", "Dichlorvos": "Dichlorvos (DDVP)",
    "Permethrin": "Permethrins (cis + trans)", "Piperonyl butoxide": "Piperonyl butoxide",
    **{c: "Pyrethrins (sum of 6)" for c in ("Pyrethrin I", "Pyrethrin II", "Cinerin I", "Cinerin II",
                                            "Jasmolin I", "Jasmolin II")},
    "Spinosyn A": "Spinosad", "Spinosyn D": "Spinosad",
}


def norm(name: str) -> str:
    """Match names across tables, forgiving case, punctuation and the SOP tables' misspellings."""
    s = re.sub(r"[^a-z0-9]", "", name.lower())
    return s.replace("imazilil", "imazalil").replace("imidicloprid", "imidacloprid").replace("ivermectrin", "ivermectin")


@dataclass
class Method:
    """Everything that differs between input sets; write_set does the rest the same way for both."""

    out: Path
    info: dict
    column: tuple[float, float, float]                 # length mm, id mm, particle um
    flow: float
    gradient: list[tuple[float, float]]
    run_time: float
    ph: float
    lc_name: str
    tx: pd.DataFrame                                   # compound, precursor/product, role, RT, window, polarity, ...
    istds: list[str]
    pairs: dict[str, str]
    frac: dict[str, float]
    measured_rt: dict[str, float]                      # retention times log kw is fitted to; tx RT otherwise
    rt_label: str
    acquisition: dict
    prep: dict
    calibration: dict
    levels: tuple[list[float], list[float]]            # pesticide and mycotoxin vial levels, ng/mL
    samples: list[dict]
    recipe: dict
    acceptance: dict
    scouting: dict[str, list[tuple[float, float]]]
    suppression: tuple[float, float]                   # late matrix suppression centre and width, min
    gaps: str
    flags: list[str] = field(default_factory=list)


def write_yaml(out: Path, name: str, data, header: str) -> None:
    (out / name).write_text(f"# {header}\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                            encoding="utf-8")


def gradient_yaml(program) -> dict:
    return {"points": [{"time_min": t, "percent_b": b} for t, b in program]}


def conditions(m: Method, program=None) -> LCConditions:
    prog = GradientProgram.model_validate(gradient_yaml(program or m.gradient))
    length, idm, _ = m.column
    vm = TOTAL_POROSITY * math.pi * (idm / 20) ** 2 * (length / 10)
    return LCConditions.from_program(prog, t0=vm / m.flow, tD=DWELL_ML / m.flow, run_time=m.run_time)


def write_set(m: Method) -> None:
    rng = np.random.default_rng(SEED)
    out = m.out
    out.mkdir(parents=True, exist_ok=True)
    header = f"{m.info['name']}; generated by scripts/make_example_data.py, see its docstring and method.yaml."
    cond = conditions(m)
    analytes, transitions, rt_src = [], [], {}
    for n in dict.fromkeys(m.tx.compound):
        t = m.tx[m.tx.compound == n].reset_index(drop=True)
        q = t[t.role == "quantifier"].iloc[0]
        mw = q.precursor_mz - 1.007 if q.polarity == "positive" else q.precursor_mz + 1.007
        S = estimate_S(mw)
        rt = float(m.measured_rt.get(n, q.retention_time_min))
        src = m.rt_label if n in m.measured_rt else "transition table"
        rt_src[src] = rt_src.get(src, 0) + 1
        target = max(rt, 1.03 * cond.t0)
        if target != rt:
            m.flags.append(f"{n}: RT {rt} min is at or before the void time {cond.t0:.2f} min; modelled at {target:.2f}")
        is_istd = n in m.istds
        ab, pka = ACID_BASE.get(n, ("neutral", None))
        analytes.append({
            "name": n, "is_internal_standard": is_istd, "internal_standard": "" if is_istd else m.pairs.get(n, ""),
            "log_kw": round(fit_log_kw(target, S, cond), 5), "S": round(S, 4), "acid_base": ab,
            "pka": "" if pka is None else pka,
            "response_factor": RESPONSE_FACTOR_MYCO if n in MYCOTOXINS else RESPONSE_FACTOR,
            "regulated_as": "" if is_istd else REGULATED.get(n, n),
            "isomer_fraction": m.frac.get(n, ""), "molecular_weight": round(mw, 2), "provenance": "fitted",
        })
        qi = 0
        for _, r in t.iterrows():
            rel = 1.0 if r.role == "quantifier" else QUAL_RATIOS[min(qi, 2)]
            qi += r.role != "quantifier"
            transitions.append({
                "analyte": n, "precursor_mz": r.precursor_mz, "product_mz": r.product_mz, "role": r.role,
                "retention_time_min": r.retention_time_min, "window_min": r.window_min, "polarity": r.polarity,
                "adduct": "", "fragmentor_v": r.fragmentor_v, "collision_energy_v": r.collision_energy_v,
                "cell_accel_v": r.get("cell_accel_v", ""), "relative_response": rel,
                "internal_standard": "" if is_istd else m.pairs.get(n, ""),
            })
    pd.DataFrame(analytes).to_csv(out / "analytes.csv", index=False)
    pd.DataFrame(transitions).to_csv(out / "transitions.csv", index=False)
    targets = [a["name"] for a in analytes if not a["is_internal_standard"]]

    length, idm, dp = m.column
    write_yaml(out, "system.yaml", {
        "provenance": "assumed", "column_length_mm": length, "column_id_mm": idm, "particle_size_um": dp,
        "t0_min": round(cond.t0, 4), "reference_flow_ml_min": m.flow, "dwell_volume_ml": DWELL_ML,
        "plate_number": 15000.0, "extra_column_sigma_s": 0.5,
    }, header)
    write_yaml(out, "lc_method.yaml", {"name": m.lc_name, "flow_ml_min": m.flow, "gradient": gradient_yaml(m.gradient),
                                       "run_time_min": m.run_time, "ph": m.ph}, header)
    write_yaml(out, "acquisition.yaml", m.acquisition, header)

    write_yaml(out, "scouting_gradients.yaml", {
        "flow_ml_min": m.flow, "gradients": {k: gradient_yaml(v) for k, v in m.scouting.items()}}, header)
    runs = []
    for a in analytes:
        for gid, prog in m.scouting.items():
            tr = retention_time(a["log_kw"], a["S"], conditions(m, prog)) + rng.normal(0, 0.005)
            runs.append({"analyte": a["name"], "gradient_id": gid, "retention_time_min": round(tr, 3)})
    pd.DataFrame(runs).to_csv(out / "scouting_runs.csv", index=False)

    # Synthetic infusion traces: suppression at the void and where the cannabinoids elute.
    t = np.round(np.arange(0.0, m.run_time + 1e-9, 0.01), 2)
    solvent = 1.0e6 * (1 - 0.01 * t / m.run_time) * (1 + rng.normal(0, 0.01, t.size))
    late, width = m.suppression
    mf = (1 - 0.7 * np.exp(-0.5 * ((t - 1.15 * cond.t0) / 0.25) ** 2)
            - 0.5 * np.exp(-0.5 * ((t - late) / width) ** 2))
    pd.DataFrame({"time_min": t, "signal": np.round(solvent, 0)}).to_csv(out / "infusion_solvent.csv", index=False)
    pd.DataFrame({"time_min": t, "signal": np.round(solvent * mf * (1 + rng.normal(0, 0.01, t.size)), 0)}).to_csv(
        out / "infusion_matrix.csv", index=False)

    write_yaml(out, "prep.yaml", m.prep, header)
    write_yaml(out, "calibration.yaml", m.calibration, header)
    pest, myco = m.levels
    rows = [{"level": f"Cal {i + 1}", **{n: (myco if n in MYCOTOXINS else pest)[i] for n in targets}}
            for i in range(len(pest))]
    pd.DataFrame(rows).to_csv(out / "calibration_design.csv", index=False)
    write_yaml(out, "samples.yaml", {"samples": m.samples}, header)
    write_yaml(out, "prep_recipe.yaml", m.recipe, header)
    write_yaml(out, "acceptance.yaml", m.acceptance, header)
    write_yaml(out, "method.yaml", m.info, "What this input set is and where each input comes from.")
    (out / "GAPS.md").write_text(m.gaps, encoding="utf-8")

    n_is = len(m.istds)
    print(f"{out}: {len(analytes)} compounds ({len(targets)} targets, {n_is} IS), {len(transitions)} transitions")
    print(f"t0 {cond.t0:.2f} min, tD {cond.tD:.2f} min; retention fitted to: {rt_src}")
    for f in m.flags:
        print("FLAG", f)


def qc_amounts(targets, frac, pest_ng_g: float, myco_ng_g: float) -> dict[str, float]:
    """True concentration (ug/kg = ng/g) of every target in a QC spiked at the given totals."""
    return {n: round((myco_ng_g if n in MYCOTOXINS else pest_ng_g) * frac.get(n, 1.0), 6) for n in targets}


# =================================================================================================
# The public example: Agilent 5994-1734EN on the Minnesota list
# =================================================================================================

APPNOTE_NAMES = {"Daminozid": "Daminozide", "Dibrom Naled": "Naled", "Kresoxim methyl": "Kresoxim-methyl"}
# Isotope-labelled internal standards (assumed: the note uses none), bracketing the run from the void to the late
# zone. Parent, deuterium count, whether the label stays on the parent's quantifier product, and why (label
# positions read from the PubChem structures in data/reference/structures/).
LABELLED = [
    ("Daminozide-D4", "Daminozide", 4, True, "label on the succinyl CH2CH2; the 143 product only loses water"),
    ("Imidacloprid-D4", "Imidacloprid", 4, True, "ring label; the 209 product keeps the ring (NO2 lost)"),
    ("Carbaryl-D7", "Carbaryl", 7, True, "naphthyl label; the 145 product is the naphthol"),
    ("Myclobutanil-D4", "Myclobutanil", 4, True, "chlorophenyl label; the 125 product is the chlorobenzyl ion"),
    ("Azoxystrobin-D4", "Azoxystrobin", 4, True, "ring label; the 372 product only loses methanol"),
    ("Tebuconazole-D9", "Tebuconazole", 9, False, "tert-butyl label; the 125 product is the chlorobenzyl ion"),
    ("Etoxazole-D5", "Etoxazole", 5, False, "ethoxy label; the 141 product is the difluorobenzoyl ion"),
    ("Pyridaben-D13", "Pyridaben", 13, True, "label on the tert-butylbenzyl half; the 309 product loses the "
                                             "other tert-butyl as isobutene"),
    ("Permethrin-D5", "Permethrin", 5, True, "label on the phenoxy ring; the 355 product only loses HCl"),
]
# Typical compositions, assumed (a real standard's certificate gives them).
FRACTIONS = {"Avermectin B1a": 0.9, "Avermectin B1b": 0.1, "Spinosyn A": 0.85, "Spinosyn D": 0.15,
             "Pyrethrin I": 0.35, "Pyrethrin II": 0.32, "Cinerin I": 0.10, "Cinerin II": 0.12,
             "Jasmolin I": 0.05, "Jasmolin II": 0.06}

PUB_COLUMN = (100.0, 3.0, 2.7)
PUB_FLOW = 0.5
PUB_GRADIENT = [(0.0, 30), (1.0, 30), (2.0, 75), (8.0, 96), (9.0, 100), (9.5, 100), (9.51, 30), (13.0, 30)]
PUB_RUN = 13.0
PUB_SAMPLE_G, PUB_EXTRACT_ML = 1.0, 25.0                 # 15 mL extraction + 5 + 5 mL rinses, made to 25 mL
PUB_EXTRACT_UL, PUB_DILUENT_UL, PUB_IS_UL, PUB_CAL_UL = 50.0, 400.0, 50.0, 50.0
PUB_VIAL_UL = PUB_EXTRACT_UL + PUB_DILUENT_UL + PUB_IS_UL          # 500 uL, as in the note
PUB_IS_VIAL_NG_ML = 10.0
PUB_PEST_LEVELS = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]   # ng/mL in the vial; the note's curves run to 100
PUB_MYCO_LEVELS = [round(x * 0.02, 6) for x in PUB_PEST_LEVELS]
# Serial calibration stocks at 10x the vial level: (level, uL taken from the level above, uL solvent).
PUB_SERIAL = [(7, 500, 500), (6, 500, 500), (5, 400, 600), (4, 500, 500), (3, 200, 800), (2, 500, 500),
              (1, 200, 800)]
# QC spiking stocks: (sample id, uL pesticide stock, uL mycotoxin master, uL solvent); 100 uL into 1 g.
PUB_QC = [("LCS", 60, 200, 740), ("ICV", 200, 500, 300), ("CCV 1", 400, 400, 200), ("LOQ", 25, 25, 950)]
PUB_QC_SPIKE_UL = 100.0
PUB_PEST_STOCK, PUB_MYCO_MASTER = 10000.0, 200.0            # ng/mL
PT_SAMPLE = {"Bifenazate": 310.0, "Myclobutanil": 145.0, "Imidacloprid": 520.0, "Spinosyn A": 160.0 * 0.85,
             "Spinosyn D": 160.0 * 0.15, "Pyridaben": 95.0, "Acequinocyl": 400.0, "Aflatoxin B1": 4.0,
             "Ochratoxin A": 12.0, "Piperonyl butoxide": 1200.0}


def public_recipe(targets, frac, istds) -> dict:
    steps = []

    def step(op, vessel, note="", **kw):
        for k in ("volume_ul", "mass_g"):
            if k in kw:
                kw[k] = {"nominal": kw[k]}
        steps.append({"op": op, "vessel": vessel, **kw, **({"note": note} if note else {})})

    pest = [n for n in targets if n not in MYCOTOXINS]
    afla = [n for n in targets if n.startswith("Aflatoxin")]
    step("stock", "Pesticide mix (100 ug/mL)", volume_ul=1000.0,
         contents_ng_ul={n: 100.0 * frac.get(n, 1.0) for n in pest},
         note="The note's pesticide mixes (LGC, 100 ug/mL), drawn as one vial")
    step("transfer", "Pesticide stock 10 ppm", source="Pesticide mix (100 ug/mL)", volume_ul=100.0)
    step("solvent", "Pesticide stock 10 ppm", volume_ul=900.0, note="LC/MS-grade acetonitrile")
    step("stock", "Aflatoxin mix (100 ug/mL)", volume_ul=1000.0, contents_ng_ul={n: 100.0 for n in afla},
         note="B1, B2, G1, G2 (LGC, 100 ug/mL)")
    step("stock", "Ochratoxin A (2 ug/mL)", volume_ul=1000.0, contents_ng_ul={"Ochratoxin A": 2.0})
    step("transfer", "Aflatoxin intermediate 1 ppm", source="Aflatoxin mix (100 ug/mL)", volume_ul=10.0)
    step("solvent", "Aflatoxin intermediate 1 ppm", volume_ul=990.0)
    step("transfer", "Mycotoxin master 0.2 ppm", source="Aflatoxin intermediate 1 ppm", volume_ul=400.0)
    step("transfer", "Mycotoxin master 0.2 ppm", source="Ochratoxin A (2 ug/mL)", volume_ul=200.0)
    step("solvent", "Mycotoxin master 0.2 ppm", volume_ul=1400.0)
    # Calibration stocks at 10x the vial level: Cal 8 = 1000 ng/mL pesticides, 20 ng/mL mycotoxins.
    step("transfer", "Cal 8 stock", source="Pesticide stock 10 ppm", volume_ul=100.0)
    step("transfer", "Cal 8 stock", source="Mycotoxin master 0.2 ppm", volume_ul=100.0)
    step("solvent", "Cal 8 stock", volume_ul=800.0)
    prev = "Cal 8 stock"
    for lvl, take, add in PUB_SERIAL:
        step("transfer", f"Cal {lvl} stock", source=prev, volume_ul=float(take))
        step("solvent", f"Cal {lvl} stock", volume_ul=float(add))
        prev = f"Cal {lvl} stock"
    for sid, p_ul, m_ul, s_ul in PUB_QC:
        step("transfer", f"{sid} stock", source="Pesticide stock 10 ppm", volume_ul=float(p_ul))
        step("transfer", f"{sid} stock", source="Mycotoxin master 0.2 ppm", volume_ul=float(m_ul))
        step("solvent", f"{sid} stock", volume_ul=float(s_ul))
    for n in istds:
        step("stock", f"{n} (100 ug/mL)", volume_ul=1000.0, contents_ng_ul={n: 100.0})
        step("transfer", "IS master mix", source=f"{n} (100 ug/mL)", volume_ul=10.0)
    step("solvent", "IS master mix", volume_ul=10000.0 - 10.0 * len(istds),
         note="in the diluent, 25/75 mobile phase A / methanol")

    def extract(tube: str, spike_ul: float = 0.0):
        step("solvent", tube, volume_ul=15000.0, note="15 mL acetonitrile")
        step("extract", tube, note="shake 5 min, centrifuge, C18 SPE by gravity")
        step("solvent", tube, volume_ul=10000.0 - spike_ul, note="two 5 mL rinses through the SPE, made to 25 mL")

    def to_vial(vial: str, tube: str, cal_stock: str | None = None):
        step("transfer", vial, source=tube, volume_ul=PUB_EXTRACT_UL, note="cleaned extract")
        step("solvent", vial, volume_ul=PUB_DILUENT_UL - (PUB_CAL_UL if cal_stock else 0.0),
             note="25/75 mobile phase A / methanol")
        if cal_stock:
            step("transfer", vial, source=cal_stock, volume_ul=PUB_CAL_UL)
        step("transfer", vial, source="IS master mix", volume_ul=PUB_IS_UL)
        step("inject", vial.replace("Vial: ", "Column: "), source=vial, volume_ul=10.0)

    # Matrix-matched calibrators: blank flower taken through the preparation, spiked in the final dilution.
    step("sample", "Tube: Blank flower for calibrators", mass_g=PUB_SAMPLE_G, note="blank cannabis flower, ground")
    extract("Tube: Blank flower for calibrators")
    for k in range(1, len(PUB_PEST_LEVELS) + 1):
        to_vial(f"Vial: Cal {k}", "Tube: Blank flower for calibrators", f"Cal {k} stock")
    for sid in ("Matrix blank", *[q[0] for q in PUB_QC]):
        tube = f"Tube: {sid}"
        step("sample", tube, mass_g=PUB_SAMPLE_G, note="blank cannabis flower, ground")
        spike = PUB_QC_SPIKE_UL if sid != "Matrix blank" else 0.0
        if spike:
            step("transfer", tube, source=f"{sid} stock", volume_ul=spike, note="spiked before extraction")
        extract(tube, spike)
        to_vial(f"Vial: {sid}", tube)
    step("sample", "Tube: PT sample", mass_g=PUB_SAMPLE_G, sample_ug_kg=PT_SAMPLE, note="ground flower")
    extract("Tube: PT sample")
    to_vial("Vial: PT sample", "Tube: PT sample")
    return {"name": "Agilent 5994-1734EN preparation, step by step (stocks and QCs assumed)",
            "provenance": "assumed", "steps": steps}


def public_method() -> Method:
    ag = pd.read_csv(REFERENCE / "agilent_5994-1734en_transitions.csv")
    mn = pd.read_csv(REFERENCE / "mn_ocm_cta_v2_analytes.csv")
    ag["compound"] = ag.compound.map(lambda n: APPNOTE_NAMES.get(n, n))
    regulated = set(mn.analyte)
    keep = [n for n in dict.fromkeys(ag.compound) if REGULATED.get(n, n) in regulated]
    tx = ag[ag.compound.isin(keep)].copy()
    rts = tx[tx.role == "quantifier"].set_index("compound").retention_time_min

    rows = []
    for name, parent, nd, kept, _ in LABELLED:
        q = tx[(tx.compound == parent) & (tx.role == "quantifier")].iloc[0]
        rows.append({**q.to_dict(), "compound": name,
                     "precursor_mz": round(q.precursor_mz + nd * D_SHIFT, 4),
                     "product_mz": round(q.product_mz + (nd * D_SHIFT if kept else 0.0), 4),
                     "retention_time_min": round(q.retention_time_min - 0.01, 3)})
    tx = pd.concat([tx, pd.DataFrame(rows)], ignore_index=True)
    istds = [r[0] for r in LABELLED]
    is_rt = {n: rts[p] - 0.01 for n, p, *_ in LABELLED}
    pol = dict(tx[tx.role == "quantifier"][["compound", "polarity"]].itertuples(index=False))
    pairs = {}
    for n in keep:
        same = [i for i in istds if pol[i] == pol[n]] or istds
        pairs[n] = min(same, key=lambda i: (abs(is_rt[i] - rts[n]), i))
    targets = keep
    frac = {n: f for n, f in FRACTIONS.items() if n in targets}
    missing = sorted(regulated - {REGULATED.get(n, n) for n in targets})

    is_mix = PUB_IS_VIAL_NG_ML * PUB_VIAL_UL / PUB_IS_UL / 1000.0         # ng/uL
    df_total = PUB_EXTRACT_ML / PUB_SAMPLE_G * PUB_VIAL_UL / PUB_EXTRACT_UL
    samples = [
        {"id": "Reagent blank", "matrix": "none", "kind": "blank", "add_istd": False},
        {"id": "Matrix blank", "matrix": "blank cannabis flower", "kind": "blank"},
    ]
    for sid, p_ul, m_ul, s_ul in PUB_QC:
        vol = p_ul + m_ul + s_ul
        pest_ng = PUB_PEST_STOCK * p_ul / vol * PUB_QC_SPIKE_UL / 1000.0 / PUB_SAMPLE_G
        myco_ng = PUB_MYCO_MASTER * m_ul / vol * PUB_QC_SPIKE_UL / 1000.0 / PUB_SAMPLE_G
        samples.append({"id": sid, "matrix": "blank cannabis flower", "kind": "qc",
                        "spiked_from_calibration_stock": True,
                        "true_concentration_ug_kg": qc_amounts(targets, frac, pest_ng, myco_ng)})
    samples.insert(4, {"id": "Flower A", "matrix": "cannabis flower", "kind": "sample", "true_concentration_ug_kg": {}})
    samples.insert(5, {"id": "PT sample", "matrix": "cannabis flower", "kind": "sample",
                       "true_concentration_ug_kg": PT_SAMPLE})

    info = {
        "name": "Agilent application note 5994-1734EN on the Minnesota OCM list",
        "short": "the application note method (Agilent 5994-1734EN)",
        "description": "Pesticides and mycotoxins in cannabis flower on a 1260 Infinity II LC and an Ultivo triple "
                       "quadrupole: transitions, retention times, column, gradient and sample preparation from "
                       "Agilent's published application note, restricted to the Minnesota OCM list. Internal "
                       "standards, calibration and QC levels are assumed (the note gives none).",
        "sources": ["agilent_5994_1734en", "mn_ocm_cta_v2"],
        "provenance": [
            {"input": "Column", "value": "Poroshell 120 Phenyl-Hexyl 100 x 3.0 mm, 2.7 um, 55 C",
             "provenance": "measured"},
            {"input": "Transitions, retention times and windows", "value": "application note, Appendix A (Ultivo)",
             "provenance": "measured"},
            {"input": "Gradient and flow", "value": f"application note, {PUB_FLOW} mL/min, {PUB_RUN:g} min",
             "provenance": "measured"},
            {"input": "Dwell volume", "value": f"{DWELL_ML} mL", "provenance": "assumed"},
            {"input": "Retention parameters (log kw)",
             "value": "fitted to the note's retention times; S from molecular weight", "provenance": "fitted"},
            {"input": "Cycle, delay, polarity switch", "value": "500 ms cycle; delay and switch of an Ultivo",
             "provenance": "assumed"},
            {"input": "Detector gain, response factors, ion ratios", "value": "placeholders", "provenance": "assumed"},
            {"input": "Sample preparation", "value": "1 g, 25 mL, 50 uL in 500 uL (250-fold), as in the note",
             "provenance": "measured"},
            {"input": "Internal standards", "value": f"{len(istds)} labelled analogues, {PUB_IS_VIAL_NG_ML:g} ng/mL "
                                                     "in the vial (the note uses none)", "provenance": "assumed"},
            {"input": "Calibration and QC levels", "value": "0.1-100 ng/mL in the vial; QCs at the note's 60 / 4 ppb "
                                                            "recovery spike and three more", "provenance": "assumed"},
            {"input": "Matrix profile", "value": "synthetic", "provenance": "assumed"},
        ],
        "remarks": {},
    }
    return Method(
        out=EXAMPLES, info=info, column=PUB_COLUMN, flow=PUB_FLOW, gradient=PUB_GRADIENT, run_time=PUB_RUN, ph=3.0,
        lc_name="Agilent 5994-1734EN: Poroshell 120 Phenyl-Hexyl, 5 mM ammonium formate / 0.1% formic acid in "
                "water and methanol",
        tx=tx, istds=istds, pairs=pairs, frac=frac,
        measured_rt={n: float(r) for n, r in rts.items() if n in targets}, rt_label="application note",
        acquisition={
            "provenance": "assumed", "cycle_time_ms": 500.0, "inter_scan_delay_ms": 1.09, "polarity_switch_ms": 22.3,
            "min_points_per_peak": 10, "min_dwell_ms": 1.0, "gain_area_per_ion": 0.05, "background_rate": 50.0,
            "injection_rsd": 0.02, "point_rsd": 0.01, "detector_dead_time_s": None,
        },
        prep={
            "name": "Agilent 5994-1734EN: 1 g flower, 25 mL MeCN through C18 SPE, 50 uL extract in 500 uL",
            "provenance": "measured",
            "sample_mass_g": {"nominal": PUB_SAMPLE_G},
            "extraction_volume_ml": {"nominal": PUB_EXTRACT_ML},
            "dilution_factor": {"nominal": PUB_VIAL_UL / PUB_EXTRACT_UL},
            "injection_volume_ul": {"nominal": 10.0},
            "istd_spike_ul": {"nominal": PUB_IS_UL},
            "istd_mix_ng_ul": {n: round(is_mix, 8) for n in istds},
            "istd_stage": "in_vial",
            "vial_volume_ml": {"nominal": PUB_VIAL_UL / 1000.0},
            "recovery": {},
        },
        calibration={
            "matrix_matched": True, "preparation": "vial", "use_internal_standard": True,
            "model": "linear", "weighting": "1/x", "origin": "ignore",
            "istd_spike_ul": {"nominal": PUB_IS_UL},
            "stock_concentration_ng_ml": {"nominal": PUB_PEST_STOCK},
        },
        levels=(PUB_PEST_LEVELS, PUB_MYCO_LEVELS), samples=samples,
        recipe=public_recipe(targets, frac, istds),
        acceptance={
            "source": "SANTE/11312/2021 (identification, recovery, precision); R2 from the application note",
            "calibration_r2_min": 0.99,
            "calibration_point_recovery_pct": [80, 120],
            "qc_recovery_pct": [70, 120],
            "qc_recovery_pct_low_rsd": [60, 140],
            "istd_recovery_pct": [50, 150],
            "ion_ratio_tolerance_rel": 0.30,
            "precision_rsd_max_pct": 20,
            "duplicate_rpd_max_pct": 20,
        },
        scouting={f"g{f}x": [(0, 30), (1.0, 30), (1.0 + 1.0 * f, 75), (1.0 + 7.0 * f, 96), (1.0 + 8.0 * f, 100),
                             (1.5 + 8.0 * f, 100)] for f in (1, 2, 3)},
        suppression=(8.3, 0.8),
        gaps="# Minnesota analytes the application note does not cover\n\n"
             + "".join(f"- {n}\n" for n in missing)
             + "\nThe note's list is for other states; these have no public transitions in this example. "
               "A lab input set can add them.\n",
        flags=[f"overall dilution {df_total:g} mL/g; internal standard pairs: "
               + ", ".join(f"{i} x{sum(v == i for v in pairs.values())}" for i in istds)],
    )


def main() -> None:
    write_set(public_method())


if __name__ == "__main__":
    main()
