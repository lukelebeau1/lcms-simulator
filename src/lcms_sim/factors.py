"""Loss factors: the named things that move a transfer or an instrument stage off 100%.

A factor is one object everywhere. The Follow page draws its control from `kind`, `apply` turns the settings
into the trace's actual run, and the fault lab will sweep the same objects. Bench factors act on one transfer
or injection step at a time, so their setting key carries the step index ("12:unmixed"); stage factors act
once ("q1_share"). A transfer's efficiency is the product of its factor multipliers, 1 for a perfect transfer;
the ledger moves that share of the nominal amount and leaves the rest in the source.

Scope per factor, the troubleshooter's first question: "uniform" when the loss is the same share for every
compound (a volume, a drift, a multiplier setting), so it looks like a dilution and the internal standard cancels
it; "compound" when the share depends on the compound's chemistry or position (solubility, surface affinity,
pKa, retention time, m/z, ion rate), so some compounds move and others do not. `varies_with` names what the
loss follows. The classification is of the effect on the result, not of the cause: a dwell volume change shifts
every retention time by the same amount but clips early eluters and narrow windows first. Compound-scoped
bench factors are set per compound ("12:precipitated@Bifenthrin") and the ledger charges them to that compound
only; uniform ones ("12:pipette") move every compound in the vessel.

Provenance per factor: "assumed" for a plain multiplier with a default picked to be visible, "model" where the
simulator computes the effect from the setting (retention, width, schedule, dead time), "literature, general"
where a class rule from the literature sets the default.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from lcms_sim.faults.library import LIBRARY, Scenario
from lcms_sim.io.loaders import InputSet
from lcms_sim.io.schemas import PrepRecipe
from lcms_sim.pipeline import RunConditions
from lcms_sim.quant.ledger import with_actual, with_efficiency
from lcms_sim.trace import TYPICAL_DOWNSTREAM, Actual, spray_fraction

Kind = Literal["check", "number", "volume"]
Where = Literal["transfer", "loop", "column", "spray", "capillary", "q1", "cell", "q3", "multiplier", "counting"]
Scope = Literal["uniform", "compound"]


@dataclass(frozen=True)
class Factor:
    key: str
    name: str
    where: Where
    kind: Kind                   # check: on/off with `default` as the share it uses; number: a value; volume: the step's actual uL
    default: float
    unit: str = ""
    low: float = 0.0
    high: float = 1.0
    step: float = 0.01
    theory: str = ""
    provenance: str = "assumed"
    scope: Scope = "uniform"     # uniform: the same share for every compound; compound: depends on the compound
    varies_with: str = ""        # compound scope: the property the loss follows
    sources: tuple[str, ...] = ()  # keys into data/reference/references.yaml; empty = assumed, no literature behind it

    def __post_init__(self) -> None:
        if (self.scope == "compound") != bool(self.varies_with):
            raise ValueError(f"{self.key}: a compound-scoped factor names what it varies with, a uniform one does not")

    @property
    def scope_text(self) -> str:
        if self.scope == "uniform":
            return "Same for every compound: looks like a dilution or a gain change."
        return f"Compound-specific: follows {self.varies_with}."

    def multiplier(self, setting: float | None) -> float:
        """Share of the nominal amount a bench transfer carries because of this factor (1 = no effect)."""
        if setting is None:
            return 1.0
        if self.key == "unmixed":
            return float(setting)
        if self.key == "precipitated":
            return 1.0 - float(setting)
        if self.key == "adsorbed":
            return 1.0 - float(setting) / 100.0
        return 1.0


BENCH: tuple[Factor, ...] = (
    Factor("pipette", "Pipette setting and delivery", "transfer", "volume", 0.0, "uL actually delivered",
           theory="The ledger moves volume / source volume of every compound. A mis-set or badly used pipette "
                  "moves a different volume: the amount scales with it, and the source keeps the difference. "
                  "Viscous or volatile solvents (acetonitrile wets the tip and drips) deliver less than the "
                  "setting unless reverse pipetting is used.",
           provenance="measured: the step's actual volume", scope="uniform", sources=("mettler_pipetting",)),
    Factor("unmixed", "Not homogeneous: the last component is layered", "transfer", "check", 0.5,
           "share of the nominal concentration at the tip", 0.0, 2.0, 0.05,
           theory="A component added last sits where its solvent took it until the vial is mixed. An aqueous "
                  "chlormequat standard is denser than the acetonitrile stock and sinks, so a draw from the top "
                  "is low in chlormequat and a draw from the bottom is high; a methanol or acetonitrile standard "
                  "layers little and only diffuses slowly. The number is the share of the nominal concentration "
                  "the tip finds: below 1 low, above 1 high. Every other compound is unaffected, which is what "
                  "makes the fault diagnosable: one compound off, at the draws made before mixing. The solvent on "
                  "the chlormequat label settles the sign.",
           provenance="assumed: share; literature, general: aqueous layers sink in acetonitrile",
           scope="compound", varies_with="the solvent and density of the component's own standard against the stock",
           sources=("pubchem_acetonitrile",)),
    Factor("precipitated", "Precipitated: cold vial, not warmed and vortexed", "transfer", "check", 0.3,
           "share still out of solution when drawn", 0.0, 1.0, 0.05,
           theory="Low-solubility compounds (pyrethroids such as bifenthrin, cypermethrin and permethrin, and "
                  "other large hydrophobic molecules) crystallise in a cold acetonitrile stock. A vial drawn "
                  "straight from the freezer carries only what has redissolved; the crystals stay behind and "
                  "redissolve later, so a later draw from the same vial reads high. Warming to room temperature "
                  "and vortexing before every draw is the usual rule.",
           provenance="assumed: share; literature, general: solubility falls with temperature",
           scope="compound", varies_with="solubility in the stock solvent at its temperature (large hydrophobic molecules first)",
           sources=("restek_2021",)),
    Factor("adsorbed", "Adsorbed to the walls or the tip", "transfer", "number", 0.0, "% of the amount lost", 0.0,
           50.0, 1.0,
           theory="Some compounds stick to container walls, from dilute and mostly aqueous solutions most of all. "
                  "Basic compounds bind silanol groups on glass by ion exchange, and hydrophobic compounds adsorb to "
                  "glass and polypropylene alike; salts weaken the ionic binding, organic solvent the hydrophobic. "
                  "Pyrethroids in water lose up to half their amount to container walls (glass least in one study, "
                  "and 58 to 72% remaining in glass in another), and ordinary glass autosampler vials lost "
                  "bifenthrin and permethrin where polypropylene and high-purity borosilicate did not. Quaternary "
                  "ammoniums (chlormequat, mepiquat) are permanent cations, so silanol binding is the expected "
                  "mechanism, and polypropylene vials are the practice for them; a direct measurement for them was "
                  "not found. The loss is a share of the amount, so it looks like a stock weaker than labelled for "
                  "those compounds only. In acetonitrile or methanol stocks the hydrophobic loss is far smaller than "
                  "in water.",
           provenance="assumed: share; literature: mechanism and which classes (quats by inference)",
           scope="compound", varies_with="surface affinity (basic and permanently charged compounds to glass silanols, "
                                        "hydrophobic compounds to glass and plastic) and dilution",
           sources=("murakoshi_shimadzu", "hladik_2009", "lee_2002", "thermo_ai10382", "waters_chlormequat_2024")),
)

STAGE: tuple[Factor, ...] = (
    Factor("carryover", "Carryover from the previous injection", "loop", "number", 0.0,
           "share of the previous injection's amount", 0.0, 0.05, 0.001,
           theory="What the needle, loop and column keep from one injection reaches the next. A weak needle wash "
                  "or a hydrophobic compound makes it worse. It adds amount, so it shows as a gain, largest in a "
                  "blank after a high standard.", provenance="model: run conditions",
           scope="compound", varies_with="the previous injection's level of each compound, and its hydrophobicity",
           sources=("dolan_2006_carryover",)),
    Factor("dwell_volume", "Dwell volume or proportioning change", "column", "number", 1.0,
           "x the method's dwell time", 0.5, 3.0, 0.05,
           theory="The gradient reaches the column later when the dwell volume grows (a mixer, tubing or the "
                  "proportioning valve). Every retention time shifts late by about the same amount; peaks move "
                  "toward the edge of their acquisition windows and the share inside the window falls.",
           provenance="model: retention and window",
           scope="compound", varies_with="retention time and window width: early eluters and narrow windows clip first",
           sources=("dolan_2006_dwell", "snyder_dolan_2007")),
    Factor("organic", "Mobile phase B stronger or weaker than set", "column", "number", 0.0, "%B offset", -10.0,
           10.0, 0.5,
           theory="A %B error at every point of the gradient shifts every analyte: stronger B elutes earlier, "
                  "weaker later, by an amount set by each compound's S. The window share falls when the shift "
                  "nears the half-window.", provenance="model: retention and window",
           scope="compound", varies_with="each compound's S, its sensitivity to %B", sources=("snyder_dolan_2007",)),
    Factor("ph", "Mobile phase pH off", "column", "number", 0.0, "pH units", -1.0, 1.0, 0.05,
           theory="Only ionisable analytes move: their retention follows the share of neutral molecules, "
                  "log k = log[(1 + 10^d r) / (1 + r)] with r the ionised-to-neutral ratio at the pH. Neutral "
                  "compounds stay put, which is the fingerprint of a pH error.", provenance="model: retention",
           scope="compound", varies_with="pKa: only ionisable compounds move",
           sources=("horvath_1977", "snyder_kirkland_dolan_2010")),
    Factor("column_age", "Column integrity: plate number", "column", "number", 1.0, "x the method's plate number",
           0.2, 1.0, 0.05,
           theory="A void at the inlet, contamination or a collapsed frit broadens every peak. Width grows as "
                  "1 / sqrt(N): with a quarter of the plates, peaks are twice as wide, the apex height halves, "
                  "the points across a peak double, and tails run out of the window.",
           provenance="model: width and window", scope="uniform", sources=("snyder_kirkland_dolan_2010",)),
    Factor("spray_share", "Spray and inlet: share ionised and admitted", "spray", "number", 1.0,
           "x the nominal spray share", 0.1, 2.0, 0.05,
           theory="Of the molecules eluting, roughly 1 in 10,000 to 1 in 100 becomes a gas-phase ion that enters "
                  "the capillary at 0.6 mL/min; the rest leaves in droplets that never shed their solvent, or as "
                  "neutrals, or misses the inlet. Nebuliser pressure, drying gas temperature and flow, capillary "
                  "voltage and spray position all move it. The nominal value is back-computed from the response "
                  "factor, so this is a multiplier on it until a tune report gives transmission figures.",
           provenance="assumed: multiplier; literature: the order of magnitude", scope="uniform",
           sources=("page_2007", "kebarle_verkerk_2009")),
    Factor("matrix_scale", "Matrix suppression, scaled", "spray", "number", 1.0,
           "x the calibration matrix's suppression", 0.0, 3.0, 0.1,
           theory="Co-eluting matrix takes charge and surface in the droplets. The infusion profile gives the "
                  "suppression at each retention time for the calibration matrix; a sample with more matrix "
                  "scales it up, a cleaner one down. The loss depends on where the analyte elutes.",
           provenance="model: matrix profile",
           scope="compound", varies_with="retention time: what co-elutes with the compound",
           sources=("king_2000", "bonfiglio_1999", "matuszewski_2003")),
    Factor("drift", "Sensitivity drift per injection", "spray", "number", 0.0, "relative change per injection",
           -0.05, 0.05, 0.005,
           theory="Source fouling and tune drift change the sensitivity through the run. The effect grows with the "
                  "injection's position in the sequence; the internal standard's area trends with it.",
           provenance="model: run conditions", scope="uniform"),
    Factor("q1_share", "Q1 transmission of the precursor", "q1", "number", 0.3, "share passing", 0.05, 0.8, 0.05,
           theory="At unit resolution Q1 passes 20 to 60% of the precursor ions: the pass band is narrow enough "
                  "to reject neighbouring m/z and so clips the edges of the ion's own stability region. Wider "
                  "windows pass more but let isobaric interferences through.",
           provenance="assumed: typical share; literature: the transmission-resolution trade", scope="compound",
           varies_with="m/z: transmission drifts with mass", sources=("dawson_1995",)),
    Factor("cell_share", "Collision cell yield of the product", "cell", "number", 0.3, "share becoming the product",
           0.05, 0.8, 0.05,
           theory="At the set collision energy a share of the precursors breaks to the chosen product; the rest "
                  "stays intact or becomes other fragments. The breakdown curve peaks at an energy that depends on "
                  "the molecule and its size; a qualifier's share is its ion ratio times this.",
           provenance="assumed: typical share; literature: breakdown curves", scope="compound",
           varies_with="the molecule's breakdown curve at the set energy",
           sources=("hoffmann_stroobant_2007", "douglas_french_1992")),
    Factor("q3_share", "Q3 transmission of the product", "q3", "number", 0.3, "share passing", 0.05, 0.8, 0.05,
           theory="As Q1, for the product ion.", provenance="assumed: typical share; literature: the trade",
           scope="compound", varies_with="m/z of the product ion", sources=("dawson_1995",)),
    Factor("multiplier_share", "Multiplier registers the ion", "multiplier", "number", 0.8, "share registered",
           0.3, 1.0, 0.05,
           theory="An ion striking the first dynode starts a cascade with a probability that falls as the "
                  "multiplier ages or its voltage is set low; the gain setting then raises the area per registered "
                  "ion, not the share registered.", provenance="assumed: typical share; literature: how multipliers age",
           scope="uniform", sources=("hoffmann_stroobant_2007",)),
    Factor("dead_time_us", "Detector dead time", "multiplier", "number", 0.0, "us", 0.0, 200.0, 5.0,
           theory="After each registered ion the detector is blind for the dead time. At high ion rates it misses "
                  "a growing share: observed = true / (1 + true x dead time), integrated over the peak, so the top "
                  "of the curve bends and high samples read low.", provenance="model: integrated over the peak",
           scope="compound", varies_with="ion rate: the strongest responders and the highest levels first",
           sources=("knoll_2010",)),
    Factor("extra_transitions", "Extra concurrent transitions", "counting", "number", 0.0,
           "transitions added across the run", 0.0, 100.0, 5.0,
           theory="Each transition is counted for its dwell once per cycle, dwell = cycle / n - delay, so the share "
                  "counted is about 1 / n. Every extra transition scheduled at the same time (wider windows, more "
                  "compounds eluting together, a second method merged in) shortens every dwell. Fewer ions are "
                  "counted per peak; the software scales the count back to a rate, so the area is the same and "
                  "the scatter grows: precision worsens at low level, the mean does not move.",
           provenance="model: dMRM schedule", scope="uniform",
           sources=("agilent_5990_3595", "waters_720001168", "zeng_2023", "sciex_scheduled_mrm")),
)

FACTORS: dict[str, Factor] = {f.key: f for f in BENCH + STAGE}
STAGES: tuple[tuple[str, str], ...] = (("loop", "Injection loop"), ("column", "Column"), ("spray", "Spray"),
                                       ("capillary", "Capillary"), ("q1", "Q1"), ("cell", "Collision cell"),
                                       ("q3", "Q3"), ("multiplier", "Multiplier"), ("counting", "Counting"))


def bench_key(step: int, factor: str, compound: str | None = None) -> str:
    """Setting key of a bench factor on one step: "12:pipette", or "12:precipitated@Bifenthrin" for a
    compound-scoped factor set on one compound."""
    return f"{step}:{factor}" + (f"@{compound}" if compound else "")


def parse_key(key: str) -> tuple[int | None, str, str | None]:
    """(step, factor key, compound) of a setting key; step and compound are None where the key has none."""
    head, _, compound = key.partition("@")
    step, _, factor = head.rpartition(":")
    return (int(step) if step else None), factor, (compound or None)


def factor_of(key: str) -> Factor | None:
    return FACTORS.get(parse_key(key)[1])


def transfer_multipliers(settings: Mapping[str, float], step: int,
                         compound: str | None = None) -> list[tuple[Factor, float]]:
    """(factor, multiplier) for the factors set on one transfer step that act on `compound`, other than the
    volume: the uniform ones, and the compound-scoped ones set on that compound or set with no compound named."""
    out = []
    for f in BENCH:
        if f.kind == "volume":
            continue
        v = settings.get(bench_key(step, f.key))
        if v is None and f.scope == "compound" and compound:
            v = settings.get(bench_key(step, f.key, compound))
        if v is not None:
            out.append((f, f.multiplier(v)))
    return out


def transfer_efficiency(settings: Mapping[str, float], step: int, compound: str | None = None) -> float:
    e = 1.0
    for _, m in transfer_multipliers(settings, step, compound):
        e *= m
    return e


def bench_efficiencies(settings: Mapping[str, float],
                       compound: str) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    """The ledger's efficiencies from the bench settings: per step the uniform share (every compound), and per
    step and compound the compound's own share. A compound-scoped key with no compound named is charged to
    `compound`, the one being followed."""
    uniform: dict[int, float] = {}
    own: dict[int, dict[str, float]] = {}
    for key, v in settings.items():
        step, fk, c = parse_key(key)
        f = FACTORS.get(fk)
        if step is None or f is None or f.kind == "volume":
            continue
        m = f.multiplier(v)
        if f.scope == "uniform":
            uniform[step] = uniform.get(step, 1.0) * m
        else:
            per = own.setdefault(step, {})
            c = c or compound
            per[c] = per.get(c, 1.0) * m
    return ({k: e for k, e in uniform.items() if e != 1.0},
            {k: {c: e for c, e in per.items() if e != 1.0} for k, per in own.items()
             if any(e != 1.0 for e in per.values())})


def is_own_label(compound: str, istd: str | None) -> bool:
    """Whether the IS is an isotope-labelled form of the compound itself, by name ("Myclobutanil-D4" for
    Myclobutanil)."""
    if not istd:
        return False
    c, i = compound.lower().replace(" ", ""), istd.lower().replace(" ", "")
    return i.startswith(c) and i != c


def istd_note(f: Factor, istd: str | None, present: bool, own_label: bool) -> str:
    """Whether the internal standard cancels this factor's loss in the reported ratio, at a stop where the IS is
    `present` (in the source vessel at the draw, or on the instrument) and is or is not the compound's own
    labelled analogue."""
    if istd is None:
        return "No internal standard: the loss goes straight to the result."
    if not present:
        return f"{istd} is not in this vessel yet: the loss goes straight to the result."
    if f.scope == "uniform":
        return f"{istd} is here and loses the same share, so the ratio cancels it."
    if own_label:
        return (f"{istd} is the compound's own labelled analogue and shares its chemistry, so the ratio cancels "
                "most of it.")
    return f"{istd} is a surrogate with other chemistry, so the loss reaches the result."


def apply(inputs: InputSet, recipe: PrepRecipe, settings: Mapping[str, float], compound: str,
          injection_index: int = 0, previous_injection: str | None = None) -> Actual:
    """The actual run the settings describe: an edited recipe, a scenario for the column, run conditions, shares."""
    volumes: dict[int, float] = {}
    for k, s in enumerate(recipe.steps):
        if s.op not in ("transfer", "inject"):
            continue
        v = settings.get(bench_key(k, "pipette"))
        if v is not None and v != s.volume_ul.nominal:
            volumes[k] = float(v)
    uniform, own = bench_efficiencies(settings, compound)
    r = with_efficiency(with_efficiency(with_actual(recipe, volumes), uniform), own)

    sc = Scenario.baseline(inputs)
    if settings.get("dwell_volume", 1.0) != 1.0:
        sc = LIBRARY["dwell_volume"].apply(sc, float(settings["dwell_volume"]))
    if settings.get("organic", 0.0):
        sc = LIBRARY["organic_error"].apply(sc, float(settings["organic"]) / 100.0)
    if settings.get("ph", 0.0):
        sc = LIBRARY["ph_error"].apply(sc, float(settings["ph"]))
    if settings.get("column_age", 1.0) != 1.0:
        system = sc.inputs.system.model_copy(update={"plate_number": sc.inputs.system.plate_number
                                                     * float(settings["column_age"])})
        sc = dataclasses.replace(sc, inputs=dataclasses.replace(sc.inputs, system=system))
    acq_upd = {}
    if settings.get("dead_time_us", 0.0):
        acq_upd["detector_dead_time_s"] = float(settings["dead_time_us"]) * 1e-6
    if settings.get("extra_transitions", 0.0):
        sc = LIBRARY["too_many_transitions"].apply(sc, float(settings["extra_transitions"]))
    if acq_upd:
        sc = dataclasses.replace(sc, inputs=dataclasses.replace(sc.inputs,
                                                                acquisition=sc.inputs.acquisition.model_copy(update=acq_upd)))
    run = RunConditions(carryover_fraction=float(settings.get("carryover", 0.0)),
                        drift_per_injection=float(settings.get("drift", 0.0)),
                        mf_scale=({a.name: float(settings["matrix_scale"]) for a in sc.inputs.analytes}
                                  if "matrix_scale" in settings else {}))
    sc = dataclasses.replace(sc, run=run)

    shares: dict[str, float] = {}
    typical = {key: f for (_, f, _), key in zip(TYPICAL_DOWNSTREAM, ("q1", "cell", "q3", "multiplier"))}
    for key in typical:
        v = settings.get(f"{key}_share")
        if v is not None and v != typical[key]:
            shares[key] = float(v)
    m = settings.get("spray_share", 1.0)
    if m != 1.0:
        a = next(x for x in sc.inputs.analytes if x.name == compound)
        shares["spray"] = spray_fraction(a, sc.inputs.acquisition) * float(m)
    return dataclasses.replace(Actual.from_scenario(sc, recipe=r, injection_index=injection_index,
                                                    previous_injection=previous_injection), shares=shares)
