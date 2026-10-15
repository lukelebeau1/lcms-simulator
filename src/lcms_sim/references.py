"""The reference registry and the claims that cite it.

Every positive claim the simulator makes in a theory text, a fault signature, a bench check, a typical value or a
model note cites one or more keys from data/reference/references.yaml. Each entry there records how and when it
was verified (a DOI resolved, an ISBN on the publisher's page, a document's text read). A claim with no adequate
source keeps the label "assumed" instead of a weak citation; `unsourced()` lists those so the gap is visible.

Citing sites: `Factor.sources` (factors.py), `Fault.sources` (faults/library.py), `SeparatingCheck.sources`
(data/examples/checks.yaml), `STAGE_SOURCES` (the typical shares of trace.TYPICAL_DOWNSTREAM), `CLAIMS` (claims
made in page text and docs that are not objects), and `[key]` marks in docs/*.md. `citing_sites()` gathers them
all; tests check that every cited key exists and every entry is cited.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lcms_sim.paths import ROOT

REFERENCES_FILE = ROOT / "data" / "reference" / "references.yaml"
LAB_REFERENCES_FILE = ROOT / "data" / "lab" / "references.yaml"
DOCS = ROOT / "docs"
CITE_MARK = re.compile(r"\[([a-z0-9_]+(?:,\s*[a-z0-9_]+)*)\]")
_INITIALS = re.compile(r"^[A-Z](?:-?[A-Z])*\.?$")
_EDITOR = ("(ed.)", "(eds.)")


def _surname(name: str) -> str:
    """'de Hoffmann E' -> 'de Hoffmann', 'Van Berkel GJ' -> 'Van Berkel', 'Dawson PH (ed.)' -> 'Dawson'."""
    tokens = name.split()
    while len(tokens) > 1 and (tokens[-1] in _EDITOR or _INITIALS.match(tokens[-1])):
        tokens.pop()
    return " ".join(tokens)


@dataclass(frozen=True)
class Reference:
    key: str
    kind: str
    title: str
    authors: str = ""
    source: str = ""
    year: int | None = None
    volume: str = ""
    pages: str = ""
    doi: str = ""
    url: str = ""
    isbn: str = ""
    cite_as: str = ""
    note: str = ""
    verified: str = ""

    @property
    def link(self) -> str:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return self.url

    def short(self) -> str:
        """Kebarle and Verkerk 2009; Page et al. 2007; Dolan 2006; or the cite_as name for documents."""
        if self.cite_as:
            return self.cite_as
        names = [a.strip() for a in self.authors.split(",") if a.strip()]
        surnames = [_surname(n) for n in names if n.split()]
        if not surnames:
            return f"{self.title[:40]} {self.year or ''}".strip()
        if len(surnames) == 1:
            who = surnames[0]
        elif len(surnames) == 2:
            who = f"{surnames[0]} and {surnames[1]}"
        else:
            who = f"{surnames[0]} et al."
        return f"{who} {self.year}" if self.year else who

    def full(self) -> str:
        bits = [self.authors + "." if self.authors else "", f"{self.title}." if self.title else "", self.source]
        if self.year:
            bits.append(str(self.year))
        if self.volume:
            bits.append(f"{self.volume}:{self.pages}" if self.pages else self.volume)
        tail = " ".join(b for b in bits if b).replace("..", ".")
        if self.doi:
            tail += f". doi:{self.doi}"
        elif self.isbn:
            tail += f". ISBN {self.isbn}"
        elif self.url and not self.url.startswith("file:"):
            tail += f". {self.url}"
        return tail


_CACHE: dict[tuple[str, float], dict[str, Reference]] = {}


def load_references(path: str | Path = REFERENCES_FILE) -> dict[str, Reference]:
    """The registry, re-read when the file changes (a long-running app sees new entries without a restart)."""
    path = Path(path)
    stamp = (str(path), path.stat().st_mtime)
    if stamp in _CACHE:
        return _CACHE[stamp]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: dict[str, Reference] = {}
    for item in raw["references"]:
        r = Reference(**{k: ("" if v is None else v) for k, v in item.items()})
        if r.key in out:
            raise ValueError(f"duplicate reference key {r.key}")
        out[r.key] = r
    for k in [k for k in _CACHE if k[0] == str(path)]:
        del _CACHE[k]
    _CACHE[stamp] = out
    return out


def ref(key: str) -> Reference:
    """A public entry, or one from the lab's own registry (data/lab/references.yaml, never version-controlled), which
    a lab input set's method.yaml may cite."""
    refs = load_references()
    if key not in refs and LAB_REFERENCES_FILE.exists():
        refs = load_references(LAB_REFERENCES_FILE)
    if key not in refs:
        raise KeyError(f"unknown reference key {key!r}; add it to {REFERENCES_FILE.name}")
    return refs[key]


def short_cite(keys: Iterable[str]) -> str:
    return "; ".join(ref(k).short() for k in keys)


def markdown_cite(keys: Iterable[str]) -> str:
    """Short citations as markdown links, for the app's captions."""
    out = []
    for k in keys:
        r = ref(k)
        out.append(f"[{r.short()}]({r.link})" if r.link and not r.link.startswith("file:") else r.short())
    return ", ".join(out)


# --- claims that are not objects: typical values and page or doc statements --------------------------------------

STAGE_SOURCES: dict[str, tuple[str, ...]] = {
    "spray": ("page_2007", "kebarle_verkerk_2009"),
    "q1": ("dawson_1995",),
    "cell": ("hoffmann_stroobant_2007", "douglas_french_1992"),
    "q3": ("dawson_1995",),
    "multiplier": ("hoffmann_stroobant_2007",),
}


@dataclass(frozen=True)
class Claim:
    where: str                 # page or module
    text: str                  # the claim, in one sentence
    sources: tuple[str, ...]
    status: str = "sourced"    # sourced | partly | assumed


CLAIMS: tuple[Claim, ...] = (
    Claim("MS: molecules to response; trace typical shares",
          "Of the molecules eluting, roughly 1 in 10,000 to 1 in 100 become ions that enter the analyser: inlet sampling "
          "is high but most analyte is lost to incomplete desolvation.", ("page_2007", "kebarle_verkerk_2009"), "partly"),
    Claim("MS: molecules to response; trace typical shares",
          "Q1 and Q3 pass 20 to 60% of the ions at unit resolution; the figure is a typical value, the trade of "
          "transmission against resolution is the sourced part.", ("dawson_1995",), "partly"),
    Claim("MS: molecules to response; trace typical shares",
          "The collision cell converts 10 to 60% of precursors to the chosen product, depending on the breakdown curve.",
          ("hoffmann_stroobant_2007",), "partly"),
    Claim("MS: molecules to response; trace typical shares",
          "The electron multiplier registers 50 to 90% of arriving ions and the share falls as it ages.",
          ("hoffmann_stroobant_2007",), "partly"),
    Claim("MS: acquisition; docs/ms.md",
          "In dynamic MRM the dwell is the cycle time divided by the concurrent transitions, less the inter-scan delay.",
          ("agilent_5990_3595",)),
    Claim("MS: acquisition; docs/ms.md",
          "Points across a peak: the simulator flags below 10. The 15-point convention, a 7-point result for narrow "
          "peaks and a vendor's 6-8 at half height bracket the threshold; it is a chosen value.",
          ("waters_720001168", "zeng_2023", "sciex_scheduled_mrm", "agilent_5990_3595"), "partly"),
    Claim("MS: acquisition; docs/ms.md",
          "Ion ratio identification tolerance is +-30% relative to the calibration standards' average in the same "
          "sequence; retention time within +-0.1 min.", ("sante_11312_2021",)),
    # MS: acquisition (app/views/ms_acquisition.py, ms/acquisition.py, ms/noise.py, ms/ratios.py, ms/crosstalk.py)
    Claim("MS: acquisition",
          "Dynamic MRM shares a fixed cycle among the transitions whose retention windows are open, so more concurrent "
          "transitions means shorter dwell and fewer ions counted; the expected area is unchanged and only its "
          "precision falls.", ("agilent_5990_3595", "knoll_2010")),
    Claim("MS: acquisition (inside one cycle)",
          "Within a cycle the quadrupoles are set to one precursor > product pair at a time, each for its dwell; ions "
          "of every other eluting compound arrive meanwhile and are not counted, so a transition counts the share "
          "dwell / cycle of the ions it is sent.", ("yost_enke_1978", "agilent_5990_3595")),
    Claim("MS: acquisition",
          "The chance that an ion ratio falls outside the +-30% tolerance is read from the counting-limited RSD with a "
          "normal approximation; real scatter is at least the counting limit.",
          ("sante_11312_2021", "knoll_2010"), "partly"),
    Claim("MS: acquisition",
          "The concentrations in the ion ratio table are a fraction of each analyte's Minnesota action limit in flower.",
          ("mn_ocm_cta_v2",)),
    Claim("MS: acquisition (crosstalk); docs/ms.md",
          "A transition counts any ion that passes Q1 at its precursor setting and Q3 at its product setting, so a "
          "co-eluting compound sharing both within the unit-resolution windows adds to it whatever the dwell; the "
          "crosstalk panel lists the pairs that meet in the loaded method's transition table.",
          ("hoffmann_stroobant_2007", "dawson_1995", "agilent_5994_1734en")),
    Claim("MS: acquisition (crosstalk); docs/ms.md",
          "The Gaussian pass window (0.7 m/z FWHM) and the crosstalk yield, the share of the leaking compound's fragment "
          "formed at the receiving transition's collision energy, are assumed until the leaking compound is injected "
          "alone.", ("dawson_1995",), "assumed"),
    Claim("MS: acquisition (crosstalk)",
          "The bioanalytical guideline requires selectivity for the internal standard as well as the analyte: no "
          "significant response from interfering components at either's retention time (limits 20% of the analyte "
          "response at the LLOQ and 5% of the IS response), and a labelled IS checked for the unlabelled analyte it "
          "contains; injecting the leaking compound alone and reading the receiving transition is that check applied "
          "to a shared product ion.", ("ich_m10_2022",), "partly"),
    Claim("docs/ms.md",
          "Ion counting is Poisson: area RSD ~ 1/sqrt(N) and the ion ratio's counting limit is sqrt(1/N_quant + 1/N_qual).",
          ("knoll_2010",)),
    Claim("docs/ms.md; Fault lab (saturation)",
          "Detector dead time follows the non-paralysable model, observed = true / (1 + true x tau).", ("knoll_2010",)),
    Claim("MS: ion path; MS: how a quadrupole filters; docs/ms.md",
          "Quadrupole stability is the Mathieu diagram with a = 8zeU/(m r0^2 W^2), q = 4zeV/(m r0^2 W^2); the RF-only "
          "cell has a low-mass cutoff and collisions with gas focus ions to the axis.",
          ("dawson_1995", "douglas_french_1992")),
    Claim("MS: ion path (spray close-up)",
          "A droplet at the Rayleigh limit sheds offspring that carry about 2% of its mass and 15% of its charge.",
          ("kebarle_verkerk_2009", "gomez_tang_1994")),
    Claim("docs/lc.md; LC: gradient and retention",
          "LSS retention log k = log kw - S phi, gradient retention by integration with the dwell delay, and the "
          "closed form as the test oracle.", ("snyder_dolan_2007", "snyder_kirkland_dolan_2010")),
    Claim("docs/lc.md",
          "Peak width sigma = t0 (1 + ke) / sqrt(N) with extra-column broadening added in quadrature; the "
          "gradient compression factor G is left at 1 until checked against the source.",
          ("snyder_kirkland_dolan_2010", "snyder_dolan_2007"), "partly"),
    Claim("docs/lc.md; LC: gradient and retention (pH)",
          "An ionisable compound's retention is the ionisation-weighted sum of its neutral and ionised forms' retention.",
          ("horvath_1977", "snyder_kirkland_dolan_2010")),
    Claim("docs/lc.md",
          "S ~ 0.25 MW^0.5 as the default gradient steepness sensitivity.", ("snyder_dolan_2007",), "partly"),
    Claim("docs/quant.md; Matrix overlay",
          "The matrix factor is the response in matrix over the response in solvent, mapped against retention time by "
          "post-column infusion; suppression arises in the droplets.", ("matuszewski_2003", "bonfiglio_1999", "king_2000")),
    Claim("docs/quant.md; Calibration",
          "1/x weighting keeps the low end of the calibration accurate; the right weighting follows how the response's "
          "scatter grows with concentration.", ("almeida_2002", "gu_2014")),
    Claim("docs/quant.md; Follow (IS note)",
          "An internal standard corrects a loss only when it shares it: a labelled analogue of the analyte co-elutes and "
          "shares the chemistry, a surrogate does not.", ("stokvis_2005", "matuszewski_2003")),
    Claim("Calibration (acceptance)",
          "The example's calibration acceptance takes R2 >= 0.99 from the application note and a +-20% window for "
          "back-calculated levels from the residue guidance; a lab input set carries its own (acceptance.yaml).",
          ("agilent_5994_1734en", "sante_11312_2021"), "partly"),
    Claim("Sample to result",
          "With a linear, zero-intercept, noise-free fit the reported / true ratio factors into recovery relative to the "
          "calibrators, the analyte's matrix factor over its internal standard's relative to the calibrators, the stock's "
          "labelled / actual, the IS volume nominal / actual and the steps before the IS; the injection volume cancels "
          "once the IS is in the vial, because the IS shares every step after it is added.",
          ("stokvis_2005", "matuszewski_2003"), "partly"),
    Claim("Sample to result",
          "The default true concentration is the analyte's Minnesota action limit in flower; the example's matrix-matched "
          "calibration in the final extract is the application note's, its internal standard pairs are assumed (the "
          "note uses none).", ("mn_ocm_cta_v2", "agilent_5994_1734en"), "partly"),
    Claim("LC: gradient and retention (dwell volume)",
          "A change of dwell volume shifts every gradient-eluted peak by the change in dwell time VD / F, early peaks "
          "less because they migrate during the isocratic hold; a peak that drifts out of its acquisition window loses "
          "the area outside it.", ("dolan_2006_dwell", "snyder_dolan_2007", "agilent_5990_3595")),
    Claim("Standards and extraction",
          "A QC spiked from the calibration stock moves with the curve, so a stock error is invisible to it; the "
          "guidance's check is a standard's concentration against a second solution prepared independently.",
          ("sante_11312_2021",)),
    Claim("Bench steps",
          "An air-displacement pipette under-delivers volatile or viscous liquids unless reverse pipetting, a pre-rinsed "
          "tip or positive displacement is used; the example's extraction and vial volumes are the application note's, "
          "its stock dilutions assumed.", ("mettler_pipetting", "agilent_5994_1734en"), "partly"),
    Claim("Matrix overlay",
          "The overlay's profile is the ratio of a matrix-blank to a solvent-blank post-column infusion trace; the "
          "bundled profile is synthetic, and the analyte / IS factor ratio is the share of suppression the IS does not "
          "remove.", ("bonfiglio_1999", "matuszewski_2003", "stokvis_2005"), "partly"),
    Claim("Calibration (detector model)",
          "Each calibration level is injected through the detector model: Poisson counting, a per-injection proportional "
          "term, and optional non-paralysable dead time that bends the top of the curve down.", ("knoll_2010",)),
    Claim("docs/retention_chemistry.md",
          "The example method's transitions, energies and retention order come from the Agilent application note; "
          "the regulated list and limits from the Minnesota technical authority.", ("agilent_5994_1734en", "mn_ocm_cta_v2")),
    # LC: inside the column, "Down at the C18 surface" (app/c18_scene.py)
    Claim("LC: inside the column (C18 surface); docs/lc.md",
          "Retention is partitioning of the molecule into the bonded C18 layer; the share of time it spends held is "
          "k / (1 + k).", ("dorsey_dill_1989", "snyder_kirkland_dolan_2010")),
    Claim("LC: inside the column (C18 surface); docs/lc.md",
          "The C18 layer is a disordered brush whose chain tilt and gauche kinks depend on the grafting density; the "
          "drawn density (one chain per 50 A^2, about 3.3 umol/m^2) and tilt (25 deg) are values inside the simulated "
          "ranges, not a measurement.", ("lippa_2005", "rafferty_2008", "rafferty_2007"), "partly"),
    Claim("LC: inside the column (C18 surface); docs/lc.md",
          "Methanol enriches at the chain tops and soaks into the layer while water mostly stays outside.",
          ("rafferty_2011",)),
    Claim("LC: inside the column (C18 surface); docs/lc.md",
          "A held molecule sits with its oily end among the chains and its polar end at the chain tops facing the "
          "solvent.", ("rafferty_2007",), "partly"),
    Claim("LC: inside the column (C18 surface); docs/lc.md",
          "The methanol and water count in the scene is the mole fraction for the %B, from molar volumes of 40.7 and "
          "18.05 mL/mol.", ("pubchem_methanol",)),
    Claim("LC: inside the column (C18 surface); docs/lc.md",
          "Real sorption events last nanoseconds to microseconds and happen millions of times per run; the animation "
          "compresses time and the two-state hop timing is a drawing device.", ("rafferty_2007",), "assumed"),
    # MS: ion path (app/views/ms_ion_path.py, app/spray_cartoon.py, app/ion_stream.py, app/quad_stability.py)
    Claim("MS: ion path",
          "A triple quadrupole measures a transition as spray, Q1 passing the precursor, an RF-only collision cell "
          "breaking it, Q3 passing the product, and an electron multiplier counting; each transition for its dwell.",
          ("yost_enke_1978", "paul_steinwedel_1953", "agilent_5990_3595", "hoffmann_stroobant_2007")),
    Claim("MS: ion path (spray close-up)",
          "The excess charge comes from electrochemistry at the needle; the field pulls the liquid into a Taylor cone "
          "whose jet breaks into charged droplets; the nebuliser gas shears them finer.",
          ("blades_1991", "van_berkel_1995", "taylor_1964", "fenn_1989", "bruins_1987")),
    Claim("MS: ion path (spray close-up)",
          "Solvent leaves and the analyte stays: water and methanol vapour pressures 2.34 and 13.0 kPa and surface "
          "tensions 72.8 and 22.6 mN/m at 20 C; a droplet's area shrinks at a steady rate (the d-squared law, for a "
          "quiescent drop; the gas-flow scalings are illustrative).", ("crc_handbook_97", "langmuir_1918"), "partly"),
    Claim("MS: ion path (spray close-up)",
          "Heated nitrogen flowing against the spray supplies the heat of evaporation and sweeps neutral vapour away "
          "from the inlet while the field draws ions in; the Ultivo carries the Jet Stream source with a heated sheath "
          "gas. The application note runs drying gas 13 L/min at 200 C, sheath 10 L/min at 200 C, nebuliser 55 psi, "
          "capillary 5000 V (positive) and nozzle 500 V.",
          ("yamashita_fenn_1984", "fenn_1989", "agilent_5990_3494", "agilent_5994_1734en")),
    Claim("MS: ion path (spray close-up)",
          "Past the Rayleigh limit a droplet sheds offspring carrying about 2% of its mass and 15% of its charge; "
          "lower surface tension lowers the limit.", ("rayleigh_1882", "kebarle_verkerk_2009", "gomez_tang_1994")),
    Claim("MS: ion path (spray close-up)",
          "Ions leave the last small droplets by ion evaporation (small ions) or as the charge residue of a droplet "
          "that dries out; droplets that reach the inlet still wet carry analyte that never became an ion, and the "
          "inlet passes ions with biases of its own.",
          ("iribarne_thomson_1976", "dole_1968", "kebarle_verkerk_2009", "page_2007", "page_2009")),
    Claim("MS: ion path (spray close-up)",
          "Very hot drying gas can break fragile compounds in the source.", ("fenn_1989",), "assumed"),
    Claim("MS: ion path (Q1 close-up)",
          "At unit resolution Q1 passes ions within about 0.35 m/z of the setting (0.7 m/z wide); the example "
          "method runs Q1 and Q2 at 0.7 amu, and the identification guidance assumes unit-mass MS/MS.",
          ("agilent_5994_1734en", "sante_11312_2021", "dawson_1995")),
    Claim("MS: ion path (stability diagram)",
          "An ion's motion between the rods is the Mathieu equation with a = 8zeU/(m r0^2 W^2) and q = 4zeV/(m r0^2 W^2); "
          "the first stability region lies under the characteristic values a0(q) and b1(q), apex q = 0.706, "
          "a = 0.237; fixed U and V put every ion on one line of slope 2U/V.",
          ("paul_steinwedel_1953", "dawson_1995", "abramowitz_stegun_1964")),
    Claim("MS: ion path (stability diagram)",
          "Real edges are sloped because an ion spends a finite number of RF cycles in the rods; the attainable "
          "resolution grows roughly as n^2/12.", ("dawson_1995",), "partly"),
    Claim("MS: ion path (inside the rods)",
          "One pair carries +(U + V cos wt) and the other -(U + V cos wt): the pair with the DC's sign loses the light "
          "ions, which the RF shakes hardest, and the other loses the heavy ions, which the RF barely holds.",
          ("paul_steinwedel_1953", "dawson_1995")),
    Claim("MS: ion path (the saddle)",
          "The rod potential is a saddle; a static field cannot hold an ion in every direction, but the RF averaged "
          "over a cycle acts as a bowl whose depth falls with m/z (the pseudopotential q V / 4, a_eff = q^2 / 2 at "
          "small q).", ("dehmelt_1967", "gerlich_1992")),
    Claim("MS: ion path; MS: how a quadrupole filters (collision cell)",
          "An RF-only quadrupole holds every ion above a low-m/z cutoff, an ion guide not a filter, and collisions with "
          "gas focus ions to its axis.", ("dawson_1995", "douglas_french_1992", "gerlich_1992")),
    Claim("MS: ion path; MS: how a quadrupole filters (collision cell)",
          "The rod gap (4 mm), RF frequency (1 MHz), the cell's quadrupole geometry, length, gas pressure (5 mTorr), "
          "cell RF (60 V) and the 150 A^2 collision cross-section are drawing values, not the Ultivo's.",
          ("dawson_1995",), "assumed"),
    # MS: how a quadrupole filters (app/views/ms_quad_filter.py, app/quad_stability.py, app/collision_cell.py)
    Claim("MS: how a quadrupole filters (step 1, the saddle)",
          "Two rod pairs at opposite potentials make a saddle between them: a positive ion is pushed back to the centre "
          "along the + pair's line and out along the - pair's, more strongly further from the centre; the rods are drawn "
          "as line charges, a simplification of the hyperbolic field that round rods approximate near the axis.",
          ("paul_steinwedel_1953", "dawson_1995", "paul_1990"), "partly"),
    Claim("MS: how a quadrupole filters (step 1, the saddle)",
          "No static electric field holds a charge at rest in every direction (Earnshaw's theorem: a potential obeying "
          "Laplace's equation has no minimum), so DC alone cannot hold an ion; the flipping field can, on average.",
          ("gerlich_1992", "paul_1990")),
    Claim("MS: how a quadrupole filters (step 2, every knob)",
          "Every knob (mass, charge, RF amplitude, DC, frequency, rod gap) enters only through a = 8zeU/(m r0^2 W^2) "
          "and q = 4zeV/(m r0^2 W^2), so only m/z counts; with no DC the ion is held below q = 0.908, where b1(q) = 0, "
          "and shaken out above it.", ("paul_steinwedel_1953", "dawson_1995", "abramowitz_stegun_1964")),
    Claim("MS: how a quadrupole filters (step 2, every knob)",
          "A held ion carries a fast jiggle at the RF frequency on a slow swing through the centre at the secular "
          "frequency beta W/2, one swing per 2/beta RF cycles; the slow swing is the motion in the pseudopotential "
          "bowl, deeper for lighter ions, and beta comes from the one-cycle Floquet matrix (cos pi beta = trace/2, "
          "|trace| < 2 stable).", ("dawson_1995", "dehmelt_1967", "gerlich_1992", "abramowitz_stegun_1964")),
    Claim("MS: how a quadrupole filters (step 3, the contest)",
          "Fixed U and V put every ion on the line a = 2(U/V) q; written in volts the - pair's edge is "
          "U = -a0(q) V/(2q), the heaviest ion whose cup beats the pull, and the + pair's edge is U = b1(q) V/(2q), the "
          "lightest not shaken out; the window closes at U/V = 0.1678 and the default DC gives the example method's "
          "0.7 m/z unit resolution.",
          ("paul_steinwedel_1953", "dawson_1995", "abramowitz_stegun_1964", "agilent_5994_1734en")),
    Claim("MS: how a quadrupole filters (step 5, the collision cell)",
          "The precursor enters the cell at CE x z electronvolts; in a collision with N2 at most the centre-of-mass "
          "share M/(m + M) of that energy can go into breaking it, so a light ion gives up a large share per collision "
          "and a heavy one little, and after enough collisions it is as slow as the gas.",
          ("hoffmann_stroobant_2007", "douglas_french_1992")),
    Claim("MS: how a quadrupole filters (step 5, the collision cell)",
          "The fragment leaves with the precursor's velocity, so it keeps the fraction m_fragment/m_precursor of the "
          "kinetic energy and its q rises by m_precursor/m_fragment; kinetic energy release is neglected.",
          ("hoffmann_stroobant_2007",), "partly"),
    Claim("MS: how a quadrupole filters (step 5, the collision cell)",
          "Collisions are drawn as elastic hard spheres with thermal N2, isotropic in the centre-of-mass frame, at "
          "rate n sigma g, sampled by the null-collision method; collisional cooling focuses ions to the axis and "
          "raises transmission with pressure.", ("skullerud_1968", "douglas_french_1992"), "partly"),
    Claim("MS: how a quadrupole filters (step 5, the collision cell)",
          "The 150 A^2 collision cross-section at 200 Da, scaled as mass^(2/3), is the order of magnitude drift-tube "
          "ion mobility measures for small molecules in nitrogen (caffeine, 194 Da, about 140 A^2); the value itself "
          "is not the Ultivo's.", ("picache_2019",), "partly"),
    Claim("MS: how a quadrupole filters (step 5, the collision cell)",
          "A hexapole or octopole cell has a flatter effective potential near the axis and no sharp Mathieu cutoff, "
          "so its low-mass edge is softer than a quadrupole's.", ("gerlich_1992",), "partly"),
    Claim("MS: molecules to response",
          "An ion striking the multiplier starts a cascade of roughly a million electrons.", ("hoffmann_stroobant_2007",)),
    Claim("MS: molecules to response",
          "The simulator reports abundance as a rate (counts divided by the dwell), so the expected area does not depend "
          "on the dwell and only the ions counted, and so the scatter, do; the vendor notes state the same trade, "
          "longer dwell buys precision and signal-to-noise, not a larger peak.",
          ("knoll_2010", "sciex_scheduled_mrm", "agilent_5990_3595"), "partly"),
    Claim("MS: molecules to response",
          "The suppression stage is the matrix factor at the compound's retention time, response in matrix over "
          "response in solvent, mapped by post-column infusion; the typical range 0.2 to 1 is an assumed bracket.",
          ("matuszewski_2003", "bonfiglio_1999", "king_2000"), "partly"),
    # LC: inside the column, "Why it's retained: the molecules" (app/molecules.py)
    Claim("LC: inside the column (molecules); docs/lc.md",
          "Each atom's share of hydrophobicity is its Wildman-Crippen log P contribution, which sum to the Crippen log P; "
          "XLogP3 is PubChem's separate estimate.", ("wildman_crippen_1999", "cheng_2007")),
    Claim("LC: inside the column (molecules); docs/lc.md",
          "Parts of a molecule with positive log P share hold it on C18 and parts with negative share pull it into "
          "the mobile phase: a reading of the partitioning picture, not a computed retention.",
          ("dorsey_dill_1989", "rafferty_2007"), "partly"),
)


@dataclass(frozen=True)
class Site:
    key: str
    where: str
    what: str


def docs_citations(docs: Path = DOCS) -> list[Site]:
    out = []
    for p in sorted(docs.glob("*.md")):
        if p.name == "references.md":
            continue
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in CITE_MARK.finditer(line):
                for k in (x.strip() for x in m.group(1).split(",")):
                    out.append(Site(k, f"docs/{p.name}:{n}", line.strip()[:120]))
    return out


def citing_sites() -> list[Site]:
    from lcms_sim.factors import BENCH, STAGE
    from lcms_sim.faults.library import LIBRARY
    from lcms_sim.faults.matcher import default_checks
    sites: list[Site] = []
    for f in BENCH + STAGE:
        for k in f.sources:
            sites.append(Site(k, "Follow: factor", f.name))
    for fl in LIBRARY.values():
        for k in fl.sources:
            sites.append(Site(k, "Fault lab: fault", fl.name))
    for c in default_checks():
        for k in c.sources:
            sites.append(Site(k, "Symptom matcher: bench check", c.name))
    for stage, keys in STAGE_SOURCES.items():
        for k in keys:
            sites.append(Site(k, "trace: typical share", stage))
    for c in CLAIMS:
        for k in c.sources:
            sites.append(Site(k, c.where, c.text))
    sites += docs_citations()
    return sites


def unsourced() -> list[tuple[str, str, str]]:
    """(where, what, provenance) for factors and faults that cite nothing: the honest list of assumed claims."""
    from lcms_sim.factors import BENCH, STAGE
    from lcms_sim.faults.library import LIBRARY
    out = [("Follow: factor", f.name, f.provenance) for f in BENCH + STAGE if not f.sources]
    out += [("Fault lab: fault", fl.name, "signature from the model") for fl in LIBRARY.values() if not fl.sources]
    out += [(c.where, c.text, "assumed") for c in CLAIMS if c.status == "assumed"]
    return out


def cited_by(key: str, sites: Sequence[Site] | None = None) -> list[Site]:
    sites = citing_sites() if sites is None else sites
    return [s for s in sites if s.key == key]


def references_markdown() -> str:
    """docs/references.md: every reference with its verification and every place that cites it."""
    refs = load_references()
    sites = citing_sites()
    lines = ["# References", "",
             "Every source the simulator's claims cite, with how each was verified and where it is cited. Generated by",
             "`scripts/make_references_md.py` from `data/reference/references.yaml` and the citing sites; do not edit by",
             "hand. Claims with no adequate source stay labelled assumed and are listed at the end.", ""]
    by_kind: dict[str, list[Reference]] = {}
    for r in refs.values():
        by_kind.setdefault(r.kind, []).append(r)
    for kind in sorted(by_kind):
        lines += [f"## {kind.capitalize()}", ""]
        for r in sorted(by_kind[kind], key=lambda x: x.key):
            lines.append(f"### {r.short()} (`{r.key}`)")
            lines.append("")
            lines.append(r.full())
            if r.note:
                lines.append("")
                lines.append(f"Supports: {r.note}")
            lines.append("")
            lines.append(f"Verified: {r.verified}")
            cites = cited_by(r.key, sites)
            if cites:
                lines.append("")
                lines.append("Cited by:")
                seen = set()
                for s in cites:
                    tag = f"- {s.where}: {s.what}"
                    if tag not in seen:
                        seen.add(tag)
                        lines.append(tag)
            lines.append("")
    un = unsourced()
    lines += ["## Assumed, with no source", ""]
    if un:
        lines += [f"- {where}: {what} ({prov})" for where, what, prov in un]
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)
