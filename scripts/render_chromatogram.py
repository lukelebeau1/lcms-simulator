"""Simulate one injection of an example sample and write an interactive chromatogram to out/.

Run:  python scripts/render_chromatogram.py [sample_id] [seed]
Opens nothing; the HTML file path is printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lcms_sim.io.loaders import load_input_set
from lcms_sim.ms.acquisition import points_across_peak, report
from lcms_sim.paths import EXAMPLES, ROOT
from lcms_sim.pipeline import simulate_injection
from lcms_sim.rng import make_rng, new_seed


def ng_on_column(inputs, ppb: float) -> float:
    """ug/kg in flower -> ng on column, from the nominal prep (Phase 3 replaces this)."""
    p = inputs.prep
    vial_ng_ml = ppb * p.sample_mass_g.nominal / p.extraction_volume_ml.nominal / p.dilution_factor.nominal
    return vial_ng_ml * p.injection_volume_ul.nominal / 1000.0


def main() -> None:
    sample_id = sys.argv[1] if len(sys.argv) > 1 else "qc_at_limit"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else new_seed()
    inputs = load_input_set(EXAMPLES)
    sample = next(s for s in inputs.samples if s.id == sample_id)
    amounts = {k: ng_on_column(inputs, v) for k, v in sample.true_concentration_ug_kg.items()}
    inj = simulate_injection(inputs, amounts, make_rng(seed))

    rep = report(inj.schedule, inputs.acquisition)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05,
                        subplot_titles=(f"SIMULATED: {sample_id}, quantifier transitions (seed {seed})",
                                        "Concurrent transitions and dwell time"))
    for s in inj.signals:
        if s.role.value == "quantifier" and s.times.size:
            fig.add_trace(go.Scatter(x=s.times, y=s.abundance, mode="lines+markers", marker_size=3,
                                     name=s.analyte, line_width=1, showlegend=False,
                                     hovertemplate=f"{s.analyte}<br>%{{x:.3f}} min<br>%{{y:.3g}}<extra></extra>"),
                          row=1, col=1)
    t = inj.schedule.cycle_starts
    fig.add_trace(go.Scatter(x=t, y=inj.schedule.n_active, name="concurrent", line_color="gray"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.where(inj.schedule.n_active > 0, inj.schedule.dwell_ms, np.nan),
                             name="dwell (ms)", line_color="firebrick"), row=2, col=1)
    fig.update_xaxes(title_text="Time (min)", row=2, col=1)
    fig.update_yaxes(title_text="Abundance (area units / min)", row=1, col=1)
    fig.update_yaxes(title_text="n / dwell (ms)", type="log", row=2, col=1)
    fig.update_layout(height=800, template="plotly_white")

    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    path = out / f"chromatogram_{sample_id}.html"
    fig.write_html(path, include_plotlyjs="cdn")

    cyc = inputs.acquisition.cycle_time_ms
    worst = sorted(inj.peaks, key=lambda p: points_across_peak(p.sigma_min, cyc))[:3]
    print(f"seed {seed}; max concurrent {rep.max_concurrent}, min dwell {rep.min_dwell_ms:.1f} ms, "
          f"cycles below {inputs.acquisition.min_dwell_ms} ms: {rep.low_dwell_cycles}")
    print("fewest points across peak: " + ", ".join(
        f"{p.analyte} {points_across_peak(p.sigma_min, cyc):.1f}" for p in worst))
    clipped = sorted({s.analyte for s in inj.signals if s.clipped})
    print(f"clipped by window: {clipped or 'none'}")
    print(path)


if __name__ == "__main__":
    main()
