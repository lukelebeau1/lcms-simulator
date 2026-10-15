"""Phase 2: scheduling, counting statistics, integration, ion ratios."""

import numpy as np
import pytest


from lcms_sim.io.schemas import AcquisitionMethod, Transition
from lcms_sim.ms.acquisition import build_schedule, cycle_layout, dwell_time_ms, points_across_peak, report
from lcms_sim.ms.noise import PeakSpec, acquire, expected_counted_ions
from lcms_sim.ms.ratios import counting_limit_rsd, ion_ratios
from lcms_sim.rng import make_rng


def acq(**kw):
    base = dict(cycle_time_ms=100.0, inter_scan_delay_ms=2.0, gain_area_per_ion=10.0)
    return AcquisitionMethod.model_validate({**base, **kw})


def tr(analyte="A", role="quantifier", rt=5.0, window=1.0, rel=1.0, polarity="positive", **kw):
    return Transition.model_validate(dict(analyte=analyte, precursor_mz=300, product_mz=200, role=role,
                                          retention_time_min=rt, window_min=window, relative_response=rel,
                                          polarity=polarity, **kw))


# --- scheduling ------------------------------------------------------------


@pytest.mark.parametrize("pols", [("positive",) * 4, ("positive", "negative", "positive", "negative")])
def test_cycle_layout_fills_the_cycle_without_overlap(pols):
    a = acq(polarity_switch_ms=10.0)
    trs = [tr(analyte=f"X{i}", rt=5.0, polarity=p) for i, p in enumerate(pols)]
    s = build_schedule(trs, a, 10.0)
    lay = cycle_layout(s, a, 5.01)
    used = sum(x.dwell_ms + x.delay_ms for x in lay.slots) + sum(e - b for b, e in lay.switches_ms)
    assert used == pytest.approx(a.cycle_time_ms)
    ends = [x.start_ms + x.dwell_ms + x.delay_ms for x in lay.slots]
    assert all(e <= n.start_ms + 1e-9 for e, n in zip(ends, lay.slots[1:]))
    assert bool(lay.switches_ms) == ("negative" in pols)
    # Positive group first, as the signal model samples them.
    assert [trs[x.transition_index].polarity.value for x in lay.slots] == sorted(pols, key=lambda p: p != "positive")
    k = np.searchsorted(s.cycle_starts, 5.01, side="right") - 1
    for x in lay.slots:  # slot centre is the sample time
        mid = lay.cycle_start_min + (x.start_ms + (x.dwell_ms + x.delay_ms) / 2) / 60_000
        assert mid == pytest.approx(s.sample_times[k, x.transition_index])


def test_cycle_layout_empty_when_nothing_scheduled():
    a = acq()
    s = build_schedule([tr(rt=5.0, window=1.0)], a, 10.0)
    assert cycle_layout(s, a, 1.0).slots == []


@pytest.mark.parametrize("n", [1, 2, 5, 17])
def test_dwell_matches_formula(n):
    a = acq()
    trs = [tr(analyte=f"X{i}", rt=5.0) for i in range(n)]
    s = build_schedule(trs, a, 10.0)
    i = np.searchsorted(s.cycle_starts, 5.0)
    assert s.n_active[i] == n
    assert s.dwell_ms[i] == pytest.approx(dwell_time_ms(100.0, n, 2.0))
    assert s.dwell_ms[i] == pytest.approx(100.0 / n - 2.0)


def test_concurrency_follows_windows():
    trs = [tr(analyte="A", rt=3.0, window=1.0), tr(analyte="B", rt=3.4, window=1.0), tr(analyte="C", rt=6.0)]
    s = build_schedule(trs, acq(), 10.0)
    assert s.concurrency(2.7) == 1
    assert s.concurrency(3.2) == 2
    assert s.concurrency(4.5) == 0
    assert s.concurrency(6.0) == 1


def test_polarity_switch_costs_dwell():
    a = acq(polarity_switch_ms=10.0)
    s = build_schedule([tr(analyte="P"), tr(analyte="N", polarity="negative")], a, 10.0)
    i = np.searchsorted(s.cycle_starts, 5.0)
    assert s.dwell_ms[i] == pytest.approx((100.0 - 20.0) / 2 - 2.0)


def test_samples_are_staggered_within_cycle():
    s = build_schedule([tr(analyte=f"X{i}") for i in range(4)], acq(), 10.0)
    i = np.searchsorted(s.cycle_starts, 5.0)
    ts = s.sample_times[i]
    assert np.all(np.diff(ts) > 0)
    assert ts[0] > s.cycle_starts[i] and ts[-1] < s.cycle_starts[i] + s.cycle_min


def test_too_many_transitions_raises():
    with pytest.raises(ValueError):
        build_schedule([tr(analyte=f"X{i}") for i in range(60)], acq(), 10.0)


def test_report_flags_low_dwell():
    r = report(build_schedule([tr(analyte=f"X{i}") for i in range(40)], acq(min_dwell_ms=1.0), 10.0), acq(min_dwell_ms=1.0))
    assert r.max_concurrent == 40 and r.low_dwell_cycles > 0


def test_points_across_peak():
    assert points_across_peak(sigma_min=0.05, cycle_time_ms=500.0) == pytest.approx(24.0)


# --- counting statistics ---------------------------------------------------


def _single(area, a, rel=0.5, center=5.0, sigma=0.05, n_others=0):
    trs = [tr(role="quantifier", rel=1.0), tr(role="qualifier", rel=rel)]
    trs += [tr(analyte=f"Z{i}") for i in range(n_others)]
    s = build_schedule(trs, a, 10.0)
    return [PeakSpec("A", center, sigma, area)], trs, s


def _area_for_counts(n_counts, a, n_transitions):
    dwell = dwell_time_ms(a.cycle_time_ms, n_transitions, a.inter_scan_delay_ms)
    return n_counts * a.gain_area_per_ion * a.cycle_time_ms / dwell


def test_area_unbiased_and_counting_limited():
    a = acq()
    area = _area_for_counts(2000, a, 2)
    peaks, trs, s = _single(area, a)
    rng = make_rng(1)
    areas = np.array([acquire(peaks, trs, s, a, rng)[0].area for _ in range(1500)])
    n = expected_counted_ions(area, a.gain_area_per_ion, s.dwell_ms[s.n_active == 2][0], a.cycle_time_ms)
    assert n == pytest.approx(2000)
    assert areas.mean() == pytest.approx(area, rel=3 / np.sqrt(n * 1500) + 0.002)
    assert areas.std(ddof=1) / areas.mean() == pytest.approx(1 / np.sqrt(n), rel=0.10)


@pytest.mark.parametrize("n_quant", [100, 1_000, 10_000, 100_000])
def test_ion_ratio_rsd_matches_counting_limit(n_quant):
    a = acq()
    peaks, trs, s = _single(_area_for_counts(n_quant, a, 2), a, rel=0.5)
    rng = make_rng(n_quant)
    ratios = []
    for _ in range(2000):
        sig = acquire(peaks, trs, s, a, rng)
        ratios.append(ion_ratios(sig, {1: 0.5})[0].observed)
    ratios = np.array(ratios)
    assert ratios.mean() == pytest.approx(0.5, rel=0.02)
    assert ratios.std(ddof=1) / ratios.mean() == pytest.approx(counting_limit_rsd(n_quant, 0.5 * n_quant), rel=0.10)


def test_more_concurrent_transitions_worse_precision_same_mean():
    a = acq()
    area = _area_for_counts(500, a, 2)
    res = {}
    for others in (0, 18):
        peaks, trs, s = _single(area, a, n_others=others)
        rng = make_rng(7)
        x = np.array([acquire(peaks, trs, s, a, rng)[0].area for _ in range(1500)])
        res[others] = (x.mean(), x.std(ddof=1) / x.mean())
    assert res[18][0] == pytest.approx(res[0][0], rel=0.02)
    assert res[18][1] > 2.5 * res[0][1]


def test_injection_noise_cancels_in_ratio():
    a = acq(injection_rsd=0.10)
    peaks, trs, s = _single(_area_for_counts(1e6, a, 2), a)
    rng = make_rng(3)
    quant, ratio = [], []
    for _ in range(400):
        sig = acquire(peaks, trs, s, a, rng)
        quant.append(sig[0].area)
        ratio.append(sig[1].area / sig[0].area)
    assert np.std(quant) / np.mean(quant) == pytest.approx(0.10, rel=0.15)
    assert np.std(ratio) / np.mean(ratio) < 0.01


def test_background_is_subtracted():
    a = acq(background_rate=5e4)
    area = _area_for_counts(5000, a, 2)
    peaks, trs, s = _single(area, a)
    trs = [tr(role="quantifier", window=2.0), tr(role="qualifier", rel=0.5, window=2.0)]
    from lcms_sim.ms.acquisition import build_schedule as bs
    s = bs(trs, a, 10.0)
    rng = make_rng(5)
    areas = [acquire(peaks, trs, s, a, rng)[0].area for _ in range(300)]
    assert np.mean(areas) == pytest.approx(area, rel=0.02)


def test_window_clipping_loses_area():
    a = acq()
    area = _area_for_counts(1e6, a, 2)
    peaks, trs, s = _single(area, a, center=5.45, sigma=0.05)  # window is 4.5-5.5
    sig = acquire(peaks, trs, s, a, make_rng(0))[0]
    assert sig.clipped
    assert sig.area < 0.9 * area


def test_saturation_bends_high_signal():
    a = acq(detector_dead_time_s=1e-7)
    peaks_lo, trs, s = _single(_area_for_counts(1e4, a, 2), a)
    peaks_hi, _, _ = _single(_area_for_counts(1e10, a, 2), a)
    lo = acquire(peaks_lo, trs, s, a, make_rng(0))[0]
    hi = acquire(peaks_hi, trs, s, a, make_rng(0))[0]
    assert lo.area / lo.expected_area == pytest.approx(1.0, rel=0.05)
    assert hi.area / hi.expected_area < 0.9


def test_same_seed_same_output():
    a = acq(injection_rsd=0.02, point_rsd=0.01, background_rate=1e3)
    peaks, trs, s = _single(_area_for_counts(1000, a, 2), a)
    x = acquire(peaks, trs, s, a, make_rng(42))
    y = acquire(peaks, trs, s, a, make_rng(42))
    assert [v.area for v in x] == [v.area for v in y]
    assert all(np.array_equal(u.abundance, v.abundance) for u, v in zip(x, y))
