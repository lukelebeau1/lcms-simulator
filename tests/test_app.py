"""Phase 5: every view runs on the bundled data, without errors, in under a second per interaction."""

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = sorted((APP / "views").glob("*.py"))


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.stem)
def test_view_runs_and_responds(path):
    at = AppTest.from_file(str(path), default_timeout=30)
    at.run()  # first run loads and caches the inputs
    assert not at.exception, at.exception
    assert any("SIMULATED" in w.value for w in at.warning)
    t = time.perf_counter()
    if at.sidebar.slider:
        s = at.sidebar.slider[0]
        s.set_value(s.max if isinstance(s.value, (int, float)) else s.value).run()
    else:
        at.run()
    assert not at.exception, at.exception
    assert time.perf_counter() - t < 1.5  # 1 s target plus test-harness overhead


def test_navigation_lists_every_view_in_lab_order():
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    src = (APP / "Home.py").read_text(encoding="utf-8")
    for p in PAGES:
        assert f'"{p.name}"' in src, p.name
    groups = ('"Vials and tubes"', '"Column"', '"Spray and capillary"', '"Quadrupoles and cell"',
              '"Multiplier and counting"', '"Result"', '"Troubleshooting"', '"Your data"')
    order = [src.index(s) for s in groups]
    assert order == sorted(order)                       # the containers, in the order the portion meets them
    assert src.index('"follow.py"') < src.index('"overview.py"')   # Follow is the main page
    assert at.sidebar.selectbox("follow_injection").value and at.sidebar.selectbox("follow_compound").value


def test_follow_pickers_set_every_pages_default_and_the_rail():
    at = AppTest.from_file(str(APP / "views" / "lc_column.py"), default_timeout=30)
    at.session_state["follow_compound"] = "Bifenthrin"
    at.session_state["follow_injection"] = "Column: Cal 2"
    at.run()
    assert not at.exception, at.exception
    assert at.sidebar.selectbox[0].value == "Bifenthrin"             # the page's own picker follows
    assert any("Bifenthrin in Cal 2" in e.label for e in at.expander)  # the rail's table of stops
    labels = " ".join(m.value for m in at.markdown)
    for station in ("**Column**", "Storage", "Bench", "Injection", "Source", "Analyser", "Detector"):
        assert station in labels


def test_rail_shows_the_trace_and_marks_a_bench_edit():
    from lcms_sim.io.loaders import load_input_set, load_prep_recipe
    from lcms_sim.quant.ledger import with_actual
    from lcms_sim.trace import Actual, build_trace

    examples = APP.parent / "data" / "examples"
    inputs, recipe = load_input_set(examples), load_prep_recipe(examples / "prep_recipe.yaml")
    k = next(k for k, s in enumerate(recipe.steps) if s.op == "inject" and s.vessel == "Column: Cal 8")
    at = AppTest.from_file(str(APP / "views" / "ms_molecule_budget.py"), default_timeout=30)
    at.session_state["bench_actual"] = {k: 5.0}                       # half the injection volume
    at.run()
    assert not at.exception, at.exception
    t = build_trace(inputs, recipe, "Column: Cal 8", "Myclobutanil",
                    actual=Actual(recipe=with_actual(recipe, {k: 5.0})))
    rail = next(d for d in at.dataframe if "loss" in d.value.columns)
    assert rail.value["nominal"].iloc[-1] == pytest.approx(t.nominal_area)
    assert rail.value["actual"].iloc[-1] == pytest.approx(t.actual_area)
    assert any("x0.5 of expected" in m.value for m in at.markdown)


def test_follow_page_lists_the_containers_and_charges_a_factor_to_its_transfer():
    from lcms_sim.io.loaders import load_prep_recipe

    recipe = load_prep_recipe(APP.parent / "data" / "examples" / "prep_recipe.yaml")
    k = next(k for k, s in enumerate(recipe.steps) if s.op == "transfer" and s.source == "Ochratoxin A (2 ug/mL)")
    at = AppTest.from_file(str(APP / "views" / "follow.py"), default_timeout=30)
    at.session_state["follow_compound"] = "Ochratoxin A"
    at.session_state["follow_injection"] = "Column: Cal 8"
    at.session_state[f"fw_{k}:unmixed"] = True
    at.session_state[f"fv_{k}:unmixed"] = 0.5
    at.run()
    assert not at.exception, at.exception
    heads = [h.value for h in at.subheader]
    for name in ("Ochratoxin A (2 ug/mL)", "Mycotoxin master 0.2 ppm", "Injection: Cal 8", "Column", "Spray",
                 "Q1", "Collision cell", "Q3", "Multiplier", "Counting"):
        assert name in heads, name
    text = " ".join(m.value for m in at.markdown)
    assert "x0.5 of expected" in text                    # the unmixed draw, charged on the stock's card
    assert "actual <b>x0.5</b> of expected, 50% lost to the factors set" in text   # the sticky total
    captions = " ".join(c.value for c in at.caption)
    assert "50% lost to the factors set" in captions and "as expected" in captions   # the running tally per card


def test_hosted_import_page_keeps_files_in_the_session(monkeypatch):
    """Without LCMS_SIM_LOCAL (a hosted copy), the Import page offers upload only, never lists or writes data/lab,
    and a stale input_folder in the session cannot point the app at a lab folder."""
    monkeypatch.delenv("LCMS_SIM_LOCAL", raising=False)
    at = AppTest.from_file(str(APP / "views" / "import_data.py"), default_timeout=30)
    at.session_state["input_folder"] = str(APP.parent / "data" / "lab" / "my_lab_set")
    at.run()
    assert not at.exception, at.exception
    radios = [r.options for r in at.sidebar.radio]
    assert not any("A file already in data/lab" in o for opts in radios for o in opts)
    assert not at.sidebar.text_input                                   # no batch name: nothing is saved
    assert any("nothing is saved on the server" in c.value for c in at.sidebar.caption)
    at.sidebar.radio[0].set_value("Lab input set").run()
    assert not at.exception, at.exception
    assert any("own machine" in m.value for m in at.info)
    assert not at.button                                               # no create, use or switch buttons


def test_local_import_page_can_read_data_lab(monkeypatch):
    monkeypatch.setenv("LCMS_SIM_LOCAL", "1")
    at = AppTest.from_file(str(APP / "views" / "import_data.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    assert any("A file already in data/lab" in o for r in at.sidebar.radio for o in r.options)
    assert at.sidebar.text_input
