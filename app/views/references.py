"""References: every source the simulator's claims cite, how each was verified, and where it is cited; and the
claims that have no source and stay labelled assumed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import page  # noqa: E402

from lcms_sim.references import CLAIMS, cited_by, citing_sites, load_references, unsourced  # noqa: E402

page("References",
     "Every positive claim in the theory texts, fault signatures, bench checks and typical values cites a source "
     "from one registry. Each entry records how it was verified: a DOI resolved, an ISBN on the publisher's page, "
     "a document's text read. A claim without an adequate source keeps the label assumed and is listed at the end; "
     "that list is the honest edge of the simulator.", show_rail=False)

refs = load_references()
sites = citing_sites()

c1, c2, c3 = st.columns(3)
c1.metric("References", len(refs))
c2.metric("Citing claims", len(sites))
c3.metric("Assumed, no source", len(unsourced()))

kinds = sorted({r.kind for r in refs.values()})
pick = st.multiselect("Kind", kinds, default=kinds)
search = st.text_input("Find", "", help="Matches the key, authors, title or note")

rows = []
for r in refs.values():
    if r.kind not in pick:
        continue
    hay = " ".join([r.key, r.authors, r.title, r.note]).lower()
    if search and search.lower() not in hay:
        continue
    rows.append({"citation": r.short(), "key": r.key, "kind": r.kind, "year": r.year, "title": r.title,
                 "link": r.link if not r.link.startswith("file:") else "", "cited by": len(cited_by(r.key, sites))})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(60 + 36 * len(rows), 700),
             column_config={"link": st.column_config.LinkColumn("link"), "year": st.column_config.NumberColumn(format="%d")})

st.subheader("One reference, in full")
key = st.selectbox("Reference", [r.key for r in refs.values()], format_func=lambda k: f"{refs[k].short()}: {refs[k].title[:70]}")
r = refs[key]
st.markdown(r.full())
if r.note:
    st.markdown(f"**Supports:** {r.note}")
st.caption(f"Verified: {r.verified}")
cites = cited_by(key, sites)
if cites:
    st.markdown("**Cited by**")
    seen = set()
    for s in cites:
        line = f"- {s.where}: {s.what}"
        if line not in seen:
            seen.add(line)
            st.markdown(line)

with st.expander("Claims made in page and doc text (not objects), with their sources"):
    st.dataframe(pd.DataFrame([{"where": c.where, "claim": c.text, "sources": ", ".join(refs[k].short() for k in c.sources),
                                "status": c.status} for c in CLAIMS]), hide_index=True, width="stretch")

st.subheader("Assumed, with no source")
un = unsourced()
if un:
    st.dataframe(pd.DataFrame([{"where": w, "claim": what, "provenance": p} for w, what, p in un]), hide_index=True,
                 width="stretch")
    st.caption("These describe the model's own arithmetic or a behaviour stated without literature behind it. They are "
               "labelled assumed wherever they appear.")
else:
    st.markdown("None.")
