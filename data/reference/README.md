# Reference lists

`mn_ocm_cta_v2_analytes.csv`: pesticide (Table 6.4) and mycotoxin (Table 6.2) analytes and
action limits from the Minnesota Office of Cannabis Management, Cannabis Technical Authority v2.0,
https://mn.gov/ocm/assets/OCM_Cannabis_Technical_Authority_v2.0_tcm1202-678506.pdf
Pesticide limits are the same for flower and concentrate. Mycotoxins also fail if their total exceeds 20 ppb.
The CTA itself references ASTM D8399-23 for LC-MS/MS pesticide analysis, not a vendor app note.
Mycotoxin CAS numbers were added from general knowledge, not from the CTA.

`agilent_5994-1734en_transitions.csv`: Appendix A of Agilent application note 5994-1734EN (Ultivo transitions,
fragmentor, collision energy, cell accelerator voltage, retention time and window), parsed by
`scripts/parse_agilent_appnote.py` from the PDF, which is not included. Within a compound the first row is taken as
the quantifier; the note does not say which.

`pesticide_xlogp.csv`: PubChem XLogP3, formula and monoisotopic mass per target (`scripts/fetch_xlogp.py`).
XLogP3 is a computed estimate, not a measurement.

`structures.csv`, `structures/`: 3D structures from PubChem (`scripts/fetch_structures.py`), for the molecule views
and the C18 scene.

`references.yaml`: every source the pages and docs cite, with how and when it was verified; `docs/references.md`
is generated from it.
