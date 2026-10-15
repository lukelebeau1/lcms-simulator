# Why the analytes elute in this order

Retention times and elution order are from Agilent 5994-1734EN, Appendix A: Poroshell 120
Phenyl-Hexyl 3 x 100 mm, 2.7 um, 55 °C, 0.5 mL/min. Mobile phase A is 5 mM ammonium formate
with 0.1% formic acid in water, and B is 0.1% formic acid in methanol. The gradient holds
30% B for 1 min, rises to 75% B at 2 min and 96% B at 8 min, then reaches 100% B at 9 min.

log P values below are approximate octanol-water values recalled from the pesticide literature
(PPDB-type compilations). They are for orientation only; check them before quoting.

## The one-line answer

In reversed phase, a compound is retained in proportion to how much of its surface is
hydrophobic and how little it wants to stay in the polar mobile phase. That is roughly log P,
corrected for charge, size, and specific contacts with the stationary phase.

## The order, in five groups

| Group | App note RT (min) | Examples (approx. log P) | Why they sit here |
|---|---|---|---|
| Barely retained | 1.1-2.2 | Daminozide (-1.5), acephate (-0.9), oxamyl (-0.4), methomyl (0.1), flonicamid (-0.2) | Small, with amide, oxime, carbamate or acid groups that hydrogen bond with water. They elute during the 30% B hold, before the gradient reaches the column |
| Polar heterocycles | 2.8-3.6 | Thiamethoxam (-0.1), imidacloprid (0.6), dimethoate (0.7), acetamiprid (0.8); aflatoxins G2, G1, B2, B1 | Polar ring systems, nitro-guanidine and cyano-amidine groups. Aflatoxins are flat, polar coumarins |
| Mid-polarity | 3.6-5.1 | Propoxur, dichlorvos (1.9), carbaryl (2.4), metalaxyl (1.8), azoxystrobin (2.5), myclobutanil (2.9), tebuconazole (3.7) | Neutral, moderately hydrophobic carbamates, organophosphates, strobilurins, azoles |
| Hydrophobic, large | 5.2-7.0 | Kresoxim-methyl, spinosyns, pyrethrins, piperonyl butoxide, chlorpyrifos (4.7), fenpyroximate, abamectin (4.4) | Many carbons, few polar groups. Large molecules also have a large S, so they release late and sharply |
| Very hydrophobic | 7.0-8.5 | Pyridaben (6.4), cypermethrin and cyfluthrin (6.6), permethrin (6.1), bifenthrin (6.6), etofenprox (6.9), acequinocyl (C12 chain) | Halogenated aromatics and long alkyl chains. They need 96-100% methanol to elute |

## Effects that move a compound away from its log P position

**Charge at pH ~3.** Mobile phase A is about pH 3.
- Bases are protonated and charged, so they elute earlier than their log P suggests.
  Imazalil (log P ~3.8, pKa ~6.5) elutes at 4.0 min, beside compounds of log P ~2.
  The spinosyns (amine, pKa ~8) are fully charged and are retained only because their
  macrolide skeleton is so large.
- Acids are mostly neutral, so they are retained. Ochratoxin A's carboxylic acid (pKa ~4.4)
  sits close enough to the mobile phase pH that a small pH error moves it. That makes it the
  natural test compound for the "mobile phase pH prepared wrong" fault.
- Chlormequat is a quaternary ammonium, charged at every pH. It is not retained on C18 at
  all, which is why it is missing from the app note. Labs usually run it by HILIC or a
  separate method.

**Molecular size, through S.** Large molecules have a steep dependence of k on %B (high S).
They stay near the column head until the organic content crosses a threshold, then move
quickly. That is why abamectin (MW 873) elutes in a narrow peak despite a moderate log P.

**The aflatoxin sub-order, G2 < G1 < B2 < B1.** The G series carries an extra lactone oxygen
in place of the cyclopentenone ring, which makes it more polar, so it elutes first. Within
each pair, the dihydro form (B2, G2) elutes before the form with the furan double bond
(B1, G1). The usual explanation is the flatter, more conjugated ring of B1 and G1. On a
phenyl-hexyl phase that ring adds pi-pi contact with the phenyl groups.

**Phenyl-hexyl versus a C18 column.** The app note column retains flat, electron-poor
aromatic rings (aflatoxins, chloropyridines, nitro groups) partly through pi-pi interactions.
A C18 phase retains mostly by hydrophobic contact. Expect aromatic, planar
compounds to move somewhat earlier relative to aliphatic ones on a C18 column. The group
order above should hold; neighbours within about 0.1 min may swap. One injection of the mix
on your system fixes this.

**Dwell volume.** The first group elutes before the gradient reaches the column. Their
retention depends on the isocratic k at 30% B and on how long the column sees 30% B, which
is the hold plus the dwell time. The app note's dwell volume is about 1.7 column volumes,
against 0.5 on your 4.6 mm column. The simulator shows the effect: in the transferred method
dimethoate moves from just before to just after aflatoxin G1 and imidacloprid.

**Temperature.** 55 °C lowers retention and viscosity for everything. It is not modelled
here; the plan lists temperature as a known limit.

## Why the late eluters are ammonium adducts

The pyrethroids, etofenprox, spiromesifen, acequinocyl and abamectin have no basic site to
protonate. In electrospray they pick up NH4+ from the ammonium formate in mobile phase A, so
the app note monitors [M+NH4]+ precursors for them. That ties chromatography to signal:
- These compounds elute in near-100% methanol. That desolvates easily and helps their signal.
- They co-elute with the bulk of the cannabinoids and plant lipids, which elute in the same
  high-organic region. That is the main suppression risk, and it sits on top of the
  hardest-to-ionise compounds.
- Their adduct signal depends on the ammonium concentration reaching the source. A mobile
  phase made without ammonium formate hits these compounds hardest.

## Sources

- Agilent application note 5994-1734EN (2020), Appendix A and HPLC conditions; local copy in `data/reference/` [agilent_5994_1734en].
- Minnesota OCM, Cannabis Technical Authority v2.0, Tables 6.2 and 6.4 [mn_ocm_cta_v2].
- S ~ 0.25 MW^0.5: Snyder and Dolan, High-Performance Gradient Elution, Wiley 2007 [snyder_dolan_2007].

Full citations, verification and every citing site: docs/references.md.
