"""Parse Appendix A of Agilent 5994-1734EN into data/reference/agilent_5994-1734en_transitions.csv.

Source: Stone, Hitchcock, Roy, Deckers. "Determination of Pesticides and Mycotoxins in Cannabis
Flower as Defined by Legalized U.S. State Recreational Cannabis Regulations". Agilent 5994-1734EN, 2020.
Rows are kept in the order printed. Within a compound, the first row is taken as the quantifier;
the note does not say which transition it quantifies on.

Run:  python scripts/parse_agilent_appnote.py
"""

import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

REF = Path(__file__).resolve().parents[1] / "data" / "reference"
PDF = REF / "application-pesticide-residue-in-cannabis-with-lcms-us-5994-1734en-agilent.pdf"
ROW = re.compile(
    r"^(?P<name>.+?) (?P<pre>\d+(?:\.\d+)?) (?P<prod>\d+(?:\.\d+)?) (?P<frag>\d+) (?P<ce>\d+) "
    r"(?P<cav>\d+) (?P<rt>\d+(?:\.\d+)?) (?P<win>\d+(?:\.\d+)?) (?P<pol>Positive|Negative)$"
)


def main() -> None:
    pages = PdfReader(PDF).pages
    rows = []
    for page in pages[14:18]:
        for line in (page.extract_text() or "").splitlines():
            m = ROW.match(line.strip())
            if m:
                rows.append(m.groupdict())
    df = pd.DataFrame(rows).rename(columns={
        "name": "compound", "pre": "precursor_mz", "prod": "product_mz", "frag": "fragmentor_v",
        "ce": "collision_energy_v", "cav": "cell_accel_v", "rt": "retention_time_min",
        "win": "window_min", "pol": "polarity"})
    df["polarity"] = df["polarity"].str.lower()
    df["adduct"] = df["compound"].str.extract(r"(\[M\+NH4\]\+)", expand=False).fillna("")
    df["compound"] = df["compound"].str.replace(r"\s*\[M\+NH4\]\+", "", regex=True)
    df["role"] = "qualifier"
    df.loc[~df["compound"].duplicated(), "role"] = "quantifier"
    out = REF / "agilent_5994-1734en_transitions.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} transitions, {df['compound'].nunique()} compounds -> {out.name}")


if __name__ == "__main__":
    main()
