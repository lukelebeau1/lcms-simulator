"""Cartoons for the Follow page: a container or an instrument stage with the analyte drawn inside. Illustration.

Every drawing is a small inline SVG. The liquid level follows the vessel's fill, the number of analyte dots
follows the logarithm of the amount (the real number is printed beside the drawing), and nothing here feeds the
signal. Dot positions come from a fixed low-discrepancy pattern, not a random generator.
"""

from __future__ import annotations

import math

W, H = 150, 120
GLASS, AMBER, LIQUID, GAS = "#9aa6b2", "#b9772c", "#8fb3ff", "#d0d6dd"


def _dots(n: int, x0: float, y0: float, w: float, h: float, colour: str, r: float = 2.4) -> str:
    """n dots inside a box, spread by the golden ratio so they never line up."""
    out = []
    for i in range(max(0, n)):
        fx, fy = (0.5 + i * 0.6180339887) % 1.0, (0.25 + i * 0.7548776662) % 1.0
        out.append(f'<circle cx="{x0 + fx * w:.1f}" cy="{y0 + fy * h:.1f}" r="{r}" fill="{colour}"/>')
    return "".join(out)


def dots_for(amount: float) -> int:
    """How many dots to draw for an amount (molecules or ions): 3 per decade above a million, 1 to 45."""
    if amount <= 0:
        return 0
    return int(min(45, max(1, round(3 * (math.log10(amount) - 6)))))


def _frame(body: str, caption: str = "") -> str:
    cap = (f'<text x="{W / 2}" y="{H - 4}" font-size="10" fill="#2E2433" text-anchor="middle" '
           f'font-family="sans-serif">{caption}</text>') if caption else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
            f"{body}{cap}</svg>")


def vessel(kind: str, fill: float, n_dots: int, colour: str, caption: str = "") -> str:
    """A container with liquid to `fill` (0-1) and the analyte as dots. kind: ampoule, vial_2ml, vial_20ml,
    tube, autosampler_vial, loop."""
    fill = min(1.0, max(0.0, fill))
    if kind == "ampoule":
        x0, y0, w, h, neck = 55, 30, 40, 70, 12
        body = (f'<path d="M{x0 + neck} 8 v22 l-{neck} 10 v{h - 10} a4 4 0 0 0 4 4 h{w - 8} a4 4 0 0 0 4 -4 '
                f'v-{h - 10} l-{neck} -10 v-22 z" fill="none" stroke="{GLASS}" stroke-width="2"/>')
        liquid_box = (x0 + 2, y0 + 10 + (h - 10) * (1 - fill), w - 4, (h - 10) * fill)
    elif kind in ("vial_2ml", "vial_20ml", "autosampler_vial"):
        tall = kind == "vial_20ml"
        x0, y0 = (50, 18) if tall else (57, 32)
        w, h = (50, 90) if tall else (36, 70)
        stroke = AMBER if kind != "autosampler_vial" else GLASS
        body = (f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="5" fill="none" stroke="{stroke}" '
                f'stroke-width="2.5"/><rect x="{x0 + 4}" y="{y0 - 8}" width="{w - 8}" height="9" fill="#555"/>')
        liquid_box = (x0 + 3, y0 + 4 + (h - 8) * (1 - fill), w - 6, (h - 8) * fill)
    elif kind == "tube":
        x0, y0, w, h = 58, 14, 34, 86
        body = (f'<path d="M{x0} {y0} v{h - 20} l{w / 2} 20 l{w / 2} -20 v-{h - 20} z" fill="none" '
                f'stroke="{GLASS}" stroke-width="2.5"/><rect x="{x0 - 3}" y="{y0 - 8}" width="{w + 6}" height="9" fill="#555"/>')
        liquid_box = (x0 + 3, y0 + 4 + (h - 24) * (1 - fill), w - 6, (h - 24) * fill)
    elif kind == "loop":
        body = (f'<path d="M15 60 h30 a15 15 0 1 1 30 0 a15 15 0 1 1 30 0 h30" fill="none" stroke="{GLASS}" '
                f'stroke-width="4"/><path d="M15 60 h30 a15 15 0 1 1 30 0 a15 15 0 1 1 30 0 h30" fill="none" '
                f'stroke="{LIQUID}" stroke-width="2" stroke-dasharray="26 6"/>')
        return _frame(body + _dots(n_dots, 45, 44, 60, 32, colour), caption)
    else:
        raise ValueError(f"unknown vessel kind {kind!r}")
    lx, ly, lw, lh = liquid_box
    liquid = f'<rect x="{lx}" y="{ly:.1f}" width="{lw}" height="{lh:.1f}" fill="{LIQUID}" opacity="0.45"/>'
    return _frame(body + liquid + _dots(n_dots, lx + 2, ly + 2, lw - 4, max(lh - 4, 1), colour), caption)


def stage(kind: str, n_dots: int, colour: str, caption: str = "", band: float = 0.6) -> str:
    """An instrument stage with the analyte as dots. kind: column, spray, capillary, q1, q3, cell, multiplier,
    counting."""
    if kind == "column":
        bx = 20 + 110 * min(max(band, 0.0), 1.0)
        body = (f'<rect x="15" y="45" width="120" height="30" rx="6" fill="#2b2f36" stroke="{GLASS}" stroke-width="2"/>'
                f'<rect x="{bx - 8:.1f}" y="47" width="16" height="26" fill="{colour}" opacity="0.35"/>'
                f'<text x="75" y="35" font-size="9" fill="#5C5262" text-anchor="middle" font-family="sans-serif">flow</text>'
                f'<path d="M30 30 h90" stroke="#8b949e" stroke-width="1" marker-end="url(#a)"/>')
        return _frame(body + _dots(n_dots, bx - 7, 49, 14, 22, colour, 2.0), caption)
    if kind == "spray":
        body = (f'<rect x="10" y="54" width="40" height="12" fill="#555"/>'
                f'<path d="M50 60 L135 25 M50 60 L135 95" stroke="{GAS}" stroke-width="1" stroke-dasharray="3 3"/>'
                + "".join(f'<circle cx="{60 + 10 * i}" cy="{60 + (i % 3 - 1) * 10 * (1 + i / 4):.1f}" '
                          f'r="{max(1.5, 6 - 0.7 * i):.1f}" fill="{LIQUID}" opacity="0.5"/>' for i in range(8)))
        return _frame(body + _dots(n_dots, 60, 30, 75, 60, colour, 2.0), caption)
    if kind == "capillary":
        body = (f'<rect x="10" y="50" width="130" height="20" rx="10" fill="none" stroke="{GLASS}" stroke-width="2"/>'
                f'<text x="75" y="40" font-size="9" fill="#5C5262" text-anchor="middle" font-family="sans-serif">heated, '
                f'voltage</text>')
        return _frame(body + _dots(n_dots, 15, 54, 120, 12, colour, 2.0), caption)
    if kind in ("q1", "q3", "cell"):
        rods = "".join(f'<rect x="20" y="{y}" width="110" height="7" rx="3" fill="{GLASS}"/>' for y in (30, 83))
        extra = (f'<rect x="15" y="25" width="120" height="70" fill="none" stroke="{GAS}" stroke-dasharray="4 3"/>'
                 + _dots(14, 22, 42, 106, 36, GAS, 1.4)) if kind == "cell" else ""
        label = {"q1": "RF + DC: one m/z passes", "q3": "RF + DC: one m/z passes", "cell": "RF only, N2"}[kind]
        body = (rods + extra + f'<text x="75" y="18" font-size="9" fill="#5C5262" text-anchor="middle" '
                f'font-family="sans-serif">{label}</text>')
        return _frame(body + _dots(n_dots, 22, 44, 106, 32, colour, 2.0), caption)
    if kind == "multiplier":
        body = (f'<path d="M20 40 L70 55 L20 70 Z" fill="#555"/>'
                f'<path d="M72 55 L135 30 M72 55 L135 80 M72 55 L135 55" stroke="{colour}" stroke-width="1.5"/>'
                f'<text x="75" y="100" font-size="9" fill="#5C5262" text-anchor="middle" font-family="sans-serif">'
                f'one ion in, a cascade out</text>')
        return _frame(body + _dots(n_dots, 10, 45, 55, 20, colour, 2.0), caption)
    if kind == "counting":
        bars = "".join(f'<rect x="{20 + 11 * i}" y="{95 - h}" width="8" height="{h}" fill="{colour}" opacity="0.8"/>'
                       for i, h in enumerate((4, 9, 20, 42, 60, 48, 26, 12, 6, 3)))
        body = bars + f'<path d="M15 96 h120" stroke="{GLASS}"/>'
        return _frame(body, caption)
    raise ValueError(f"unknown stage kind {kind!r}")
