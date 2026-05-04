#!/usr/bin/env python3
"""Build per-participant HTML reports from N-back PsychoPy CSVs.

For every CSV in ``data/`` (skipping incomplete runs with no scheduled trials),
writes ``docs/reports/<participant>_<datetime>.html`` and refreshes
``docs/index.html`` so it lists every report we have.

Each participant report contains:
  * Header with run metadata and a per-N pass/fail strip.
  * Aggregate confusion matrix and an interactive Bokeh RT distribution
    (histogram + Gaussian KDE) coloured per N-back level.
  * One section per actually-run scheduleLoop block: small confusion table
    plus an interactive Bokeh timeline (RT per trial, glyph encodes the
    target/response combination, hover gives letter + outcome).

Styling matches ``docs/report.html`` (system font stack, #2c3e50 text,
#3498db / #e74c3c accents, white panels on #fafbfc) so the prerandomization
report and the participant reports feel like the same site.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import html
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.plotting import figure
from bokeh.resources import INLINE
from scipy import stats

# --- Constants --------------------------------------------------------------

DATA_DIR = Path("data")
REPORTS_DIR = Path("docs/reports")
INDEX_PATH = Path("docs/index.html")

N_LEVELS = [1, 2, 3, 4, 5]

# Per-N colour palette (Set2-derived). Stable across all reports.
N_COLOURS = {
    1: "#66c2a5",
    2: "#fc8d62",
    3: "#8da0cb",
    4: "#e78ac3",
    5: "#a6d854",
}

# Outcome colours for trial markers and confusion-matrix cells.
OUTCOME_COLOURS = {
    "hit":  "#27ae60",  # target + press → green
    "miss": "#e74c3c",  # target + no-press → red
    "fa":   "#f39c12",  # non-target + press → orange
    "cr":   "#bdc3c7",  # non-target + no-press → soft gray
}

OUTCOME_LABEL = {
    "hit":  "Hit",
    "miss": "Miss",
    "fa":   "False alarm",
    "cr":   "Correct rejection",
}

# Outcome marker for Bokeh timelines.
OUTCOME_MARKER = {
    "hit":  "circle",
    "miss": "x",
    "fa":   "triangle",
    "cr":   "dash",
}


# --- Data loading -----------------------------------------------------------


def _to_float(value: str | float | None):
    """CSV cells are strings; '' / 'None' → NaN, otherwise float."""
    if value is None or value == "" or value == "None":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def _to_bool(value):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in ("true", "1")


def _classify(target: bool, pressed: bool) -> str:
    if target and pressed:
        return "hit"
    if target and not pressed:
        return "miss"
    if not target and pressed:
        return "fa"
    return "cr"


def load_run(csv_path: Path) -> dict | None:
    """Load one PsychoPy CSV. Returns None if the run never reached the
    scheduleLoop (no main trials), so we skip it in the report listing."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = ["block", "N", "letter", "target", "correct",
                "trials.key_resp_trial.rt", "trials.key_resp_trial.keys"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None  # older CSV schema: skip silently

    main_trials = df[
        (df["block"] != "")
        & (df["N"] != "")
        & (df["letter"] != "")
        & (df["target"] != "")
        & (df["correct"] != "")
        & (df.get("trials.thisN", pd.Series([""] * len(df))) != "")
    ].copy()
    if main_trials.empty:
        return None

    main_trials["block"] = main_trials["block"].astype(int)
    main_trials["N"] = main_trials["N"].astype(int)
    main_trials["target"] = main_trials["target"].apply(_to_bool)
    main_trials["correct"] = main_trials["correct"].astype(int)
    main_trials["pressed"] = main_trials["trials.key_resp_trial.keys"].apply(
        lambda v: bool(v) and v != "None"
    )
    main_trials["rt"] = main_trials["trials.key_resp_trial.rt"].apply(_to_float)
    main_trials["outcome"] = [
        _classify(t, p)
        for t, p in zip(main_trials["target"], main_trials["pressed"])
    ]
    main_trials["trial_idx"] = main_trials.groupby(["block", "N"]).cumcount()

    # Surface metadata from the first non-empty cell in each column.
    def first_value(col, default=""):
        if col not in df.columns:
            return default
        for v in df[col]:
            if v not in ("", "None"):
                return v
        return default

    participant = first_value("participant")
    session = first_value("session")
    date_str = first_value("date")
    psychopy_version = first_value("psychopyVersion")
    n_blocks = first_value("nBlocks")
    top_n = first_value("topN")

    # Run wall-clock duration from the timestamp column thisRow.t (seconds
    # since experiment start).
    if "thisRow.t" in df.columns:
        ts = pd.to_numeric(df["thisRow.t"], errors="coerce").dropna()
        duration_s = float(ts.max() - ts.min()) if len(ts) >= 2 else None
    else:
        duration_s = None

    # Failure tracking: each block's score row carries `list_failed`.
    failed_levels = set()
    if "list_failed" in df.columns and "list_N" in df.columns:
        for _, row in df.iterrows():
            if row.get("list_failed") in ("True", "1"):
                try:
                    failed_levels.add(int(float(row["list_N"])))
                except (TypeError, ValueError):
                    pass

    blocks = []
    for (block, n_back), block_df in main_trials.groupby(["block", "N"], sort=True):
        block_df = block_df.sort_values("trial_idx").reset_index(drop=True)
        list_letter = block_df["list_letter"].iloc[0] if "list_letter" in block_df else ""
        outcomes = Counter(block_df["outcome"])
        blocks.append({
            "block": int(block),
            "N": int(n_back),
            "list_letter": list_letter,
            "list_name": f"{int(n_back)}{list_letter}",
            "n_trials": len(block_df),
            "n_hits": outcomes["hit"],
            "n_misses": outcomes["miss"],
            "n_fa": outcomes["fa"],
            "n_cr": outcomes["cr"],
            "miss_rate": outcomes["miss"] / max(outcomes["hit"] + outcomes["miss"], 1),
            "fa_rate": outcomes["fa"] / max(outcomes["fa"] + outcomes["cr"], 1),
            "failed": int(n_back) in failed_levels,
            "trials": block_df.to_dict("records"),
        })

    max_n = max(b["N"] for b in blocks if not b["failed"]) if any(not b["failed"] for b in blocks) else min(b["N"] for b in blocks)

    return {
        "csv_path": csv_path,
        "csv_name": csv_path.name,
        "participant": participant or "anon",
        "session": session,
        "date_str": date_str,
        "psychopy_version": psychopy_version,
        "n_blocks_setting": n_blocks,
        "top_n_setting": top_n,
        "duration_s": duration_s,
        "main_trials": main_trials,
        "blocks": blocks,
        "failed_levels": failed_levels,
        "max_n_reached": max_n,
        "report_slug": _slugify(csv_path.stem),
    }


def _slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "—"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {sec:02d}s"


def _parse_run_datetime(date_str: str) -> datetime.datetime | None:
    # PsychoPy writes dates as "2026-05-04_14h11.06.050".
    if not date_str:
        return None
    try:
        return datetime.datetime.strptime(date_str.split(".")[0], "%Y-%m-%d_%Hh%M")
    except ValueError:
        try:
            return datetime.datetime.strptime(date_str[:16], "%Y-%m-%d_%Hh%M")
        except ValueError:
            return None


# --- Plotting helpers -------------------------------------------------------


def _block_timeline_figure(block: dict):
    trials = block["trials"]
    if not trials:
        return None

    finite_rts = [t["rt"] for t in trials if np.isfinite(t["rt"])]
    y_top = max(finite_rts) * 1.15 if finite_rts else 1.5
    # Extend the axis a touch below zero so no-press markers (misses + CRs)
    # rendered at y=0 aren't clipped by the bottom edge.
    y_floor = -y_top * 0.06

    fig = figure(
        height=240,
        sizing_mode="stretch_width",
        x_axis_label="Trial #",
        y_axis_label="Response time (s)",
        y_range=(y_floor, y_top),
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    fig.toolbar.logo = None
    fig.background_fill_color = "#fafbfc"
    fig.border_fill_color = "white"
    fig.outline_line_color = None
    fig.xgrid.grid_line_color = "#ecf0f1"
    fig.ygrid.grid_line_color = "#ecf0f1"
    fig.axis.axis_line_color = "#bdc3c7"
    fig.axis.major_tick_line_color = "#bdc3c7"
    fig.axis.minor_tick_line_color = None
    fig.axis.axis_label_text_color = "#7f8c8d"
    fig.axis.major_label_text_color = "#7f8c8d"

    # Soft underlay rectangles at target columns for orientation. Use a
    # vbar glyph drawn before the outcome scatters so it sits *below*
    # them in z-order — Span/BoxAnnotation default to the annotation
    # layer and would obscure any glyph on a target trial.
    target_xs = [t["trial_idx"] for t in trials if t["target"]]
    if target_xs:
        fig.vbar(
            x=target_xs, top=y_top, bottom=y_floor,
            width=0.6, fill_color="#fdf2e3", fill_alpha=0.7,
            line_color=None,
        )

    # Per-outcome scatter glyphs, each with its own marker shape so the
    # legend can toggle hits / misses / FAs / CRs independently.
    for outcome, marker in OUTCOME_MARKER.items():
        outcome_trials = [t for t in trials if t["outcome"] == outcome]
        if not outcome_trials:
            continue
        sub = ColumnDataSource(dict(
            x=[t["trial_idx"] for t in outcome_trials],
            y=[t["rt"] if np.isfinite(t["rt"]) else 0 for t in outcome_trials],
            letter=[t["letter"] for t in outcome_trials],
            outcome_label=[OUTCOME_LABEL[outcome]] * len(outcome_trials),
            target=["yes" if t["target"] else "no" for t in outcome_trials],
            rt_str=[(f"{t['rt']:.3f} s" if np.isfinite(t["rt"]) else "(no press)")
                    for t in outcome_trials],
        ))
        colour = OUTCOME_COLOURS[outcome]
        fig.scatter(
            x="x", y="y", source=sub,
            marker=marker, size=12,
            fill_color=colour, line_color=colour, line_width=2, alpha=0.95,
            legend_label=OUTCOME_LABEL[outcome],
        )

    fig.legend.location = "top_right"
    fig.legend.click_policy = "hide"
    fig.legend.background_fill_alpha = 0.85
    fig.legend.label_text_font_size = "10px"
    fig.legend.spacing = 2

    fig.add_tools(HoverTool(tooltips=[
        ("Trial", "@x"),
        ("Letter", "@letter"),
        ("Target", "@target"),
        ("Outcome", "@outcome_label"),
        ("RT", "@rt_str"),
    ]))
    return fig


def _rt_distribution_figure(main_trials: pd.DataFrame):
    """One overlay of histogram + KDE per N-back level. Hits only —
    response times for non-presses are undefined."""
    pressed = main_trials[(main_trials["pressed"]) & main_trials["rt"].notna()]
    if pressed.empty:
        return None

    rts = pressed["rt"].to_numpy()
    rt_max = float(np.percentile(rts, 99)) * 1.05
    bins = np.linspace(0, rt_max, 36)
    grid = np.linspace(0, rt_max, 200)

    fig = figure(
        height=320,
        sizing_mode="stretch_width",
        x_axis_label="Response time (s)",
        y_axis_label="Density",
        x_range=(0, rt_max),
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    fig.toolbar.logo = None
    fig.background_fill_color = "#fafbfc"
    fig.border_fill_color = "white"
    fig.outline_line_color = None
    fig.xgrid.grid_line_color = "#ecf0f1"
    fig.ygrid.grid_line_color = "#ecf0f1"
    fig.axis.axis_line_color = "#bdc3c7"
    fig.axis.major_tick_line_color = "#bdc3c7"
    fig.axis.minor_tick_line_color = None
    fig.axis.axis_label_text_color = "#7f8c8d"
    fig.axis.major_label_text_color = "#7f8c8d"

    for n_back, group in pressed.groupby("N"):
        colour = N_COLOURS.get(int(n_back), "#3498db")
        rts_n = group["rt"].to_numpy()
        if len(rts_n) == 0:
            continue
        counts, edges = np.histogram(rts_n, bins=bins, density=True)
        fig.quad(
            top=counts, bottom=0,
            left=edges[:-1], right=edges[1:],
            fill_color=colour, fill_alpha=0.18,
            line_color=colour, line_alpha=0.4,
            legend_label=f"{int(n_back)}-back  (n={len(rts_n)})",
        )
        if len(rts_n) >= 3 and rts_n.std() > 0:
            kde = stats.gaussian_kde(rts_n)
            density = kde(grid)
            fig.line(grid, density, color=colour, line_width=2.5,
                     legend_label=f"{int(n_back)}-back  (n={len(rts_n)})")

    fig.legend.location = "top_right"
    fig.legend.click_policy = "hide"
    fig.legend.background_fill_alpha = 0.85
    return fig


# --- Confusion matrix (SVG) -------------------------------------------------


def _confusion_svg(n_hits: int, n_misses: int, n_fa: int, n_cr: int,
                   width: int = 360, cell: int = 110) -> str:
    """Two-by-two SVG confusion matrix with counts, percentages, colour."""
    total = max(n_hits + n_misses + n_fa + n_cr, 1)
    cells = [
        (0, 0, "hit",  "Target / Press",        n_hits),
        (1, 0, "miss", "Target / No press",     n_misses),
        (0, 1, "fa",   "Non-target / Press",    n_fa),
        (1, 1, "cr",   "Non-target / No press", n_cr),
    ]
    pad = 4
    total_w = cell * 2 + pad
    total_h = cell * 2 + pad
    parts = [
        f'<svg width="{total_w}" height="{total_h}" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, sans-serif">'
    ]
    for col, row, kind, label, count in cells:
        x = col * (cell + pad)
        y = row * (cell + pad)
        colour = OUTCOME_COLOURS[kind]
        pct = 100 * count / total
        parts.append(
            f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
            f'fill="{colour}" fill-opacity="0.18" stroke="{colour}" '
            f'stroke-width="1.5" rx="6"/>'
        )
        parts.append(
            f'<text x="{x + cell/2}" y="{y + cell/2 - 8}" '
            f'text-anchor="middle" font-size="28" font-weight="600" '
            f'fill="#2c3e50">{count}</text>'
        )
        parts.append(
            f'<text x="{x + cell/2}" y="{y + cell/2 + 14}" '
            f'text-anchor="middle" font-size="11" fill="#7f8c8d">'
            f'{pct:.1f}%</text>'
        )
        parts.append(
            f'<text x="{x + cell/2}" y="{y + cell - 8}" '
            f'text-anchor="middle" font-size="10" fill="#7f8c8d">'
            f'{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _level_strip_svg(blocks_by_n: dict[int, list[dict]],
                     failed_levels: set[int]) -> str:
    """Per-N-level summary chip row at the top of the report."""
    parts = []
    for n in N_LEVELS:
        cls = "level-chip"
        ran = blocks_by_n.get(n, [])
        if not ran:
            status = "not reached"
            cls += " level-chip-empty"
        elif n in failed_levels:
            status = "failed"
            cls += " level-chip-fail"
        else:
            status = "passed"
            cls += " level-chip-pass"
        n_blocks_ran = len(ran)
        parts.append(
            f'<div class="{cls}">'
            f'<span class="level-chip-n" style="color:{N_COLOURS[n]}">{n}-back</span>'
            f'<span class="level-chip-status">{status}</span>'
            f'<span class="level-chip-detail">{n_blocks_ran} block(s)</span>'
            f"</div>"
        )
    return f'<div class="level-strip">{"".join(parts)}</div>'


# --- HTML rendering ---------------------------------------------------------


CSS = """
:root {
  --ink: #2c3e50;
  --muted: #7f8c8d;
  --line: #ecf0f1;
  --card: #ffffff;
  --bg: #fafbfc;
  --accent: #3498db;
  --warn: #e74c3c;
  --pass: #27ae60;
  --fail: #e74c3c;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--bg);
  line-height: 1.5;
}
.wrap { max-width: 1200px; margin: 0 auto; padding: 1.5em 1em 4em; }
h1 { margin: 0 0 0.2em; font-weight: 600; letter-spacing: -0.01em; }
h2 { margin: 1.6em 0 0.6em; font-weight: 600; letter-spacing: -0.005em; }
h3 { margin: 0 0 0.5em; font-weight: 600; }
.meta {
  color: var(--muted); font-size: 0.9em; margin: 0 0 1.5em;
}
.kv {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 0.6em 1.2em;
  margin: 0.8em 0 1.2em;
}
.kv > div { font-size: 0.9em; }
.kv .label { display: block; color: var(--muted); font-size: 0.78em;
             text-transform: uppercase; letter-spacing: 0.06em; }
.kv .value { font-weight: 600; font-size: 1.05em; color: var(--ink); }
.level-strip {
  display: flex; gap: 0.6em; flex-wrap: wrap; margin: 0.8em 0 1.6em;
}
.level-chip {
  flex: 1 1 130px; min-width: 130px;
  padding: 0.6em 0.8em; border-radius: 8px;
  background: var(--card); border: 1px solid var(--line);
  display: flex; flex-direction: column; gap: 2px;
}
.level-chip-n { font-weight: 700; font-size: 1.0em; }
.level-chip-status { font-size: 0.78em; text-transform: uppercase;
                     letter-spacing: 0.06em; color: var(--muted); }
.level-chip-detail { font-size: 0.8em; color: var(--muted); }
.level-chip-pass { border-color: #aedeb6; background: #f1faf3; }
.level-chip-pass .level-chip-status { color: var(--pass); }
.level-chip-fail { border-color: #f1b0b1; background: #fdf3f3; }
.level-chip-fail .level-chip-status { color: var(--fail); }
.level-chip-empty { opacity: 0.5; }
.summary {
  display: grid; grid-template-columns: 1fr 2fr; gap: 1.5em; align-items: start;
}
@media (max-width: 760px) { .summary { grid-template-columns: 1fr; } }
.card {
  background: var(--card); border: 1px solid var(--line);
  border-radius: 10px; padding: 1.1em 1.2em;
}
.card.tight { padding: 0.8em 0.9em; }
.block-grid {
  display: grid; grid-template-columns: minmax(280px, 1fr) minmax(0, 2fr);
  gap: 1em; align-items: start;
}
@media (max-width: 760px) { .block-grid { grid-template-columns: 1fr; } }
.block-head {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.8em;
  margin-bottom: 0.3em;
}
.block-tag {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-weight: 600; font-size: 0.95em;
  padding: 0.15em 0.55em; border-radius: 6px;
  background: var(--bg);
}
.block-head .pass-pill, .block-head .fail-pill {
  font-size: 0.72em; text-transform: uppercase; letter-spacing: 0.08em;
  padding: 0.2em 0.55em; border-radius: 999px;
}
.pass-pill { background: #e9f7ef; color: var(--pass); }
.fail-pill { background: #fbe9e7; color: var(--fail); }
.block-stats {
  font-size: 0.85em; color: var(--muted); display: flex; gap: 0.8em;
  flex-wrap: wrap;
}
.block-stats b { color: var(--ink); }
.confusion-row { display: flex; align-items: center; gap: 1.2em; }
table.report {
  border-collapse: collapse; width: 100%; font-size: 0.92em;
}
table.report th, table.report td {
  padding: 0.45em 0.7em; border-bottom: 1px solid var(--line); text-align: left;
}
table.report th {
  background: var(--bg); color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.75em;
}
table.report tbody tr:hover { background: var(--bg); }
table.report .num { text-align: right; font-variant-numeric: tabular-nums; }
.legend {
  font-size: 0.85em; color: var(--muted); margin: 0.4em 0 0.8em;
  display: flex; flex-wrap: wrap; gap: 0.8em;
}
.legend .swatch {
  display: inline-block; width: 12px; height: 12px; border-radius: 3px;
  vertical-align: middle; margin-right: 4px;
}
.back-link { font-size: 0.9em; }
.back-link a { color: var(--accent); text-decoration: none; }
.back-link a:hover { text-decoration: underline; }
hr { border: none; border-top: 1px solid var(--line); margin: 2em 0; }
"""


def _legend_html() -> str:
    return (
        '<div class="legend">'
        + "".join(
            f'<span><span class="swatch" style="background:{OUTCOME_COLOURS[k]}"></span>{label}</span>'
            for k, label in OUTCOME_LABEL.items()
        )
        + "</div>"
    )


def render_report(run: dict, out_path: Path) -> None:
    blocks_by_n = {}
    for b in run["blocks"]:
        blocks_by_n.setdefault(b["N"], []).append(b)

    # Summary plot — RT distribution per N (hits + FAs use a real key press).
    rt_fig = _rt_distribution_figure(run["main_trials"])

    # Per-block timeline plots, indexed by ordering.
    block_figs = {(b["block"], b["N"]): _block_timeline_figure(b)
                  for b in run["blocks"]}

    fig_objects = [rt_fig] + [f for f in block_figs.values() if f is not None]
    fig_objects = [f for f in fig_objects if f is not None]
    script, divs = components(fig_objects) if fig_objects else ("", ())

    # Map from figure -> div in the order we passed them.
    div_iter = iter(divs)
    rt_div = next(div_iter) if rt_fig is not None else ""
    block_divs = {key: (next(div_iter) if fig is not None else "")
                  for key, fig in block_figs.items()}

    # Header.
    head_kv = [
        ("Participant", run["participant"]),
        ("Session", run["session"] or "—"),
        ("Date", run["date_str"] or "—"),
        ("Duration", _format_duration(run["duration_s"])),
        ("Blocks setting", run["n_blocks_setting"] or "—"),
        ("topN setting", run["top_n_setting"] or "—"),
        ("PsychoPy", run["psychopy_version"] or "—"),
        ("Max N reached", str(run["max_n_reached"])),
    ]

    aggregate = Counter()
    for b in run["blocks"]:
        aggregate["hit"] += b["n_hits"]
        aggregate["miss"] += b["n_misses"]
        aggregate["fa"] += b["n_fa"]
        aggregate["cr"] += b["n_cr"]

    aggregate_svg = _confusion_svg(
        aggregate["hit"], aggregate["miss"],
        aggregate["fa"], aggregate["cr"],
    )

    # Per-N RT means table.
    per_n_rows = []
    for n in sorted(blocks_by_n):
        bs = blocks_by_n[n]
        all_rts = []
        n_hits = n_misses = n_fa = n_cr = 0
        for b in bs:
            n_hits += b["n_hits"]
            n_misses += b["n_misses"]
            n_fa += b["n_fa"]
            n_cr += b["n_cr"]
            for t in b["trials"]:
                if t["pressed"] and np.isfinite(t["rt"]):
                    all_rts.append(t["rt"])
        n_total_targets = n_hits + n_misses
        n_total_nontargets = n_fa + n_cr
        per_n_rows.append({
            "n": n,
            "blocks": len(bs),
            "hit_rate": n_hits / max(n_total_targets, 1),
            "fa_rate": n_fa / max(n_total_nontargets, 1),
            "rt_mean": float(np.mean(all_rts)) if all_rts else None,
            "rt_median": float(np.median(all_rts)) if all_rts else None,
            "n_pressed": len(all_rts),
        })

    # Build per-block cards.
    block_cards = []
    for b in sorted(run["blocks"], key=lambda x: (x["block"], x["N"])):
        cm = _confusion_svg(b["n_hits"], b["n_misses"], b["n_fa"], b["n_cr"],
                            cell=78)
        block_div = block_divs.get((b["block"], b["N"]), "")
        verdict = (
            '<span class="fail-pill">failed</span>' if b["failed"]
            else '<span class="pass-pill">passed</span>'
        )
        if b["n_trials"] == 0:
            verdict = '<span class="fail-pill">no trials</span>'
        block_cards.append(f"""
        <article class="card">
          <div class="block-head">
            <span class="block-tag" style="border-left:4px solid {N_COLOURS[b['N']]};padding-left:0.6em">
              Block {b['block']} &middot; {b['N']}-back &middot; list {html.escape(b['list_name'])}.csv
            </span>
            {verdict}
          </div>
          <div class="block-stats">
            <span><b>{b['n_trials']}</b> trials</span>
            <span><b>{b['n_hits']}</b> hits</span>
            <span><b>{b['n_misses']}</b> misses</span>
            <span><b>{b['n_fa']}</b> FAs</span>
            <span><b>{b['n_cr']}</b> CRs</span>
            <span>Miss rate: <b>{b['miss_rate']*100:.1f}%</b></span>
            <span>FA rate: <b>{b['fa_rate']*100:.1f}%</b></span>
          </div>
          <div class="block-grid">
            <div>{cm}</div>
            <div>{block_div}</div>
          </div>
        </article>
        """)

    # Per-N RT table.
    per_n_html = "".join(
        f'<tr>'
        f'<td><span style="color:{N_COLOURS[r["n"]]};font-weight:600">{r["n"]}-back</span></td>'
        f'<td class="num">{r["blocks"]}</td>'
        f'<td class="num">{r["hit_rate"]*100:.1f}%</td>'
        f'<td class="num">{r["fa_rate"]*100:.1f}%</td>'
        f'<td class="num">{f"{r["rt_mean"]:.3f}" if r["rt_mean"] is not None else "—"}</td>'
        f'<td class="num">{f"{r["rt_median"]:.3f}" if r["rt_median"] is not None else "—"}</td>'
        f'<td class="num">{r["n_pressed"]}</td>'
        f'</tr>'
        for r in per_n_rows
    )

    head_kv_html = "".join(
        f'<div><span class="label">{html.escape(k)}</span>'
        f'<span class="value">{html.escape(str(v))}</span></div>'
        for k, v in head_kv
    )

    body = f"""
<div class="wrap">
  <p class="back-link"><a href="../index.html">&larr; All reports</a></p>
  <h1>{html.escape(run['participant'])} <span style="color:var(--muted);font-weight:400">&middot; {html.escape(run['date_str'] or '')}</span></h1>
  <p class="meta">{html.escape(run['csv_name'])}</p>
  <div class="kv">{head_kv_html}</div>

  {_level_strip_svg(blocks_by_n, run['failed_levels'])}

  <h2>Aggregate</h2>
  <div class="summary">
    <div class="card">
      <h3>Confusion matrix</h3>
      <div class="confusion-row">{aggregate_svg}</div>
      <h3 style="margin-top:1em">By level</h3>
      <table class="report">
        <thead><tr>
          <th>Level</th><th class="num">Blocks</th>
          <th class="num">Hit rate</th><th class="num">FA rate</th>
          <th class="num">Mean RT (s)</th><th class="num">Median RT (s)</th>
          <th class="num">Pressed</th>
        </tr></thead>
        <tbody>{per_n_html}</tbody>
      </table>
    </div>
    <div class="card">
      <h3>Response time distribution</h3>
      <p class="meta" style="margin-top:-0.2em">Histogram + Gaussian KDE for every key press, by N-back level. Click a legend entry to hide that level.</p>
      {rt_div}
    </div>
  </div>

  <h2>Blocks ({len(run['blocks'])})</h2>
  {_legend_html()}
  {''.join(block_cards)}
</div>
"""

    timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(run['participant'])} — N-back report</title>
{INLINE.render()}
<style>{CSS}</style>
</head>
<body>
{body}
{script}
<footer style="text-align:center;color:var(--muted);font-size:0.8em;padding:2em 1em">
Generated {html.escape(timestamp)} by report.py
</footer>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page)


# --- Index --------------------------------------------------------------------


def render_index(runs: list[dict], index_path: Path) -> None:
    runs_sorted = sorted(
        runs, key=lambda r: r.get("date_str") or "", reverse=True,
    )

    rows = []
    for r in runs_sorted:
        report_href = f"reports/{r['report_slug']}.html"
        max_n = r["max_n_reached"]
        max_color = N_COLOURS.get(max_n, "#3498db") if max_n else "#bdc3c7"
        n_blocks_ran = len({(b["block"], b["N"]) for b in r["blocks"]})
        rows.append(f"""
        <tr>
          <td><a href="{html.escape(report_href)}" style="color:var(--accent);text-decoration:none">{html.escape(r['participant'])}</a></td>
          <td>{html.escape(r['session'] or '—')}</td>
          <td>{html.escape(r['date_str'] or '')}</td>
          <td class="num">{_format_duration(r['duration_s'])}</td>
          <td><span style="color:{max_color};font-weight:700">{max_n}-back</span></td>
          <td class="num">{n_blocks_ran}</td>
          <td><a href="{html.escape(report_href)}" style="color:var(--accent);text-decoration:none">open &rarr;</a></td>
        </tr>
        """)

    timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>N-back participant reports</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>N-back participant reports</h1>
  <p class="meta">{len(runs_sorted)} run(s) &middot; <a href="report.html" style="color:var(--accent)">prerandomization report &rarr;</a></p>
  <div class="card" style="padding:0;overflow:hidden">
    <table class="report">
      <thead><tr>
        <th>Participant</th><th>Session</th><th>Date</th>
        <th class="num">Duration</th><th>Max N reached</th>
        <th class="num">Blocks ran</th><th></th>
      </tr></thead>
      <tbody>{''.join(rows) if rows else '<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:2em">No reports yet.</td></tr>'}</tbody>
    </table>
  </div>
</div>
<footer style="text-align:center;color:var(--muted);font-size:0.8em;padding:2em 1em">
Index updated {html.escape(timestamp)}
</footer>
</body>
</html>
"""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(page)


# --- Main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR,
                        help="data directory containing PsychoPy CSVs")
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR,
                        help="output directory for per-participant reports")
    parser.add_argument("--index", type=Path, default=INDEX_PATH,
                        help="path to write the index HTML")
    parser.add_argument("--csv", type=Path, action="append",
                        help="generate a report for this CSV only "
                             "(may be passed multiple times)")
    args = parser.parse_args()

    csv_paths = (args.csv if args.csv
                 else sorted(args.data.glob("*.csv")))
    if not csv_paths:
        print(f"No CSVs in {args.data}")
        return

    runs = []
    skipped = []
    for path in csv_paths:
        try:
            run = load_run(path)
        except Exception as exc:
            print(f"  ! failed to load {path.name}: {exc}")
            continue
        if run is None:
            skipped.append(path.name)
            continue
        out_path = args.reports / f"{run['report_slug']}.html"
        render_report(run, out_path)
        runs.append(run)
        print(f"  + {path.name} -> {out_path.relative_to(out_path.parent.parent)}")

    if skipped:
        print(f"  - skipped {len(skipped)} run(s) with no main trials")

    # Always rebuild the index from whatever finished runs we have so it's
    # complete (not just additive); pick up any reports already in
    # docs/reports/ that may have been generated by an older invocation.
    if not args.csv:
        # Full rebuild: only the runs we just saw represent the current state.
        render_index(runs, args.index)
    else:
        # Subset run: merge with existing reports by re-loading every CSV.
        all_runs = []
        for p in sorted(args.data.glob("*.csv")):
            try:
                r = load_run(p)
            except Exception:
                continue
            if r is not None:
                all_runs.append(r)
        render_index(all_runs, args.index)
    print(f"  index -> {args.index}")


if __name__ == "__main__":
    main()
