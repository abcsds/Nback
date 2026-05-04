#!/usr/bin/env python3
"""Pre-randomize letter sequences for the N-back experiment.

Generates CSV files under lists/, one per (N, list-letter) combination, plus
training lists. Each row is a single trial: a letter and whether it is a
target (matches the letter shown N positions earlier).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import html
import random
import re
import string
from collections import Counter
from pathlib import Path

# --- Experiment parameters (mirror the PsychoPy task) -----------------------

ALL_LETTERS = [c for c in string.ascii_uppercase if c != "N"]  # N omitted: confused testers
TRAINING_LETTERS = list("ABCDEFGH")  # smaller alphabet so training is easier

PRESENTATION_DURATION = 1.2  # seconds the letter is on screen
ITI = 0.6                    # inter-trial interval in seconds (fix routine: 0.3 s cross + 0.3 s blank)
TRIAL_DURATION = PRESENTATION_DURATION + ITI  # 1.8 s

BLOCK_DURATION = 64.8      # seconds per main block (36 trials × 1.8 s)
TRAINING_DURATION = 21.6   # seconds per training block (12 trials × 1.8 s)

TARGET_RATIO = 0.25  # exactly 25% of trials are targets

N_LEVELS = [1, 2, 3, 4, 5]
LISTS_PER_LEVEL = 10
TRAINING_LEVELS = [1, 2, 3]

INSTRUCTIONS_DURATION = 5 * 60  # seconds reserved for the instructions screen
INTER_BLOCK_BREAK = 60          # 1-min break between every pair of consecutive lists

DEFAULT_SEED = 20260427  # change this to regenerate; LOG.md records each new seed

N_PARTICIPANTS = 1000        # 3-digit codes 000..999
# MAX_BLOCKS == LISTS_PER_LEVEL is load-bearing: each block consumes one
# list-letter per N, and the per-N permutation is exactly LISTS_PER_LEVEL
# letters long, so this guarantees no list-letter is reused for the same N
# within a participant even in the no-failure worst case.
MAX_BLOCKS = LISTS_PER_LEVEL  # 10

# --- Helpers ----------------------------------------------------------------


def n_stimuli(duration: float) -> int:
    raw = duration / TRIAL_DURATION
    n = round(raw)
    assert abs(raw - n) < 1e-9, (
        f"Block duration {duration}s is not a multiple of trial duration {TRIAL_DURATION}s"
    )
    return n


def n_targets_for(length: int) -> int:
    raw = length * TARGET_RATIO
    n = round(raw)
    assert abs(raw - n) < 1e-9, (
        f"List length {length} does not allow exactly {TARGET_RATIO:.0%} targets"
    )
    return n


def _sample_non_adjacent(rng: random.Random, low: int, high: int, k: int) -> list[int]:
    """Sample k distinct integers from [low, high) with no two consecutive.

    Bijection: any k-subset of {0..m-k} maps to a non-adjacent k-subset of
    {0..m-1} by adding the rank-index to each sorted pick. Using a uniform
    sample on the smaller range gives a uniform sample on the constrained
    one. (m = high - low.)
    """
    m = high - low
    max_k = (m + 1) // 2
    assert k <= max_k, (
        f"cannot place {k} non-adjacent targets in {m} slots (max {max_k})"
    )
    raw = sorted(rng.sample(range(m - k + 1), k))
    return [low + r + i for i, r in enumerate(raw)]


def generate_sequence(rng: random.Random, n_back: int, length: int, letters: list[str]):
    """Build a sequence with exactly n_targets matches at distance n_back.

    Targets are placed first by choosing positions in [n_back, length) with
    no two consecutive (to avoid attentional-blink confounds); each target
    position copies the letter from n_back trials earlier. Non-target
    positions pick a letter different from the one n_back back, so they
    cannot accidentally form a target.
    """
    n_t = n_targets_for(length)
    target_positions = set(_sample_non_adjacent(rng, n_back, length, n_t))

    seq: list[str] = []
    targets: list[bool] = []
    for i in range(length):
        if i < n_back:
            letter = rng.choice(letters)
            is_target = False
        else:
            nback_letter = seq[i - n_back]
            if i in target_positions:
                letter = nback_letter
                is_target = True
            else:
                letter = rng.choice([c for c in letters if c != nback_letter])
                is_target = False
        seq.append(letter)
        targets.append(is_target)
    return seq, targets


def verify(seq, targets, n_back: int, expected_length: int, expected_targets: int, letters):
    assert len(seq) == expected_length, f"length {len(seq)} != {expected_length}"
    assert len(targets) == expected_length
    assert all(c in letters for c in seq), "letter outside alphabet"
    assert sum(targets) == expected_targets, (
        f"target count {sum(targets)} != {expected_targets}"
    )
    for i in range(n_back):
        assert targets[i] is False, f"position {i} cannot be a target (only {n_back} prior trials)"
    for i in range(n_back, expected_length):
        actual_match = seq[i] == seq[i - n_back]
        assert actual_match == targets[i], (
            f"position {i}: target flag {targets[i]} disagrees with letter match {actual_match}"
        )
    for i in range(1, expected_length):
        assert not (targets[i] and targets[i - 1]), (
            f"adjacent targets at positions {i - 1} and {i} (attentional-blink risk)"
        )


def write_csv(path: Path, seq, targets):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["letter", "target"])
        for letter, is_target in zip(seq, targets):
            w.writerow([letter, "true" if is_target else "false"])


def generate_schedule(rng, n_levels, max_blocks, lists_per_level):
    """One participant's schedule: max_blocks * len(n_levels) rows.

    For each N, the list-letters are a single permutation of [a..j] used
    without replacement across max_blocks blocks. Rows are emitted
    block-major (block 1's N=1..5 first, then block 2, ...).
    """
    letters = [chr(ord("a") + i) for i in range(lists_per_level)]
    perms = {n: rng.sample(letters, lists_per_level) for n in n_levels}
    rows = []
    for block in range(1, max_blocks + 1):
        for n in n_levels:
            letter = perms[n][block - 1]
            rows.append((block, n, letter, f"lists/{n}{letter}.csv"))
    return rows


def verify_schedule(rows, n_levels, max_blocks, lists_per_level):
    assert len(rows) == max_blocks * len(n_levels), (
        f"schedule has {len(rows)} rows, expected {max_blocks * len(n_levels)}"
    )
    # Block-major ordering is load-bearing: the experiment consumes rows in
    # order with no re-sort, so block_i / N=1..k must precede block_{i+1}.
    expected_order = sorted(rows, key=lambda r: (r[0], r[1]))
    assert rows == expected_order, "rows must be ordered (block, N) ascending"
    expected_letters = {chr(ord("a") + i) for i in range(lists_per_level)}
    by_n = {n: [] for n in n_levels}
    for block, n, letter, conds in rows:
        assert 1 <= block <= max_blocks
        assert n in n_levels
        assert conds == f"lists/{n}{letter}.csv"
        by_n[n].append(letter)
    for n in n_levels:
        assert set(by_n[n]) == expected_letters, (
            f"N={n}: letters {expected_letters - set(by_n[n])} missing"
        )
        assert len(by_n[n]) == lists_per_level, (
            f"N={n}: list-letter reused"
        )


def write_schedule(path, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "N", "list_letter", "condsFile"])
        w.writerows(rows)


def read_last_seed(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    pattern = re.compile(r"seed:\s*(-?\d+)")
    last = None
    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            last = int(m.group(1))
    return last


def _svg_timeline(seq, targets, cell_w: int = 24, cell_h: int = 28, gap: int = 2) -> str:
    n = len(seq)
    width = n * (cell_w + gap)
    height = cell_h + 18
    parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    for i, (letter, is_t) in enumerate(zip(seq, targets)):
        x = i * (cell_w + gap)
        fill = "#e74c3c" if is_t else "#ecf0f1"
        text_color = "#ffffff" if is_t else "#2c3e50"
        parts.append(
            f'<rect x="{x}" y="0" width="{cell_w}" height="{cell_h}" fill="{fill}" rx="3"/>'
            f'<text x="{x + cell_w / 2:.1f}" y="{cell_h / 2 + 5:.1f}" '
            f'text-anchor="middle" fill="{text_color}" '
            f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
            f'font-size="13" font-weight="600">{html.escape(letter)}</text>'
        )
        if i % 5 == 0 or i == n - 1:
            parts.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{cell_h + 14}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="10" '
                f'fill="#7f8c8d">{i}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_bars(items, max_val=None, label_w: int = 110, bar_max: int = 320,
              row_h: int = 22, color: str = "#3498db") -> str:
    if not items:
        return ""
    if max_val is None:
        max_val = max(v for _, v in items) or 1
    width = label_w + bar_max + 60
    height = row_h * len(items) + 4
    parts = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    for i, (label, value) in enumerate(items):
        y = i * row_h
        bar_w = (value / max_val) * bar_max
        parts.append(
            f'<text x="{label_w - 6}" y="{y + row_h / 2 + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12" fill="#2c3e50">'
            f'{html.escape(str(label))}</text>'
            f'<rect x="{label_w}" y="{y + 3}" width="{bar_w:.1f}" height="{row_h - 6}" '
            f'fill="{color}" rx="2"/>'
            f'<text x="{label_w + bar_w + 5:.1f}" y="{y + row_h / 2 + 4:.1f}" '
            f'font-family="sans-serif" font-size="12" fill="#2c3e50">{value}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s" if sec else f"{h}h {m:02d}m"
    return f"{m}m {sec:02d}s" if sec else f"{m}m"


def _task_time(k: int, max_n: int) -> float | None:
    """Seconds for training + main blocks + inter-list breaks (no instructions).

    Topping out at `max_n` < 5 means the participant attempted (and failed)
    N=max_n+1, which counts as one extra main list (and one extra training
    if max_n+1 <= 3). After that failure cap drops to max_n; subsequent
    blocks only run levels 1..max_n.

    Topping out at max_n=1 forces the experiment to end (cap=1 < 2), so it's
    only reachable with K=1; for K>1 with max_n=1 we return None.
    """
    if max_n == 1 and k > 1:
        return None  # unreachable: failing N=2 ends the experiment immediately
    if max_n < max(N_LEVELS):
        # +1 failed attempt at N=max_n+1; trainings up to that level if <=3.
        train_blocks = min(max_n + 1, len(TRAINING_LEVELS))
        main_blocks = max_n * k + 1
    else:
        # Reached the ceiling without failing: K full sweeps, no extra attempt.
        train_blocks = len(TRAINING_LEVELS)
        main_blocks = max_n * k
    total_blocks = train_blocks + main_blocks
    breaks = max(total_blocks - 1, 0)
    return (
        train_blocks * TRAINING_DURATION
        + main_blocks * BLOCK_DURATION
        + breaks * INTER_BLOCK_BREAK
    )


def _schedule_table_html() -> str:
    max_level = max(N_LEVELS)
    head = (
        "<thead>"
        '<tr>'
        '<th rowspan="2">Lists per N-back level (K)</th>'
        '<th rowspan="2">Instructions</th>'
        f'<th colspan="{len(N_LEVELS)}">Task time if participant tops out at N-back =</th>'
        "</tr>"
        "<tr>"
        + "".join(
            f'<th>{n}{" (max)" if n == max_level else ""}</th>' for n in N_LEVELS
        )
        + "</tr></thead>"
    )
    instr = _format_duration(INSTRUCTIONS_DURATION)
    body_rows = []
    for k in range(1, LISTS_PER_LEVEL + 1):
        cells = [f"<td>{k}</td>", f"<td>{instr}</td>"]
        for max_n in N_LEVELS:
            cls = ' class="max-col"' if max_n == max_level else ""
            seconds = _task_time(k, max_n)
            cell = _format_duration(seconds) if seconds is not None else "&mdash;"
            cells.append(f"<td{cls}>{cell}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        '<table class="schedule">'
        f"{head}<tbody>{''.join(body_rows)}</tbody></table>"
    )


def render_html_report(report_path: Path, runs: dict, seed: int) -> None:
    """Write a single-page HTML report with one tab per N-back level.

    `runs` maps n_back -> {'main': [(name, seq, targets), ...],
                           'training': (name, seq, targets) or None}.
    """
    timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    n_levels = sorted(runs.keys())

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1200px; margin: 1em auto; padding: 0 1em; color: #2c3e50; }
    h1 { margin-bottom: 0.2em; }
    h2 { margin-top: 0.5em; }
    h3 { margin-top: 1.4em; margin-bottom: 0.4em; }
    .meta { color: #7f8c8d; font-size: 0.9em; margin-bottom: 1em; }
    .legend { margin: 0.5em 0 1em; font-size: 0.9em; color: #34495e; }
    .legend .swatch { display: inline-block; width: 14px; height: 14px;
                      border-radius: 3px; vertical-align: middle; margin-right: 4px; }
    .tab-bar { display: flex; gap: 4px; border-bottom: 2px solid #ecf0f1;
               margin-bottom: 1em; flex-wrap: wrap; }
    .tab-button { padding: 10px 20px; border: none; background: none;
                  cursor: pointer; font-size: 14px; color: #7f8c8d;
                  border-radius: 4px 4px 0 0; }
    .tab-button:hover { background: #ecf0f1; }
    .tab-button.active { background: #3498db; color: white; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .summary { display: flex; gap: 2em; flex-wrap: wrap; }
    .summary > div { flex: 1; min-width: 320px; }
    .list-block { margin: 0.6em 0; padding: 0.6em 1em;
                  border: 1px solid #ecf0f1; border-radius: 6px; background: #fafbfc;
                  overflow-x: auto; }
    .list-block.training { background: #fef9e7; border-color: #f5d76e; }
    .list-header { display: flex; justify-content: space-between;
                   align-items: baseline; margin-bottom: 0.3em; gap: 1em; }
    .list-name { font-weight: 600;
                 font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .list-stats { color: #7f8c8d; font-size: 0.85em; }
    .schedule { border-collapse: collapse; margin: 0.5em 0 1em;
                font-family: -apple-system, sans-serif; font-size: 13px; }
    .schedule th, .schedule td { padding: 6px 14px;
                                  border-bottom: 1px solid #ecf0f1; text-align: right; }
    .schedule th:first-child, .schedule td:first-child { text-align: left; }
    .schedule th { background: #ecf0f1; font-weight: 600; }
    .schedule tbody tr:hover { background: #f4f6f7; }
    .schedule .max-col { background: #fdf2e3; }
    .schedule-note { color: #7f8c8d; font-size: 0.85em; margin: 0 0 1.5em; }
    """

    js = """
    document.addEventListener('click', function (e) {
      if (!e.target.matches('.tab-button')) return;
      var tab = e.target.dataset.tab;
      document.querySelectorAll('.tab-button').forEach(function (b) {
        b.classList.toggle('active', b.dataset.tab === tab);
      });
      document.querySelectorAll('.tab-panel').forEach(function (p) {
        p.classList.toggle('active', p.id === 'tab-' + tab);
      });
    });
    """

    legend_html = (
        '<div class="legend">'
        '<span class="swatch" style="background:#ecf0f1"></span>non-target &nbsp;'
        '<span class="swatch" style="background:#e74c3c"></span>target'
        "</div>"
    )

    tab_buttons = []
    tab_panels = []
    for idx, n in enumerate(n_levels):
        active_cls = " active" if idx == 0 else ""
        tab_buttons.append(
            f'<button class="tab-button{active_cls}" data-tab="{n}">{n}-back</button>'
        )

        main_lists = runs[n]["main"]
        training = runs[n].get("training")

        total_targets = sum(sum(t) for _, _, t in main_lists)
        total_non = sum(len(t) - sum(t) for _, _, t in main_lists)
        total = total_targets + total_non
        target_pct = 100 * total_targets / total if total else 0
        non_pct = 100 - target_pct

        target_bar = _svg_bars(
            [
                (f"target ({target_pct:.1f}%)", total_targets),
                (f"non-target ({non_pct:.1f}%)", total_non),
            ],
            color="#e74c3c",
        )

        letter_counter: Counter = Counter()
        for _, seq, _ in main_lists:
            letter_counter.update(seq)
        letter_items = sorted(letter_counter.items())
        letter_bar = _svg_bars(letter_items, color="#9b59b6", label_w=40, bar_max=380)

        per_list = []
        for name, seq, tgs in main_lists:
            n_t = sum(tgs)
            per_list.append(
                f'<div class="list-block">'
                f'<div class="list-header">'
                f'<span class="list-name">{html.escape(name)}.csv</span>'
                f'<span class="list-stats">{len(seq)} trials &middot; '
                f'{n_t} targets ({100 * n_t / len(seq):.1f}%)</span>'
                f"</div>{_svg_timeline(seq, tgs)}</div>"
            )

        training_html = ""
        if training is not None:
            tname, tseq, ttg = training
            n_t = sum(ttg)
            training_html = (
                "<h3>Training list</h3>"
                f'<div class="list-block training">'
                f'<div class="list-header">'
                f'<span class="list-name">{html.escape(tname)}.csv</span>'
                f'<span class="list-stats">{len(tseq)} trials &middot; '
                f'{n_t} targets ({100 * n_t / len(tseq):.1f}%) &middot; '
                f"alphabet: {''.join(sorted(set(tseq)))}</span>"
                f"</div>{_svg_timeline(tseq, ttg)}</div>"
            )

        panel = (
            f'<div class="tab-panel{active_cls}" id="tab-{n}">'
            f"<h2>{n}-back</h2>"
            '<div class="summary">'
            "<div>"
            "<h3>Target / non-target (expected 25 / 75)</h3>"
            f"{target_bar}"
            "</div>"
            "<div>"
            f"<h3>Letter frequency across all {len(main_lists)} lists</h3>"
            f"{letter_bar}"
            "</div>"
            "</div>"
            f"<h3>Per-list timelines ({len(main_lists)} lists)</h3>"
            f"{''.join(per_list)}"
            f"{training_html}"
            "</div>"
        )
        tab_panels.append(panel)

    n_main = sum(len(runs[n]["main"]) for n in n_levels)
    n_train = sum(1 for n in n_levels if runs[n].get("training"))

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NBack lists report</title>
<style>{css}</style>
</head>
<body>
<h1>NBack lists report</h1>
<p class="meta">Generated {timestamp} &middot; seed: {seed} &middot;
{n_main} main lists &middot; {n_train} training lists</p>
{legend_html}
<h2>Experiment schedule</h2>
<p class="schedule-note">A participant who fails &gt;50% of a level does not
proceed to the next, so the task time depends on the highest N-back level they
complete. The rightmost column ({max(N_LEVELS)}-back) is the upper bound; in
practice most participants top out around 4-back. Topping out at <em>X</em>
&lt; 5 means the participant attempted N=<em>X</em>+1 and failed it, so each
non-max cell includes one extra main list ({BLOCK_DURATION:g} s) plus its
training if <em>X</em>+1 &le; 3. The &ldquo;1-back&rdquo; column has only the
<em>K</em>=1 cell because failing N=2 drops the cap below 2 and ends the
experiment immediately, so <em>K</em>&gt;1 with max-N=1 is unreachable
(&mdash;). The &ldquo;Instructions&rdquo; column is fixed
({int(INSTRUCTIONS_DURATION / 60)} min); the per-level columns cover, for each
level reached (and the failed attempt where applicable), the corresponding
training list ({TRAINING_DURATION:g} s; only 1-, 2-, and 3-back have training),
the main lists ({BLOCK_DURATION:g} s each), and a {int(INTER_BLOCK_BREAK)} s
break between every consecutive list. Add the &ldquo;Instructions&rdquo;
column to a per-level cell to get the wall-clock total.</p>
{_schedule_table_html()}
<h3>Participant schedules</h3>
<p><code>prerandomize.py</code> also writes {N_PARTICIPANTS} per-participant
schedules (<code>schedules/000.csv</code> &hellip; <code>schedules/{N_PARTICIPANTS - 1:03d}.csv</code>),
one for each possible 3-digit participant code. Each schedule has
{MAX_BLOCKS * len(N_LEVELS)} rows = {MAX_BLOCKS} blocks &times; {len(N_LEVELS)} N-levels,
with the {LISTS_PER_LEVEL} list-letters per N-level used in a permuted order.
The experiment loads <code>schedules/{{participant}}.csv</code> at runtime and
skips rows where the cap has dropped below the row's N or where the row's
block exceeds the experimenter-set <code>nBlocks</code>.</p>
<div class="tab-bar">{''.join(tab_buttons)}</div>
{''.join(tab_panels)}
<script>{js}</script>
</body>
</html>
"""
    report_path.write_text(doc)


def append_log(log_path: Path, seed: int) -> None:
    timestamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    entry = f"- {timestamp} — seed: {seed}\n"
    if not log_path.exists():
        header = (
            "# Pre-randomization seed log\n\n"
            "Each entry records a run of `prerandomize.py` whose seed differed\n"
            "from the previous run. This is the audit trail for reproducibility:\n"
            "all CSVs in `lists/` were generated by the most recent seed below.\n\n"
        )
        log_path.write_text(header + entry)
    else:
        with log_path.open("a") as f:
            f.write(entry)


# --- Main -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--output", type=Path, default=Path("lists"),
                        help="output directory for CSV files (default: lists)")
    parser.add_argument("--log", type=Path, default=Path("LOG.md"),
                        help="seed log file (default: LOG.md)")
    parser.add_argument("--report", type=Path, default=Path("docs/report.html"),
                        help="HTML report file (default: docs/report.html)")
    parser.add_argument("--schedules", type=Path, default=Path("schedules"),
                        help="output directory for participant schedules (default: schedules)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.schedules.mkdir(parents=True, exist_ok=True)

    block_length = n_stimuli(BLOCK_DURATION)
    block_targets = n_targets_for(block_length)
    train_length = n_stimuli(TRAINING_DURATION)
    train_targets = n_targets_for(train_length)

    master_rng = random.Random(args.seed)

    runs: dict[int, dict] = {n: {"main": [], "training": None} for n in N_LEVELS}

    # Main lists: 10 per N-back level, named e.g. 1a.csv ... 5j.csv
    block_lengths_seen = []
    for n_back in N_LEVELS:
        for i in range(LISTS_PER_LEVEL):
            sub_seed = master_rng.getrandbits(64)
            rng = random.Random(sub_seed)
            seq, targets = generate_sequence(rng, n_back, block_length, ALL_LETTERS)
            verify(seq, targets, n_back, block_length, block_targets, ALL_LETTERS)
            block_lengths_seen.append(len(seq))
            suffix = chr(ord("a") + i)
            name = f"{n_back}{suffix}"
            write_csv(args.output / f"{name}.csv", seq, targets)
            runs[n_back]["main"].append((name, seq, targets))

    assert len(set(block_lengths_seen)) == 1, (
        f"main lists vary in length: {sorted(set(block_lengths_seen))}"
    )
    assert len(block_lengths_seen) == len(N_LEVELS) * LISTS_PER_LEVEL

    # Training lists: one per N for 1, 2, 3-back
    train_lengths_seen = []
    for n_back in TRAINING_LEVELS:
        sub_seed = master_rng.getrandbits(64)
        rng = random.Random(sub_seed)
        seq, targets = generate_sequence(rng, n_back, train_length, TRAINING_LETTERS)
        verify(seq, targets, n_back, train_length, train_targets, TRAINING_LETTERS)
        train_lengths_seen.append(len(seq))
        name = f"train_{n_back}"
        write_csv(args.output / f"{name}.csv", seq, targets)
        runs[n_back]["training"] = (name, seq, targets)

    assert len(set(train_lengths_seen)) == 1, (
        f"training lists vary in length: {sorted(set(train_lengths_seen))}"
    )

    # Per-participant schedules (3-digit IDs 000..999)
    for pid in range(N_PARTICIPANTS):
        sub_seed = master_rng.getrandbits(64)
        sub_rng = random.Random(sub_seed)
        rows = generate_schedule(sub_rng, N_LEVELS, MAX_BLOCKS, LISTS_PER_LEVEL)
        verify_schedule(rows, N_LEVELS, MAX_BLOCKS, LISTS_PER_LEVEL)
        write_schedule(args.schedules / f"{pid:03d}.csv", rows)

    render_html_report(args.report, runs, args.seed)

    last_seed = read_last_seed(args.log)
    if last_seed != args.seed:
        append_log(args.log, args.seed)
        log_msg = f"seed changed ({last_seed} -> {args.seed}); appended to {args.log}"
    else:
        log_msg = f"seed unchanged ({args.seed}); {args.log} not updated"

    print(
        f"Wrote {len(N_LEVELS) * LISTS_PER_LEVEL} main lists "
        f"({block_length} trials, {block_targets} targets each) "
        f"and {len(TRAINING_LEVELS)} training lists "
        f"({train_length} trials, {train_targets} targets each) to {args.output}/"
    )
    print(f"Wrote report to {args.report}")
    print(f"Wrote {N_PARTICIPANTS} participant schedules to {args.schedules}/")
    print(log_msg)


if __name__ == "__main__":
    main()
