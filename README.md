# N-back task

N-back task for cognitive mental workload implemented in PsychoPy. Written in
Spanish using letters of the English alphabet.

## Parameters

| Parameter | Value |
| --- | --- |
| Letter alphabet (main) | ASCII uppercase minus `N` (25 letters; `N` was removed because testers confused it with the N-back instruction) |
| Letter alphabet (training) | `A B C D E F G H` (8-letter subset, easier for first exposure) |
| Letter presentation duration | **2.0 s** |
| Inter-trial interval (ITI) | **0.5 s** |
| Trial duration | 2.5 s (presentation + ITI) |
| Main block duration | **90 s** → 36 trials per list |
| Training block duration | **30 s** → 12 trials per list |
| Target probability | **25%** (exact, not probabilistic) — 9 targets per main list, 3 per training list |
| N-back levels (main) | 1, 2, 3, 4, 5 |
| Lists per level (main) | 10 |
| N-back levels (training) | 1, 2, 3 (one list each) |

A *target* is a trial whose letter matches the letter shown N positions
earlier. Non-target trials are guaranteed to differ from the letter N back, so
they cannot accidentally form a target.

## Pre-randomization

All letter sequences are generated ahead of time and stored as CSV files in
`lists/`. Each row is a single trial:

```csv
letter,target
Y,false
V,false
...
M,true
```

The `target` column is `true` when the letter matches the one N trials
earlier, otherwise `false`.

### File naming

- Main lists: `N{a..j}.csv`, where `N` is the N-back level (1–5) and the
  letter (`a` … `j`) distinguishes the 10 lists per level. Examples:
  `1a.csv`, `1b.csv`, …, `5j.csv`.
- Training lists: `train_1.csv`, `train_2.csv`, `train_3.csv`.

### Generating the lists

With Nix:

```sh
nix run .#prerandomize
```

Or directly:

```sh
python3 prerandomize.py [--seed N] [--output lists] [--log LOG.md]
```

The script:

1. Builds 50 main lists (5 N-levels × 10 lists) of 36 trials each, with
   exactly 9 targets per list, using letters from the main 25-letter
   alphabet.
2. Builds 3 training lists (one each for 1-, 2-, 3-back) of 12 trials each,
   with exactly 3 targets per list, using the 8-letter training alphabet.
3. Asserts every list at the same difficulty/level has the same length, that
   the target count is exactly 25%, and that each `target` flag agrees with
   the actual letter match.
4. Compares the seed used against the most recent entry in `LOG.md` and
   appends a new line (timestamp + seed) if it has changed.

### HTML report

Every run also writes `docs/report.html` (single page, no external
dependencies). It has one tab per N-back level (1-back … 5-back); each tab
shows:

- The aggregate target / non-target split across the 10 lists in that level
  (expected to be 25 / 75 by construction).
- A bar plot of letter-frequency aggregated across the 10 lists.
- A timeline for every list (one cell per trial, letter inside, targets in
  red), so the full sequence of every individual list is visible.
- The matching training list (1, 2, or 3-back) at the bottom of its
  corresponding tab.

Open `docs/report.html` in any browser to review the lists before running
the experiment.

### Reproducibility & seed

The default seed is set at the top of `prerandomize.py` (constant
`DEFAULT_SEED`). Running the script with the same seed always produces
byte-identical CSV files. To regenerate with a new randomization, change
`DEFAULT_SEED` (or pass `--seed`) and re-run; `LOG.md` will be appended with
the new seed and the run timestamp. `LOG.md` is the audit trail: the most
recent entry identifies which seed produced the current contents of `lists/`.

## Implementation notes

The PsychoPy experiment (`nback.psyexp` / `nback_lastrun.py`) currently
generates letters at runtime with a 20% target probability. The
pre-randomization script is the new path: it produces exactly 25% targets
and lets the experimenter audit and version-control the stimulus order
before any participant runs the task.
