# N-back task

N-back task for cognitive mental workload implemented in PsychoPy. Written in
Spanish using letters of the English alphabet.

## Parameters

| Parameter | Value |
| --- | --- |
| Letter alphabet (main) | ASCII uppercase minus `N` (25 letters; `N` was removed because testers confused it with the N-back instruction) |
| Letter alphabet (training) | `A B C D E F G H` (8-letter subset, easier for first exposure) |
| Letter presentation duration | **1.2 s** |
| Inter-trial interval (ITI) | **0.6 s** (fix routine: 0.3 s cross + 0.3 s blank) |
| Trial duration | 1.8 s (presentation + ITI) |
| Main block duration | **64.8 s** → 36 trials per list |
| Training block duration | **21.6 s** → 12 trials per list |
| Target probability | **25%** (exact, not probabilistic) — 9 targets per main list, 3 per training list |
| N-back levels (main) | 1, 2, 3, 4, 5 |
| Lists per level (main) | 10 |
| N-back levels (training) | 1, 2, 3 (one list each) |

A *target* is a trial whose letter matches the letter shown N positions
earlier. Non-target trials are guaranteed to differ from the letter N back, so
they cannot accidentally form a target. Targets are also constrained to
**never be adjacent** to another target — i.e. trial *i* and *i + 1* are
never both targets — to avoid attentional-blink confounds.

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

### Participant schedules

`prerandomize.py` also emits 1000 per-participant schedule files in
`schedules/`, named `000.csv` … `999.csv` — one per possible 3-digit
participant code. Each schedule has 50 rows (10 blocks × 5 N-levels);
for each N, the 10 list-letters `a..j` are permuted once and assigned
across the 10 blocks (without replacement). Columns:

```csv
block,N,list_letter,condsFile
1,1,c,lists/1c.csv
1,2,a,lists/2a.csv
...
10,5,d,lists/5d.csv
```

The experiment loads `schedules/{participant}.csv` at runtime and skips
rows where the cap has dropped below the row's N or where the row's
block exceeds the experimenter-set `nBlocks`. With the same `DEFAULT_SEED`,
all 1000 schedule files are byte-identical.

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

- A schedule table at the top showing experiment duration when using *K*
  lists per N-back level (K = 1…10). Because a participant who fails
  &gt;50% of a level does not advance to the next, each row breaks the
  duration into a fixed *Instructions* column (5 min) plus 5 task-time
  columns (one per highest N-back level reached, 1…5); the 5-back column
  is the upper bound while the 4-back column reflects the typical realistic
  case. Task time covers the matching training list before each of 1-, 2-,
  3-back (4- and 5-back have no training), the *K* main lists, and a 1-min
  break between every consecutive list (the K=1 / 1-back cell skips that
  break as the minimum experiment unit). Add the *Instructions* column to
  a task-time cell for the wall-clock total.
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

## Per-participant reports

After every experiment run, build a per-participant HTML report from the
PsychoPy CSV in `data/`:

```sh
nix run .#report
```

This scans every `data/*.csv`, writes `docs/reports/<csv-stem>.html` for
each run that reached the main `scheduleLoop`, and rebuilds
`docs/index.html` so it lists every available report newest-first.
Older-schema CSVs (from before the pre-randomized lists existed) are
skipped silently.

Each participant report contains:

- Run metadata (participant, session, date, duration, max-N reached) and
  a per-N pass / fail / not-reached strip across levels 1..5.
- An aggregate 2×2 confusion matrix (target × press) plus a per-N table
  with hit rate, false-alarm rate, mean and median RT.
- An interactive Bokeh figure: histogram + Gaussian KDE of every
  key-press response time, coloured per N-back level (legend toggles
  hide/show).
- One section per actually-run scheduleLoop block: a small confusion
  matrix and an interactive Bokeh timeline (RT vs. trial index, glyphs
  encode hit / miss / false alarm / correct rejection, hover surfaces
  letter + outcome).

Reports are self-contained (Bokeh JS/CSS inlined) so they open offline.

## Implementation notes

The PsychoPy experiment (`NBack.psyexp`) consumes the pre-randomized
lists. Two entries in the participant-info dialog control the run:

| Field | Range | Purpose |
| --- | --- | --- |
| `nBlocks` | 1..10 | Number of adaptive sweeps (blocks) attempted |
| `topN` | 2..5 | Initial N ceiling (`cap`); only ever decreases |

At runtime, `code_welcome` builds the path
`schedules/{participant:03d}.csv` and the outer `scheduleLoop` reads the
50 rows. The inner `trials` loop's `conditionsFile` is set per-row via
`$condsFile` (one of `lists/{N}{letter}.csv`). After each list the
`scoreList` routine computes per-list miss-rate and false-alarm-rate; if
either exceeds 0.5, the cap drops to `N - 1` and all subsequent rows
where `N > cap` are skipped. When the cap drops below 2,
`scheduleLoop.finished = True` ends the experiment.
