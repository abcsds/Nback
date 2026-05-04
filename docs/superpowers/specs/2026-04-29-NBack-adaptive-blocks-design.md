# NBack adaptive-blocks design

Design for the new `NBack.psyexp` PsychoPy experiment: a variable number of
adaptive "blocks", each sweeping the participant up the N-back ladder
(N = 1, 2, …, cap) until they fail a level. Failure shrinks the cap for all
subsequent blocks; failing N=2 ends the experiment.

## Goals

- Replace the old `nback.psyexp` runtime letter-generation path with the
  pre-randomized CSV path (`lists/{N}{letter}.csv`).
- Add a configurable number of blocks (`nBlocks`, 1..10) per session.
- Cap N adaptively: when a participant fails a list, no future block presents
  N at that level or higher.
- Keep stimulus data in CSV files; keep the runtime adaptive logic in a small,
  clearly-scoped Python layer.
- Make the per-participant schedule auditable before the session starts.

## Non-goals

- Training lists (`train_1`/`train_2`/`train_3`). One-time training before
  block 1 is in scope for the session author to wire up in Builder; the
  adaptive design does not interact with it.
- d-prime or other psychometric scoring beyond per-list miss-rate and
  false-alarm-rate thresholds.
- Online/web (PsychoJS) export. Python-only.
- Reading per-trial data back into PsychoPy from a previous session.

## Scoring rule

Each main list has 36 trials = 9 targets + 27 non-targets.

A list is **failed** if `miss_rate > 0.5 OR false_alarm_rate > 0.5`, where
`miss_rate = misses / 9` and `false_alarm_rate = false_alarms / 27`. Hits and
correct rejections are not penalised. The double threshold catches both
"never-press" and "press-everything" strategies.

## Block semantics

A "block" is a sweep through N = 1, 2, …, `cap` in ascending order, one
pre-randomized list per N. The order is fixed (ascending ladder) — predictability
is acceptable.

Block lifecycle:

1. Initialise per-list counters.
2. For each N in 1..`cap` (one list at a time):
   - Run the list's 36 trials.
   - Score the list. If failed, set `cap = N - 1` immediately and stop the rest
     of this block (subsequent N's in the same block are skipped because they
     now violate `N <= cap`).
3. End of block: if `cap < 2`, end the experiment; otherwise continue to the
   next block with the (possibly lower) cap.

The cap only ever decreases. The initial cap is the experimenter-set `topN`
(default 5). When `cap < 2`, the experiment ends — running pure 1-back has no
working-memory load and isn't useful data.

## Participant input dialog (`expInfo`)

```python
{
    'participant': 'f"{randint(0, 999):03.0f}"',  # 3-digit, indexes schedules/
    'session': '001',
    'nBlocks': '4',                                # 1..10 (Builder stores all dialog values as strings)
    'topN': '5',                                   # 2..5
}
```

Builder writes whatever the dialog returns into `expInfo`, and number-typed
fields come back as strings (`'4'`, `'5'`) regardless of how they're shown in
the dialog — hence the `int(expInfo[...])` casts in `code_init` below.

`participant` is the index into `schedules/`. `nBlocks` is the number of sweeps
attempted (the experiment may end earlier if the cap drops below 2). `topN` is
the initial cap; lower it to skip levels you expect a participant to fail.

## Pre-randomized schedules

`prerandomize.py` is extended to write **1000 per-participant schedule files**
(one per possible 3-digit ID) in a new `schedules/` directory. Each schedule
file is a CSV with columns:

```csv
block,N,list_letter,condsFile
1,1,c,lists/1c.csv
1,2,a,lists/2a.csv
1,3,j,lists/3j.csv
1,4,b,lists/4b.csv
1,5,e,lists/5e.csv
2,1,h,lists/1h.csv
…
10,5,d,lists/5d.csv
```

50 rows per file = `MAX_BLOCKS (=10) × len(N_LEVELS) (=5)` — the worst-case
schedule (no failures). The experiment loads
`schedules/{participant:03d}.csv` and **skips rows at runtime** when the cap
has dropped below `N` or when `block > nBlocks`.

For each participant and each N independently, the 10 list-letters `a..j` are
sampled as a single permutation; entry `i` of that permutation is used in
block `i + 1`. This guarantees no list-letter is reused for the same N within
a single participant's session, even in the worst case where the participant
plays all 10 blocks at full cap.

### Determinism

- The 50 main + 3 training lists consume the first 53 sub-seeds from
  `master_rng`; that ordering is unchanged, so existing list CSVs remain
  byte-identical for the same `DEFAULT_SEED`.
- Schedule generation consumes the next 1000 sub-seeds, one per participant.
- Same `DEFAULT_SEED` → byte-identical schedule files. `LOG.md` continues to
  audit seed changes.
- Repo footprint: 1000 schedule files × ~1 KB ≈ 1 MB. Tracked in git, not
  gitignored — they are part of the audit trail.

## Builder structure

```
welcome
intro
intro_ex1
trials_train1                        ← user-handled training (existing)
[user adds intro_ex2 → trials_train2 → intro_ex3 → trials_train3]

[scheduleLoop, conditionsFile=$schedule_file, nReps=1]   ← outer loop, 50 rows
   displayn       ← skipIf: $N > cap or block > n_blocks
   fix            ← skipIf: $N > cap or block > n_blocks
   [trials, conditionsFile=$condsFile, nReps=$1 if (N <= cap and block <= n_blocks) else 0]
      trial       ← End Routine: accumulate hits/misses/FAs/CRs
   [end trials]
   scoreList      ← Begin Routine: compute, drop cap on fail, end loop if cap<2
[end scheduleLoop]

end
```

Two loops total (`scheduleLoop` and the existing `trials`); one new code-only
routine (`scoreList`).

### Loop and routine settings

| Element | Setting |
|---|---|
| `scheduleLoop` (NEW) | `conditionsFile = $schedule_file`, `nReps = 1`, `loopType = sequential` |
| `displayn` | `skipIf = $N > cap or block > n_blocks`; text updated (see below) |
| `fix` | `skipIf = $N > cap or block > n_blocks` |
| `trials` (modified) | `conditionsFile = $condsFile`, `nReps = $1 if (N <= cap and block <= n_blocks) else 0`, `loopType = sequential` |
| `scoreList` (NEW, code-only) | `skipIf = $N > cap or block > n_blocks` |

### `displayn` text update

The current `text_displayn.text` references `{blocks.thisN+1}` — leftover
from the old `nback.psyexp` which had an outer loop literally named `blocks`.
The current `NBack.psyexp` has no such loop, so this f-string would raise
`NameError` if the routine were ever reached. The rewrite replaces it with
`$block` (from the schedule CSV row) and `n_blocks` (from init code):

```
$f"""Éste es el comienzo del bloque {block} de {n_blocks}\n\nPresiona la barra espaciadora si la letra que aparece es igual a la letra que aparece {N} letras antes.\n\nN = {N}\n\n[Presiona la barra espaciadora para continuar...]\n"""
```

(Set every repeat.)

## Variable lifecycle

| Variable | Set by | Lifetime | Purpose |
|---|---|---|---|
| `n_blocks`, `top_n` | welcome init | experiment | parsed from `expInfo` |
| `cap` | welcome init; mutated in `scoreList` | experiment | current N ceiling; only decreases |
| `schedule_file` | welcome init | experiment | path consumed by `scheduleLoop.conditionsFile` |
| `n_hits`, `n_misses`, `n_fa`, `n_cr` | welcome init; accumulated in `trial`; reset in `scoreList` | per-list | response tally |
| `block`, `N`, `list_letter`, `condsFile` | schedule CSV row | per-list | block index, N-level, list-letter, inner conditions file |
| `letter`, `target` | inner trials CSV row | per-trial | stimulus + target flag |

## Code components

Three new Python code components, each placed below the existing code
component(s) in its routine.

### 1. `welcome` → new code component `code_init` → Begin Experiment

```python
from pathlib import Path

n_blocks = int(expInfo['nBlocks'])
top_n = int(expInfo['topN'])
participant_id = int(expInfo['participant'])
assert 1 <= n_blocks <= 10, "nBlocks must be between 1 and 10"
assert 2 <= top_n <= 5, "topN must be between 2 and 5"
assert 0 <= participant_id <= 999, "participant must be a 3-digit code 000..999"

cap = top_n
schedule_file = f"schedules/{participant_id:03d}.csv"
assert Path(schedule_file).exists(), (
    f"missing {schedule_file}; run prerandomize.py first"
)

n_hits = n_misses = n_fa = n_cr = 0
```

`cap` is bound at module scope (Builder emits routine code inline in `run()`),
so the later `cap = N - 1` assignment in `scoreList` updates the same name
that `nReps`/`skipIf` re-evaluate on each row of the outer loop.

### 2. `trial` → new code component `code_accumulate` → End Routine

```python
is_target = str(target).strip().lower() == 'true'
pressed = bool(key_resp_trial.keys)
if is_target:
    if pressed:
        n_hits += 1
    else:
        n_misses += 1
else:
    if pressed:
        n_fa += 1
    else:
        n_cr += 1
thisExp.addData('correct', int(pressed == is_target))
thisExp.addData('block', block)
thisExp.addData('N', N)
```

No skip-guard needed — when a row is skipped, the inner `trials` loop's
`nReps = 0` ensures this routine never runs.

### 3. `scoreList` → new code component `code_score` → Begin Routine

```python
miss_rate = n_misses / 9
fa_rate = n_fa / 27
failed = miss_rate > 0.5 or fa_rate > 0.5

thisExp.addData('list_block', block)
thisExp.addData('list_N', N)
thisExp.addData('list_letter', list_letter)
thisExp.addData('list_n_hits', n_hits)
thisExp.addData('list_n_misses', n_misses)
thisExp.addData('list_n_fa', n_fa)
thisExp.addData('list_n_cr', n_cr)
thisExp.addData('list_miss_rate', miss_rate)
thisExp.addData('list_fa_rate', fa_rate)
thisExp.addData('list_failed', failed)

if failed:
    cap = N - 1
    if cap < 2:
        scheduleLoop.finished = True

n_hits = n_misses = n_fa = n_cr = 0
```

The reset at the end primes the counters for the next list.

`scheduleLoop.finished = True` is the **experiment-end** signal — it only
fires when `cap < 2`. Block boundaries do not need explicit handling here:
when a list is failed mid-block, `cap = N - 1` immediately invalidates
all higher-N rows, and the per-row `skipIf = $N > cap or block > n_blocks`
silently passes them through. Same mechanism handles the
`block > n_blocks` rows at the tail of the schedule.

## `prerandomize.py` extension

### Constants (top of file, near `DEFAULT_SEED`)

```python
N_PARTICIPANTS = 1000        # 3-digit codes 000..999
# MAX_BLOCKS == LISTS_PER_LEVEL is load-bearing: each block consumes one
# list-letter per N, and the per-N permutation is exactly LISTS_PER_LEVEL
# letters long, so this guarantees no list-letter is reused for the same N
# within a participant even in the no-failure worst case.
MAX_BLOCKS = LISTS_PER_LEVEL  # 10
```

### New helper functions

```python
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
```

### Integration into `main()`

After the existing main-list and training-list loops, before
`render_html_report`:

```python
schedules_dir = args.output / "schedules"
schedules_dir.mkdir(exist_ok=True)
for pid in range(N_PARTICIPANTS):
    sub_seed = master_rng.getrandbits(64)
    sub_rng = random.Random(sub_seed)
    rows = generate_schedule(sub_rng, N_LEVELS, MAX_BLOCKS, LISTS_PER_LEVEL)
    verify_schedule(rows, N_LEVELS, MAX_BLOCKS, LISTS_PER_LEVEL)
    write_schedule(schedules_dir / f"{pid:03d}.csv", rows)
```

And add to the final summary print:

```python
print(f"Wrote {N_PARTICIPANTS} participant schedules to {schedules_dir}/")
```

## HTML report addition

Add one paragraph above the existing "Experiment schedule" section, e.g.:

> **Participant schedules.** `prerandomize.py` also writes 1000 per-participant
> schedules (`schedules/000.csv` … `schedules/999.csv`), one for each possible
> 3-digit participant code. Each schedule has 50 rows = 10 blocks × 5
> N-levels, with the 10 list-letters per N-level used in a permuted order. The
> experiment loads `schedules/{participant}.csv` at runtime and skips rows
> where the cap has dropped below the row's N or where the row's block
> exceeds the experimenter-set `nBlocks`.

No new tab is needed — individual schedules can be audited by opening the
CSVs directly.

## README updates

- Add a **Participant schedules** subsection under "Pre-randomization"
  describing `schedules/`, the 1000-participant convention, the per-N
  permutation, and audit guarantees.
- Add `nBlocks`, `topN`, and the `schedules/{participant}.csv` lookup to the
  "Implementation notes" section.
- Update the "Pre-randomization" section's command examples to mention that
  `prerandomize.py` now also produces schedules.

## Worked session example

`nBlocks = 4`, `topN = 5`, participant `042`. Schedule
`schedules/042.csv` has 50 rows.

1. Init: `cap = 5`, `n_blocks = 4`, `schedule_file = "schedules/042.csv"`.
2. Row 1 (`block=1, N=1`): runs. Suppose the participant passes.
3. Rows 2–4 (`N = 2, 3, 4`): all run, all pass.
4. Row 5 (`block=1, N=5`): runs. `miss_rate = 0.67`. `scoreList` sets
   `cap = 4`.
5. Rows 6–9 (`block=2, N=1..4`): run, all pass. Row 10 (`block=2, N=5`) is
   skipped (`5 > 4`).
6. Rows 11–14 (`block=3, N=1..4`): run; the participant fails `N=4`.
   `cap = 3`. Row 15 (`block=3, N=5`) was already going to be skipped;
   nothing else changes.
7. Rows 16–18 (`block=4, N=1..3`): run, all pass. Rows 19–20
   (`block=4, N=4..5`): skipped.
8. Rows 21..50 (`block > 4`): all skipped because `block > n_blocks`. The
   `scheduleLoop` exits.
9. `end` routine plays.

If at step 6 the participant had failed `N = 2`, `cap` would drop to 1, and
`scheduleLoop.finished = True` would short-circuit the experiment.

## Data file columns

The PsychoPy wide CSV will accumulate, in addition to the standard PsychoPy
columns, the following experiment-specific columns:

- Per-trial rows (from inside `trials` loop): `letter`, `target`,
  `key_resp_trial.keys`, `key_resp_trial.rt`, `correct`, `block`, `N`.
- Per-list rows (from `scoreList`): `list_block`, `list_N`, `list_letter`,
  `list_n_hits`, `list_n_misses`, `list_n_fa`, `list_n_cr`,
  `list_miss_rate`, `list_fa_rate`, `list_failed`.
- Outer-loop rows (from the `scheduleLoop` row context): `block`, `N`,
  `list_letter`, `condsFile`.

`block` and `N` appear in both the per-trial accumulator and the outer-loop
row context. PsychoPy's wide CSV merges these into single columns rather than
duplicating them — the `thisExp.addData('block', block)` call in the
accumulator overwrites the auto-emitted outer-loop value within the same
row. Distinct values are kept under the `list_*` prefix in `scoreList`.

## Edge cases

- **Participant ID outside 0..999 or schedule file missing.** `code_init`
  asserts both `0 <= participant_id <= 999` and `Path(schedule_file).exists()`
  before any loop is reached, so the experiment aborts with a clear message
  rather than crashing inside `data.importConditions`. This catches both a
  miss-typed 4-digit ID and a session run before `prerandomize.py` has been
  executed.
- **`nBlocks > 10` or `topN > 5`.** The init asserts both bounds and aborts
  before the first block.
- **Failure on the first list of the experiment (block 1, N=1).** `cap`
  drops to 0, `scheduleLoop.finished = True`, and the experiment ends after
  one list.
- **No failures across all `nBlocks` blocks.** `cap` stays at `topN`; every
  block runs `topN` lists; total = `nBlocks × topN` lists.
- **Pool exhaustion.** With `MAX_BLOCKS = LISTS_PER_LEVEL = 10`, the
  worst-case 10-block-no-failure session uses exactly 10 lists per N,
  matching the schedule's per-N permutation length. By construction, no list
  is reused within a participant.
