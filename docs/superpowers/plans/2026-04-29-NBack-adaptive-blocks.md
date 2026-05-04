# NBack adaptive-blocks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the adaptive-blocks redesign described in [`docs/superpowers/specs/2026-04-29-NBack-adaptive-blocks-design.md`](../specs/2026-04-29-NBack-adaptive-blocks-design.md): extend `prerandomize.py` to emit 1000 per-participant `schedules/{NNN}.csv` files, and wire `NBack.psyexp` to consume them with cap-driven row-skipping.

**Architecture:** The plan has three tracks:
- **Track A (Python, agent-doable, TDD):** extend `prerandomize.py` with `generate_schedule` / `verify_schedule` / `write_schedule` and integrate into `main()`. Add a `tests/test_prerandomize.py` runnable via stdlib `unittest`.
- **Track B (PsychoPy Builder, you do this in the GUI):** add the `scheduleLoop` and `scoreList` routine, configure `skipIf`/`nReps` expressions, and paste the three code components from the spec. This is a manual checklist — Builder is a GUI tool, not a text format we can safely diff-edit.
- **Track C (docs, agent-doable):** update `README.md` and the HTML-report description string in `prerandomize.py`.

**Tech Stack:** Python 3 stdlib only (`csv`, `random`, `pathlib`, `unittest`). PsychoPy 2026.1.3 (Builder GUI) for the experiment file. Nix flake unchanged.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `prerandomize.py` | modify | add schedule constants + 3 helper functions + main() integration |
| `tests/test_prerandomize.py` | create | unittest cases for `generate_schedule`, `verify_schedule`, `write_schedule`; existing-CSV reproducibility check |
| `tests/__init__.py` | create | empty package marker (lets `python3 -m unittest discover tests/` find tests) |
| `schedules/000.csv` … `schedules/999.csv` | create (by run) | 1000 per-participant schedule CSVs, byte-identical given `DEFAULT_SEED` |
| `NBack.psyexp` | modify (in PsychoPy Builder) | add scheduleLoop + scoreList routine + 3 code components per spec |
| `README.md` | modify | add Participant schedules subsection + nBlocks/topN to Implementation notes |

The new test file is the only architectural addition — keeps `prerandomize.py` self-contained and matches the existing "stdlib-only" style.

---

## Track A — `prerandomize.py` extensions (Python, TDD)

### Task 1: Set up test scaffolding and the `generate_schedule` failing test

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_prerandomize.py`

- [ ] **Step 1: Create empty `tests/__init__.py`**

```bash
mkdir -p tests
: > tests/__init__.py
```

- [ ] **Step 2: Write `tests/test_prerandomize.py` with the first failing test for `generate_schedule`**

```python
"""Tests for prerandomize.py — schedule generation and existing-list reproducibility."""
from __future__ import annotations

import hashlib
import random
import sys
import unittest
from pathlib import Path

# Make the repo root importable so we can `import prerandomize` from tests/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import prerandomize  # noqa: E402


class TestGenerateSchedule(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(0)
        self.n_levels = prerandomize.N_LEVELS
        self.max_blocks = prerandomize.LISTS_PER_LEVEL
        self.lists_per_level = prerandomize.LISTS_PER_LEVEL

    def test_row_count(self) -> None:
        rows = prerandomize.generate_schedule(
            self.rng, self.n_levels, self.max_blocks, self.lists_per_level
        )
        self.assertEqual(len(rows), self.max_blocks * len(self.n_levels))

    def test_block_major_ordering(self) -> None:
        rows = prerandomize.generate_schedule(
            self.rng, self.n_levels, self.max_blocks, self.lists_per_level
        )
        self.assertEqual(rows, sorted(rows, key=lambda r: (r[0], r[1])))

    def test_each_letter_used_once_per_n(self) -> None:
        rows = prerandomize.generate_schedule(
            self.rng, self.n_levels, self.max_blocks, self.lists_per_level
        )
        letters_by_n: dict[int, list[str]] = {n: [] for n in self.n_levels}
        for _block, n, letter, _conds in rows:
            letters_by_n[n].append(letter)
        expected = sorted(chr(ord("a") + i) for i in range(self.lists_per_level))
        for n in self.n_levels:
            self.assertEqual(sorted(letters_by_n[n]), expected, f"N={n}")

    def test_condsfile_path_format(self) -> None:
        rows = prerandomize.generate_schedule(
            self.rng, self.n_levels, self.max_blocks, self.lists_per_level
        )
        for _block, n, letter, conds in rows:
            self.assertEqual(conds, f"lists/{n}{letter}.csv")

    def test_deterministic_for_same_seed(self) -> None:
        a = prerandomize.generate_schedule(
            random.Random(42), self.n_levels, self.max_blocks, self.lists_per_level
        )
        b = prerandomize.generate_schedule(
            random.Random(42), self.n_levels, self.max_blocks, self.lists_per_level
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -v`
Expected: All four `TestGenerateSchedule.*` tests fail with `AttributeError: module 'prerandomize' has no attribute 'generate_schedule'`.

- [ ] **Step 4: Commit the failing tests**

```bash
git add tests/__init__.py tests/test_prerandomize.py
git commit -m "test: add failing tests for generate_schedule"
```

---

### Task 2: Implement `generate_schedule` and the supporting constant

**Files:**
- Modify: `prerandomize.py` (add constant near `DEFAULT_SEED`; add function near other helpers)

- [ ] **Step 1: Add the `MAX_BLOCKS` constant near `DEFAULT_SEED`**

In `prerandomize.py`, after the line `DEFAULT_SEED = 20260427`:

```python
N_PARTICIPANTS = 1000        # 3-digit codes 000..999
# MAX_BLOCKS == LISTS_PER_LEVEL is load-bearing: each block consumes one
# list-letter per N, and the per-N permutation is exactly LISTS_PER_LEVEL
# letters long, so this guarantees no list-letter is reused for the same N
# within a participant even in the no-failure worst case.
MAX_BLOCKS = LISTS_PER_LEVEL  # 10
```

- [ ] **Step 2: Add `generate_schedule` after the existing `write_csv` helper**

In `prerandomize.py`, add after `write_csv`:

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
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -v`
Expected: All four `TestGenerateSchedule.*` tests pass.

- [ ] **Step 4: Commit**

```bash
git add prerandomize.py
git commit -m "feat: add generate_schedule and MAX_BLOCKS constant"
```

---

### Task 3: Add `verify_schedule` tests (positive and negative cases)

**Files:**
- Modify: `tests/test_prerandomize.py`

- [ ] **Step 1: Append `TestVerifySchedule` to `tests/test_prerandomize.py`**

After the `TestGenerateSchedule` class (before `if __name__ == "__main__":`), add:

```python
class TestVerifySchedule(unittest.TestCase):
    def setUp(self) -> None:
        self.n_levels = prerandomize.N_LEVELS
        self.max_blocks = prerandomize.LISTS_PER_LEVEL
        self.lists_per_level = prerandomize.LISTS_PER_LEVEL
        self.good_rows = prerandomize.generate_schedule(
            random.Random(7),
            self.n_levels,
            self.max_blocks,
            self.lists_per_level,
        )

    def _verify(self, rows):
        prerandomize.verify_schedule(
            rows, self.n_levels, self.max_blocks, self.lists_per_level
        )

    def test_accepts_well_formed_schedule(self) -> None:
        self._verify(self.good_rows)  # must not raise

    def test_rejects_wrong_row_count(self) -> None:
        with self.assertRaises(AssertionError):
            self._verify(self.good_rows[:-1])

    def test_rejects_out_of_range_block(self) -> None:
        bad = list(self.good_rows)
        bad[0] = (0, 1, "a", "lists/1a.csv")
        with self.assertRaises(AssertionError):
            self._verify(bad)

    def test_rejects_unknown_n(self) -> None:
        bad = list(self.good_rows)
        bad[0] = (1, 99, "a", "lists/99a.csv")
        with self.assertRaises(AssertionError):
            self._verify(bad)

    def test_rejects_malformed_condsfile(self) -> None:
        block, n, letter, _conds = self.good_rows[0]
        bad = list(self.good_rows)
        bad[0] = (block, n, letter, f"wrong/{n}{letter}.csv")
        with self.assertRaises(AssertionError):
            self._verify(bad)

    def test_rejects_reused_letter_within_same_n(self) -> None:
        # Find two rows for N=1 and force them to share a letter.
        bad = list(self.good_rows)
        idx_first = next(i for i, r in enumerate(bad) if r[1] == 1)
        idx_second = next(
            i for i, r in enumerate(bad) if r[1] == 1 and i != idx_first
        )
        b, n, _letter, _conds = bad[idx_second]
        shared = bad[idx_first][2]
        bad[idx_second] = (b, n, shared, f"lists/{n}{shared}.csv")
        with self.assertRaises(AssertionError):
            self._verify(bad)

    def test_rejects_wrong_row_ordering(self) -> None:
        # Swap the first and last rows — block-major ordering is broken.
        bad = list(self.good_rows)
        bad[0], bad[-1] = bad[-1], bad[0]
        with self.assertRaises(AssertionError):
            self._verify(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest discover -s tests -v`
Expected: All seven `TestVerifySchedule.*` tests fail with `AttributeError: module 'prerandomize' has no attribute 'verify_schedule'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_prerandomize.py
git commit -m "test: add failing tests for verify_schedule"
```

---

### Task 4: Implement `verify_schedule` and `write_schedule`

**Files:**
- Modify: `prerandomize.py`

- [ ] **Step 1: Add `verify_schedule` and `write_schedule` after `generate_schedule`**

In `prerandomize.py`, immediately after `generate_schedule`:

```python
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

- [ ] **Step 2: Run tests to verify they pass**

Run: `python3 -m unittest discover -s tests -v`
Expected: All `TestGenerateSchedule.*` and `TestVerifySchedule.*` tests pass (11 tests).

- [ ] **Step 3: Commit**

```bash
git add prerandomize.py
git commit -m "feat: add verify_schedule and write_schedule"
```

---

### Task 5: Snapshot existing list CSVs for the reproducibility check

The next task wires the schedule generation into `main()`. Before that, capture a hash of the current `lists/` contents so we can verify the new code doesn't perturb existing list output.

**Files:**
- Modify: `tests/test_prerandomize.py`

- [ ] **Step 1: Add `TestExistingListsUnchanged` to `tests/test_prerandomize.py`**

Append before `if __name__ == "__main__":`:

```python
class TestExistingListsUnchanged(unittest.TestCase):
    """Re-running prerandomize.py with DEFAULT_SEED must not change lists/*.csv.

    Existing list files are checked into the repo; this asserts that adding
    schedule generation doesn't shift the master_rng's first 53 sub-seeds.
    """

    def test_lists_dir_hash_stable(self) -> None:
        lists_dir = ROOT / "lists"
        # Sort filenames so the hash is filesystem-order-independent.
        digest = hashlib.sha256()
        for path in sorted(lists_dir.glob("*.csv")):
            digest.update(path.name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        observed = digest.hexdigest()

        snapshot_file = ROOT / "tests" / "lists_sha256.txt"
        if not snapshot_file.exists():
            snapshot_file.write_text(observed + "\n")
            self.skipTest(
                f"wrote initial snapshot to {snapshot_file}; "
                "re-run after main() integration to assert"
            )
        expected = snapshot_file.read_text().strip()
        self.assertEqual(observed, expected, (
            "lists/*.csv hash drifted; "
            "schedule generation perturbed master_rng consumption order"
        ))
```

- [ ] **Step 2: Run tests once to write the snapshot file**

Run: `python3 -m unittest discover -s tests -v`
Expected: 11 prior tests pass; `test_lists_dir_hash_stable` is reported as `skipped` and `tests/lists_sha256.txt` is now present with the current hash.

- [ ] **Step 3: Verify the snapshot file was written**

Run: `cat tests/lists_sha256.txt`
Expected: a single 64-char hex string followed by a newline.

- [ ] **Step 4: Commit the snapshot and the new test**

```bash
git add tests/test_prerandomize.py tests/lists_sha256.txt
git commit -m "test: snapshot lists/ hash for reproducibility check"
```

---

### Task 6: Integrate schedule generation into `main()`

**Files:**
- Modify: `prerandomize.py` (add CLI arg, schedule-generation block in `main()`, summary print)

The PsychoPy experiment will look for `schedules/{participant:03d}.csv` *relative to the experiment file's directory* (the repo root). The schedules therefore live at the repo root, not under `lists/`. We add a dedicated `--schedules` CLI arg (mirroring the existing `--output` and `--report` args) defaulting to `schedules/`.

- [ ] **Step 1: Add the `--schedules` CLI argument**

In `prerandomize.py`, in `main()`, find the existing argparse block (the four `parser.add_argument(...)` calls). After the `--report` argument, add:

```python
    parser.add_argument("--schedules", type=Path, default=Path("schedules"),
                        help="output directory for participant schedules (default: schedules)")
```

- [ ] **Step 2: Make the schedules directory at the top of `main()`**

Find the existing line `args.report.parent.mkdir(parents=True, exist_ok=True)` near the top of `main()`. Immediately after it, add:

```python
    args.schedules.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: Insert the schedule-generation block in `main()`**

In `prerandomize.py`, find the block beginning with `# Training lists: one per N for 1, 2, 3-back` and ending with the `assert len(set(train_lengths_seen)) == 1, ...` assertion. **Immediately after that assertion**, before the call to `render_html_report(...)`, insert:

```python
    # Per-participant schedules (3-digit IDs 000..999)
    for pid in range(N_PARTICIPANTS):
        sub_seed = master_rng.getrandbits(64)
        sub_rng = random.Random(sub_seed)
        rows = generate_schedule(sub_rng, N_LEVELS, MAX_BLOCKS, LISTS_PER_LEVEL)
        verify_schedule(rows, N_LEVELS, MAX_BLOCKS, LISTS_PER_LEVEL)
        write_schedule(args.schedules / f"{pid:03d}.csv", rows)
```

- [ ] **Step 4: Add a summary print near the existing prints at the bottom of `main()`**

After the existing `print(f"Wrote report to {args.report}")` line, add:

```python
    print(f"Wrote {N_PARTICIPANTS} participant schedules to {args.schedules}/")
```

- [ ] **Step 5: Run prerandomize.py end-to-end**

Run: `python3 prerandomize.py`
Expected output ends with:

```
Wrote 50 main lists (36 trials, 9 targets each) and 3 training lists (12 trials, 3 targets each) to lists/
Wrote report to docs/report.html
Wrote 1000 participant schedules to schedules/
seed unchanged (20260427); LOG.md not updated
```

(Or `seed changed (...) -> 20260427; appended to LOG.md` if your local LOG.md doesn't already end with that seed.)

- [ ] **Step 6: Spot-check the output**

Run: `ls schedules | head -5 && head -6 schedules/000.csv`
Expected: filenames `000.csv 001.csv 002.csv 003.csv 004.csv`; the first file has the header `block,N,list_letter,condsFile` and rows starting at `1,1,...`, `1,2,...`, etc.

- [ ] **Step 7: Run the full test suite and confirm reproducibility**

Run: `python3 -m unittest discover -s tests -v`
Expected: all 12 tests pass (the previously-skipped `test_lists_dir_hash_stable` now runs and asserts the existing lists are byte-identical).

- [ ] **Step 8: Commit**

```bash
git add prerandomize.py schedules/
git commit -m "feat: emit 1000 per-participant schedules in prerandomize.py"
```

---

## Track B — PsychoPy Builder edits (manual, in the GUI)

These tasks are checklists you (the experimenter) follow inside the PsychoPy Builder. The goal is to wire `NBack.psyexp` to consume `schedules/{participant}.csv` and the existing `lists/{N}{letter}.csv` files, per the design spec. There is no automated test for these — verification is by smoke-running the experiment.

### Task 7: Add `nBlocks` and `topN` to `expInfo` and the `code_init` component

**Files:**
- Modify (in Builder): `NBack.psyexp` — Settings dialog and `welcome` routine

- [ ] **Step 1: Open `NBack.psyexp` in PsychoPy Builder**

- [ ] **Step 2: Edit Experiment Info (Settings → Experiment info dialog)**

Add two entries to the dictionary so it reads:

```python
{'participant': 'f"{randint(0, 999):03.0f}"', 'session': '001', 'nBlocks': '4', 'topN': '5'}
```

(If the participant entry currently still has `randint(0, 999999):06.0f`, change it to the 3-digit form above.)

- [ ] **Step 3: In the `welcome` routine, insert a new code component below `code_welcome`**

Right-click the `welcome` routine in Builder → Insert Component → Code. Name it `code_init`. Place it **below** `code_welcome` in the routine.

In the **Begin Experiment** tab, paste:

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

Leave all other tabs (Begin Routine, Each Frame, …) empty.

- [ ] **Step 4: Save the experiment**

Save via Ctrl+S. Builder will regenerate `NBack_lastrun.py`.

- [ ] **Step 5: Smoke-test the dialog**

Run the experiment (Run button). The participant-info dialog should now include `nBlocks` and `topN` fields. Cancel out of the dialog (or hit OK and let it crash later — the rest of the wiring isn't done yet; the goal of this step is to confirm the dialog renders the new fields). Then close PsychoPy.

- [ ] **Step 6: Commit**

```bash
git add NBack.psyexp NBack_lastrun.py
git commit -m "feat(NBack): add nBlocks/topN expInfo and code_init"
```

---

### Task 8: Restructure the loops — outer `scheduleLoop` + modify inner `trials`

**Files:**
- Modify (in Builder): `NBack.psyexp` — Flow panel

- [ ] **Step 1: Open `NBack.psyexp` and switch to the Flow panel**

The current flow has a `trials` loop wrapping `displayn` → `fix` → `trial`. The first thing to do is **remove the inner contents from the existing `trials` loop and replace its conditions file**, then add a new outer `scheduleLoop` around the structure.

- [ ] **Step 2: Re-target the existing `trials` loop**

Click the `trials` loop initiator. In the loop dialog:
- Set `conditionsFile` to: `$condsFile`
- Set `nReps` to: `$1 if (N <= cap and block <= n_blocks) else 0`
- Set `loopType` to: `sequential`
- Move the loop's start/end so it wraps **only the `trial` routine** (not `displayn` or `fix`). Currently `trials` wraps `displayn → fix → trial`; you want it to wrap just `trial`.

- [ ] **Step 3: Add the new outer `scheduleLoop`**

In the Flow panel:
- Click "Insert Loop"
- Name: `scheduleLoop`
- `conditionsFile`: `$schedule_file`
- `nReps`: `1`
- `loopType`: `sequential`
- random seed: leave empty
- Position the loop so it wraps `displayn → fix → trials → scoreList` (you'll add `scoreList` in the next task; for now wrap `displayn → fix → trials`).

- [ ] **Step 4: Save**

Save via Ctrl+S.

- [ ] **Step 5: Spot-check the saved XML**

Run: `grep -n "loopType\|name=\"scheduleLoop\"\|name=\"trials\"\|conditionsFile" NBack.psyexp`
Expected: a `LoopInitiator name="scheduleLoop"` with `conditionsFile val="$schedule_file"`, and the `trials` loop showing `conditionsFile val="$condsFile"` and `nReps val="$1 if (N <= cap and block <= n_blocks) else 0"`.

- [ ] **Step 6: Commit**

```bash
git add NBack.psyexp NBack_lastrun.py
git commit -m "feat(NBack): wrap trials in scheduleLoop and switch to dynamic condsFile"
```

---

### Task 9: Add `scoreList` routine and the two new code components

**Files:**
- Modify (in Builder): `NBack.psyexp` — new routine, and code components on `trial` and `scoreList`

- [ ] **Step 1: Create the `scoreList` routine**

In Builder: Routines → Insert Routine → name it `scoreList`. Leave it empty for now (no visible stim).

- [ ] **Step 2: Add `code_score` to `scoreList`**

In `scoreList`, insert a Code component named `code_score`. In the **Begin Routine** tab, paste:

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

- [ ] **Step 3: Place `scoreList` in the flow**

In the Flow panel, drop `scoreList` **inside** the `scheduleLoop` and **after** the inner `trials` loop, so the order inside `scheduleLoop` is: `displayn → fix → trials → scoreList`.

- [ ] **Step 4: Add `code_accumulate` to the `trial` routine**

Open the `trial` routine. Insert a new Code component named `code_accumulate`, placed **below** any existing code component(s) in `trial`. In the **End Routine** tab, paste:

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

Leave all other tabs (Begin Experiment, Begin Routine, …) empty.

- [ ] **Step 5: Save**

Save via Ctrl+S.

- [ ] **Step 6: Commit**

```bash
git add NBack.psyexp NBack_lastrun.py
git commit -m "feat(NBack): add scoreList routine + accumulator/scorer code"
```

---

### Task 10: Configure `skipIf` on `displayn`, `fix`, and `scoreList`; update `displayn` text

**Files:**
- Modify (in Builder): `NBack.psyexp` — routine settings on `displayn`, `fix`, `scoreList`; text on `displayn.text_displayn`

- [ ] **Step 1: Set `skipIf` on `displayn`**

Open the `displayn` routine. Click on the routine settings (the `displayn` row at the top of the routine pane). In the `skipIf` field, paste:

```
$N > cap or block > n_blocks
```

- [ ] **Step 2: Set the same `skipIf` on `fix`**

Same procedure for the `fix` routine: `skipIf` = `$N > cap or block > n_blocks`.

- [ ] **Step 3: Set the same `skipIf` on `scoreList`**

Same procedure for the `scoreList` routine: `skipIf` = `$N > cap or block > n_blocks`.

- [ ] **Step 4: Update `text_displayn.text`**

In `displayn`, edit `text_displayn`. The current `text` field references `{blocks.thisN+1}` — leftover from the old experiment with a loop named `blocks`. Replace the `text` field (set every repeat) with:

```
$f"""Éste es el comienzo del bloque {block} de {n_blocks}\n\nPresiona la barra espaciadora si la letra que aparece es igual a la letra que aparece {N} letras antes.\n\nN = {N}\n\n[Presiona la barra espaciadora para continuar...]\n"""
```

- [ ] **Step 5: Save**

Save via Ctrl+S.

- [ ] **Step 6: Commit**

```bash
git add NBack.psyexp NBack_lastrun.py
git commit -m "feat(NBack): wire skipIf and rewrite displayn text"
```

---

### Task 11: Smoke-test the full experiment end-to-end

**Files:**
- No file changes — this is a runtime check.

- [ ] **Step 1: Sanity check — happy path**

Run the experiment. Enter:
- `participant`: `0`
- `nBlocks`: `1`
- `topN`: `2`

This is the smallest meaningful session: 1 block × 2 N-levels = 2 lists.

Press space when targets appear. Try to be accurate — the cap should *not* drop. Expected behaviour:
- Welcome → Intro → (training routines, if you've wired any) → "Block 1 de 1, N = 1" message → fix cross → 36-trial 1-back list → silent `scoreList` (no visible stim) → "Block 1 de 1, N = 2" → fix → 36-trial 2-back list → silent `scoreList` → end screen.

- [ ] **Step 2: Sanity check — early-fail path**

Run again with the same `nBlocks=1, topN=2`, but during the N=2 list, deliberately press space on every trial (high false-alarm rate). The experiment should end immediately after that list (cap drops to 1, `scheduleLoop.finished = True` short-circuits). You should see the `end` routine, not "Block 2".

- [ ] **Step 3: Sanity check — adaptive run**

Run with `participant=42, nBlocks=4, topN=5`. Behave naturally; deliberately fail at, say, N=4 in block 2. Verify the `data/*.csv` output contains `list_failed=True` for the failed list and that subsequent rows skip N≥4.

- [ ] **Step 4: Spot-check the data file**

Run: `ls -t data/*.csv | head -1 | xargs head -3`
Expected: a header row including `block, N, letter, target, key_resp_trial.keys, key_resp_trial.rt, correct, list_failed, list_miss_rate, list_fa_rate, ...`.

- [ ] **Step 5: If everything works, commit any auto-regenerated files and the data file is gitignored already**

```bash
git status
# Confirm only NBack_lastrun.py changed (or nothing). The data/ output is gitignored.
git diff --stat
```

If `NBack_lastrun.py` regenerated, commit:

```bash
git add NBack_lastrun.py
git commit -m "chore: regenerate NBack_lastrun.py after smoke test"
```

---

## Track C — Documentation

### Task 12: Update HTML report description in `prerandomize.py`

**Files:**
- Modify: `prerandomize.py` (in `render_html_report`)

- [ ] **Step 1: Locate the report's schedule-note paragraph**

In `prerandomize.py`, find the block in `render_html_report` that emits `<p class="schedule-note">...` (around the section that ends with "Add the &ldquo;Instructions&rdquo; column to a per-level cell to get the wall-clock total."). **Immediately before** that `<p class="schedule-note">` paragraph, add a new paragraph describing the participant schedules.

Replace the line:

```python
{_schedule_table_html()}
```

with:

```python
<h3>Participant schedules</h3>
<p><code>prerandomize.py</code> also writes {N_PARTICIPANTS} per-participant
schedules (<code>schedules/000.csv</code> &hellip; <code>schedules/{N_PARTICIPANTS - 1:03d}.csv</code>),
one for each possible 3-digit participant code. Each schedule has
{MAX_BLOCKS * len(N_LEVELS)} rows = {MAX_BLOCKS} blocks &times; {len(N_LEVELS)} N-levels,
with the {LISTS_PER_LEVEL} list-letters per N-level used in a permuted order.
The experiment loads <code>schedules/{{participant}}.csv</code> at runtime and
skips rows where the cap has dropped below the row's N or where the row's
block exceeds the experimenter-set <code>nBlocks</code>.</p>
{_schedule_table_html()}
```

(Note: the `{{participant}}` is escaped braces so the f-string outputs literal `{participant}`.)

- [ ] **Step 2: Regenerate the report and visually inspect**

Run: `python3 prerandomize.py && grep -A 2 "Participant schedules" docs/report.html | head -10`
Expected: the new paragraph is present with concrete numbers (e.g., "1000 per-participant schedules", "50 rows = 10 blocks × 5 N-levels").

- [ ] **Step 3: Commit**

```bash
git add prerandomize.py docs/report.html
git commit -m "docs: describe participant schedules in HTML report"
```

---

### Task 13: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a `Participant schedules` subsection under `## Pre-randomization`**

In `README.md`, find the `### File naming` subsection. Immediately after the closing of that subsection (before `### Generating the lists`), insert:

```markdown
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
```

- [ ] **Step 2: Update the `## Implementation notes` section**

In `README.md`, find the `## Implementation notes` section (it currently describes how `nback.psyexp` generates letters at runtime). Replace the entire section with:

```markdown
## Implementation notes

The new PsychoPy experiment (`NBack.psyexp` / `NBack_lastrun.py`) consumes
the pre-randomized lists. Two new entries appear in the participant-info
dialog:

| Field | Range | Purpose |
| --- | --- | --- |
| `nBlocks` | 1..10 | Number of adaptive sweeps (blocks) attempted |
| `topN` | 2..5 | Initial N ceiling (`cap`); only ever decreases |

At runtime, `code_init` builds the path `schedules/{participant:03d}.csv`
and the outer `scheduleLoop` reads the 50 rows. The inner `trials` loop's
`conditionsFile` is set per-row via `$condsFile` (one of
`lists/{N}{letter}.csv`). After each list the `scoreList` routine
computes per-list miss-rate and false-alarm-rate; if either exceeds 0.5,
the cap drops to `N - 1` and all subsequent rows where `N > cap` are
skipped. When the cap drops below 2, `scheduleLoop.finished = True` ends
the experiment.

The legacy `nback.psyexp` (lowercase) generated letters at runtime with a
20% target probability; it is kept in the repo for reference but is not
the active experiment.
```

- [ ] **Step 3: Spot-check the README renders**

Run: `head -130 README.md | tail -50`
Expected: the "Participant schedules" subsection and the rewritten "Implementation notes" both visible.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: describe nBlocks/topN flow and per-participant schedules"
```

---

## Self-review checklist (run before declaring done)

- [ ] All tests pass: `python3 -m unittest discover -s tests -v` shows 12 passes, 0 failures, 0 skipped.
- [ ] `prerandomize.py` is idempotent: running it twice in a row produces no diffs in `lists/` or `schedules/`.
- [ ] `schedules/000.csv` exists and has 51 lines (1 header + 50 data rows).
- [ ] `lists/1a.csv` is byte-identical to its pre-implementation state (proven by `tests/lists_sha256.txt` + `test_lists_dir_hash_stable`).
- [ ] `NBack.psyexp` opens cleanly in PsychoPy Builder with no warnings.
- [ ] Smoke-test sessions in Task 11 ran end-to-end and the data file contains `list_failed`, `list_miss_rate`, `list_fa_rate`, `correct` columns.
- [ ] README and HTML report describe the new flow.

---

## Summary of commits

A clean execution leaves this commit graph (one commit per major step, plus the inevitable Builder regenerations):

1. `test: add failing tests for generate_schedule`
2. `feat: add generate_schedule and MAX_BLOCKS constant`
3. `test: add failing tests for verify_schedule`
4. `feat: add verify_schedule and write_schedule`
5. `test: snapshot lists/ hash for reproducibility check`
6. `feat: emit 1000 per-participant schedules in prerandomize.py`
7. `feat(NBack): add nBlocks/topN expInfo and code_init`
8. `feat(NBack): wrap trials in scheduleLoop and switch to dynamic condsFile`
9. `feat(NBack): add scoreList routine + accumulator/scorer code`
10. `feat(NBack): wire skipIf and rewrite displayn text`
11. `docs: describe participant schedules in HTML report`
12. `docs: describe nBlocks/topN flow and per-participant schedules`
