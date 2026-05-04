"""Tests for prerandomize.py — schedule generation and existing-list reproducibility."""
from __future__ import annotations

import hashlib
import random
import sys
import unittest
from pathlib import Path

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
        bad = list(self.good_rows)
        bad[0], bad[-1] = bad[-1], bad[0]
        with self.assertRaises(AssertionError):
            self._verify(bad)


class TestExistingListsUnchanged(unittest.TestCase):
    """Re-running prerandomize.py with DEFAULT_SEED must not change lists/*.csv.

    Existing list files are checked into the repo; this asserts that adding
    schedule generation doesn't shift the master_rng's first 53 sub-seeds.
    """

    def test_lists_dir_hash_stable(self) -> None:
        lists_dir = ROOT / "lists"
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


if __name__ == "__main__":
    unittest.main()
