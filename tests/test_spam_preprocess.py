"""
Forrest Leung
Last updated: 11/1/26

Tests for spam preprocessing functions.

In terminal, run:
python tests/test_spam_preprocess.py
"""
import unittest
import json
import tempfile
from pathlib import Path

from data_preprocess.spam_prep_common import (
    clean_text,
    prepare_rows,
    row_id,
    split_rows,
    stratified_sample,
    write_splits,
)

# Test cases for the spam preprocessing functions
class CleanTextTests(unittest.TestCase):
    def test_separates_punctuation(self):
        self.assertEqual(
            clean_text("Hello, world! Don't stop. (Really?)", 64),
            "Hello , world ! Do n't stop . ( Really ? )",
        )

    def test_preserves_leading_user_markers(self):
        self.assertEqual(
            clean_text("@USER @USER Re: What's up?", 64),
            "@ USER @ USER Re : What 's up ?",
        )

    def test_preserves_internal_user_marker(self):
        self.assertEqual(
            clean_text("Hello @USER, how are you?", 64),
            "Hello @ USER , how are you ?",
        )

    def test_truncates_after_tokenization(self):
        self.assertEqual(clean_text("one, two three", 3), "one , two")

    def test_deduplicates_case_and_whitespace_normalized_text_before_split(self):
        rows = [
            ("Same email, body.", 0),
            ("same   email, body.", 0),
            ("Different email.", 1),
            ("Another ham email.", 0),
            ("Another spam email.", 1),
            ("More ham email.", 0),
            ("More spam email.", 1),
        ]
        prepared, _ = prepare_rows(rows, max_words=64, min_words=1)
        self.assertEqual(len(prepared), 6)
        self.assertEqual(
            len({row_id(sentence) for sentence, _ in prepared}),
            len(prepared),
        )

    def test_discards_conflicting_duplicate_labels(self):
        rows = [
            ("Same email body.", 0),
            ("same   email body.", 1),
            ("Ham email one.", 0),
            ("Ham email two.", 0),
            ("Spam email one.", 1),
            ("Spam email two.", 1),
        ]
        prepared, stats = prepare_rows(rows, max_words=64, min_words=1)
        self.assertEqual(len(prepared), 4)
        self.assertNotIn("Same email body .", {text for text, _ in prepared})
        self.assertEqual(stats["conflicting_duplicates"], {0: 1, 1: 1})

    def test_stratified_sample_is_deterministic(self):
        rows = [(f"sample email number {index}", index % 2) for index in range(100)]
        first = stratified_sample(rows, 20, seed=1024)
        second = stratified_sample(rows, 20, seed=1024)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual({label for _, label in first}, {0, 1})

    def test_splits_are_stratified_and_80_10_10(self):
        rows = [(f"sample email number {index}", index % 2) for index in range(100)]
        splits = split_rows(rows, 0.8, 0.1, seed=1024)
        self.assertEqual(
            {name: len(split) for name, split in splits.items()},
            {"train": 80, "dev": 10, "test": 10},
        )
        for split in splits.values():
            self.assertEqual({label for _, label in split}, {0, 1})

    def test_writes_split_ids(self):
        splits = {
            "train": [("A message .", 0)],
            "dev": [("Another message .", 1)],
            "test": [("Last message .", 0)],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            write_splits(splits, output)
            manifest = json.loads((output / "split_ids.json").read_text())
        self.assertEqual(
            manifest,
            [
                {"split": "train", "id": row_id("A message ."), "label": 0},
                {"split": "dev", "id": row_id("Another message ."), "label": 1},
                {"split": "test", "id": row_id("Last message ."), "label": 0},
            ],
        )

if __name__ == "__main__":
    unittest.main()