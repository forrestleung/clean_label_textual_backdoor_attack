"""
Forrest Leung
Last updated: 11/1/26

Shared helpers used by enron_prep.py, spamassassin_prep.py, and trec_prep.py for the Gan et al. pipeline.
"""

import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path

from nltk.tokenize import TreebankWordTokenizer


FOLDER_LABELS = {
    "spam": 1,
    "spam_2": 1,
    "ham": 0,
    "ham_2": 0,
    "easy_ham": 0,
    "easy_ham_2": 0,
    "hard_ham": 0,
}

TEXT_COLUMNS = (
    "sentence",
    "text",
    "message",
    "body",
    "email",
    "email text",
    "email body",
    "content",
)
LABEL_COLUMNS = ("label", "class", "target", "category")
_TOKENIZER = TreebankWordTokenizer()

# HTML text extraction helper class
class _HTMLTextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

# Text cleaning and tokenization functions
def clean_text(text, max_words):
    if max_words < 1:
        raise ValueError("--max_words must be positive")
    text = collapse_whitespace(text)
    text = re.sub(r"(?<=[\w])\.(?=\s|$|[)\]}])", " . ", text)
    tokens = _TOKENIZER.tokenize(text)
    return " ".join(tokens[:max_words])

# Whitespace collapsing function
def collapse_whitespace(text):
    return re.sub(r"\s+", " ", str(text)).strip()


# MIME header and boundary detection patterns
_H = r"\s*-\s*"
_SP = r"\s*"
_CTE_VALUES = (
    rf"7{_SP}bit|8{_SP}bit|base{_SP}64|quoted{_H}printable|binary"
)

_MIME_HEADER_RE = re.compile(
    rf"^(?:Received|From|To|Cc|Bcc|Subject|Reply{_H}To|Return{_H}Path|"
    rf"Message{_H}ID|In{_H}Reply{_H}To|References|MIME{_H}Version|Date|"
    rf"Content(?:{_H}\w+)+|X(?:{_H}\w+)+){_SP}:",
    re.IGNORECASE,
)
_MIME_CTE_RUNON_RE = re.compile(
    rf"^Content{_H}Transfer{_H}Encoding{_SP}:{_SP}(?:{_CTE_VALUES})"
    r"\s+(?=\S)",
    re.IGNORECASE,
)
# Require at least one letter/digit in the token
_MIME_BOUNDARY_RE = re.compile(r"^--(?=\S*[A-Za-z0-9])\S+$")

# Catches MIME tokens that aren't at a line start and deliberately narrows to MIME-specific syntax
_MIME_CONTENT_TYPE_MIDLINE_RE = re.compile(
    rf"Content{_H}Type{_SP}:{_SP}[\w.+-]+\s*/\s*[\w.+-]+"
    rf"(?:\s*;\s*(?:charset|boundary){_SP}={_SP}(?:\"[^\"]*\"|'[^']*'|[\w.-]+))*",
    re.IGNORECASE,
)
_MIME_CTE_MIDLINE_RE = re.compile(
    rf"Content{_H}Transfer{_H}Encoding{_SP}:{_SP}(?:{_CTE_VALUES})",
    re.IGNORECASE,
)
_MIME_PARAM_MIDLINE_RE = re.compile(
    rf"(?<![\w-])(?:charset|boundary){_SP}={_SP}(?:\"[^\"]*\"|'[^']*'|[\w.-]+)",
    re.IGNORECASE,
)
_MIME_BOUNDARY_TOKEN_MIDLINE_RE = re.compile(
    r"-{2,}(?:[_=]+[\w.=_-]{4,}|Apple-Mail[\w=_-]*)",
    re.IGNORECASE,
)

# Replacement function for MIME boundary tokens
def _boundary_token_replacement(match):
    return " " if any(c.isalnum() for c in match.group()) else match.group()

# Drop MIME headers, header continuations, and MIME boundary markers from text
def strip_mime_headers(text):
    out_lines = []
    in_header_continuation = False
    for line in str(text).splitlines():
        runon = _MIME_CTE_RUNON_RE.match(line)
        if runon:
            out_lines.append(line[runon.end():])
            in_header_continuation = False
            continue
        if _MIME_HEADER_RE.match(line):
            in_header_continuation = True
            continue
        if in_header_continuation and line[:1] in (" ", "\t"):
            continue
        in_header_continuation = False
        if _MIME_BOUNDARY_RE.match(line.strip()):
            continue
        out_lines.append(line)
    result = "\n".join(out_lines)

    result = _MIME_CONTENT_TYPE_MIDLINE_RE.sub(" ", result)
    result = _MIME_CTE_MIDLINE_RE.sub(" ", result)
    result = _MIME_PARAM_MIDLINE_RE.sub(" ", result)
    result = _MIME_BOUNDARY_TOKEN_MIDLINE_RE.sub(_boundary_token_replacement, result)
    return result

# Reads an email from a file path, extracts the subject and body, and optionally cleans the body using a provided function.
def read_email(path, clean_body=None):
    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    subject = str(message.get("subject", ""))
    plain_parts = []
    html_parts = []

    for part in message.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type not in ("text/plain", "text/html"):
            continue

        payload = part.get_payload(decode=True)
        if payload is None:
            content = str(part.get_payload())
        else:
            charset = part.get_content_charset() or "utf-8"
            try:
                content = payload.decode(charset, errors="replace")
            except LookupError:
                content = payload.decode("utf-8", errors="replace")

        if content_type == "text/plain":
            plain_parts.append(content)
        else:
            html_parts.append(content)

    body_parts = plain_parts or html_parts
    body = " ".join(body_parts)
    if not plain_parts and html_parts:
        parser = _HTMLTextParser()
        parser.feed(body)
        body = " ".join(parser.parts)

    if clean_body:
        body = clean_body(body)
    return " ".join(part for part in (subject, body) if part)

# Converts a given value to a binary label (0 for ham, 1 for spam).
def label_from_value(value):
    normalized = str(value).strip().lower()
    if normalized in ("spam", "junk"):
        return 1
    if normalized in ("ham", "legitimate", "legit", "non-spam", "nonspam", "not spam"):
        return 0
    try:
        numeric = float(normalized)
    except ValueError as exc:
        raise ValueError(f"Unrecognized email label: {value!r}") from exc
    if numeric in (0, 1):
        return int(numeric)
    raise ValueError(f"Email label must be ham/spam or 0/1, got {value!r}")

# Finds the column name in a list of field names that matches any of the candidate names.
def find_column(fieldnames, candidates):
    normalized = {
        re.sub(r"[^a-z0-9]", "", name.lower()): name
        for name in fieldnames
        if name
    }
    for candidate in candidates:
        match = normalized.get(re.sub(r"[^a-z0-9]", "", candidate))
        if match:
            return match
    raise ValueError(
        f"Could not find a column matching any of {', '.join(candidates)} "
        f"in {fieldnames}"
    )

# Loads CSV/TSV files containing text and label columns, optionally cleaning the body text.
def load_csv(path, clean_body=None):
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        text_column = find_column(reader.fieldnames, TEXT_COLUMNS)
        label_column = find_column(reader.fieldnames, LABEL_COLUMNS)
        normalized_names = {
            re.sub(r"[^a-z0-9]", "", name.lower()): name
            for name in reader.fieldnames
            if name
        }
        subject_column = normalized_names.get("subject")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if row.get(text_column) is None or row.get(label_column) is None:
                raise ValueError(f"{path}:{line_number} is missing text or label")
            body = row[text_column]
            if clean_body:
                body = clean_body(body)
            if subject_column and subject_column != text_column:
                text = f"{row.get(subject_column) or ''} {body}".strip()
            else:
                text = body
            rows.append((text, label_from_value(row[label_column])))
    return rows

# Loads labelled email files from a folder structure, optionally cleaning the body text.
def load_labelled_folders(root, clean_body=None):
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        label = next(
            (FOLDER_LABELS[parent.name.lower()] for parent in path.parents
             if parent.name.lower() in FOLDER_LABELS),
            None,
        )
        if label is not None:
            rows.append((read_email(path, clean_body=clean_body), label))
    return rows

# Prepares rows by cleaning text, removing too short emails, and handling duplicates.
def prepare_rows(rows, max_words, min_words):
    if min_words < 1:
        raise ValueError("--min_words must be positive")
    groups = defaultdict(list)
    stats = {
        "input": {0: 0, 1: 0},
        "too_short": {0: 0, 1: 0},
        "duplicates": {0: 0, 1: 0},
        "conflicting_duplicates": {0: 0, 1: 0},
    }
    for text, label in rows:
        if label not in (0, 1):
            raise ValueError(f"Expected a binary label, got {label!r}")
        stats["input"][label] += 1
        sentence = clean_text(text, max_words)
        content_hash = row_id(sentence)
        if len(sentence.split()) < min_words:
            stats["too_short"][label] += 1
            continue
        groups[content_hash].append((sentence, label))

    prepared = []
    for group in groups.values():
        labels = {label for _, label in group}
        if len(labels) > 1:
            for _, label in group:
                stats["conflicting_duplicates"][label] += 1
            continue
        prepared.append(group[0])
        stats["duplicates"][group[0][1]] += len(group) - 1
    if not prepared:
        raise ValueError("No usable emails remain after cleaning")
    if {label for _, label in prepared} != {0, 1}:
        raise ValueError("Input must contain at least one ham and one spam email")
    stats["after_deduplication"] = {
        label: sum(row_label == label for _, row_label in prepared)
        for label in (0, 1)
    }
    return prepared, stats

# Splits a given size into counts according to the specified fractions, ensuring each count is at least 1 and the total matches the size.
def _split_counts(size, fractions):
    exact = [size * fraction for fraction in fractions]
    counts = [max(1, int(value)) for value in exact]

    while sum(counts) > size:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        if not candidates:
            raise ValueError("Each class needs at least three examples to split")
        index = max(candidates, key=lambda i: counts[i] - exact[i])
        counts[index] -= 1

    while sum(counts) < size:
        index = max(range(len(counts)), key=lambda i: exact[i] - counts[i])
        counts[index] += 1
    return counts

# Performs stratified sampling of rows to ensure each class is represented proportionally in the sample.
def stratified_sample(rows, sample_size, seed):
    if sample_size < 1:
        raise ValueError("--max_samples must be positive")
    if sample_size >= len(rows):
        return rows

    by_label = defaultdict(list)
    for row in rows:
        by_label[row[1]].append(row)
    if sample_size < len(by_label) * 3:
        raise ValueError(
            "--max_samples must allow at least three examples per class "
            "for stratified train/dev/test splits"
        )

    labels = sorted(by_label)
    quotas = [sample_size * len(by_label[label]) / len(rows) for label in labels]
    counts = [min(len(by_label[label]), max(1, int(quota))) for label, quota in zip(labels, quotas)]
    while sum(counts) < sample_size:
        candidates = [
            i for i, label in enumerate(labels)
            if counts[i] < len(by_label[label])
        ]
        index = max(candidates, key=lambda i: quotas[i] - counts[i])
        counts[index] += 1
    while sum(counts) > sample_size:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        index = max(candidates, key=lambda i: counts[i] - quotas[i])
        counts[index] -= 1

    rng = random.Random(seed)
    sampled = []
    for label, count in zip(labels, counts):
        sampled.extend(rng.sample(by_label[label], count))
    rng.shuffle(sampled)
    return sampled

# Splits rows into train, dev, and test sets in a stratified manner, ensuring each class is represented proportionally in each split.
def split_rows(rows, train_fraction, dev_fraction, seed):
    if train_fraction <= 0 or dev_fraction <= 0 or train_fraction + dev_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than 1")

    fractions = (train_fraction, dev_fraction, 1 - train_fraction - dev_fraction)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row[1]].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "dev": [], "test": []}
    for label, examples in sorted(by_label.items()):
        if len(examples) < 3:
            raise ValueError(
                f"Class {label} has {len(examples)} examples; at least three "
                "are required for stratified train/dev/test splits"
            )
        rng.shuffle(examples)
        train_count, dev_count, _ = _split_counts(len(examples), fractions)
        splits["train"].extend(examples[:train_count])
        splits["dev"].extend(examples[train_count:train_count + dev_count])
        splits["test"].extend(examples[train_count + dev_count:])

    for examples in splits.values():
        rng.shuffle(examples)
    return splits

# Generates a unique identifier for a given sentence by normalizing it and computing its SHA-256 hash.
def row_id(sentence):
    normalized = " ".join(sentence.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

# Writes the train, dev, and test splits to TSV files and generates a JSON file containing the split IDs.
def write_splits(splits, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_ids = []
    for name, rows in splits.items():
        path = output_dir / f"{name}.tsv"
        with path.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(("sentence", "label"))
            writer.writerows(rows)

        with path.open("r", encoding="utf-8", newline="") as source:
            loaded = list(csv.reader(source, delimiter="\t"))
        if len(loaded) != len(rows) + 1:
            raise RuntimeError(f"{path}: TSV row count changed on reload")
        if any(len(row) != 2 or row[1] not in ("0", "1") for row in loaded[1:]):
            raise RuntimeError(f"{path}: invalid TSV row on reload")

        spam_rate = sum(label for _, label in rows) / len(rows)
        mean_words = sum(len(text.split()) for text, _ in rows) / len(rows)
        max_words = max(len(text.split()) for text, _ in rows)
        split_ids.extend(
            {"split": name, "id": row_id(text), "label": label}
            for text, label in rows
        )
        print(
            f"{name}: {len(rows)} rows, spam {spam_rate:.3f}, "
            f"words mean {mean_words:.1f} / max {max_words}"
        )

    ids_path = output_dir / "split_ids.json"
    with ids_path.open("w", encoding="utf-8") as target:
        json.dump(split_ids, target, indent=2)
        target.write("\n")

# Processes the given rows by preparing them, optionally subsampling, splitting into train/dev/test, and writing the splits to files.
def process_rows(rows, output_dir, max_words, min_words, seed,
                 train_fraction=0.8, dev_fraction=0.1, max_samples=None,
                 dataset_name="dataset"):
    prepared, stats = prepare_rows(rows, max_words, min_words)
    if max_samples is not None and max_samples < 0:
        raise ValueError("--max_samples cannot be negative")
    stats["subsampling"] = {0: 0, 1: 0}
    if max_samples is not None and max_samples > 0 and max_samples < len(prepared):
        prepared = stratified_sample(prepared, max_samples, seed)
        for label in (0, 1):
            stats["subsampling"][label] = (
                stats["after_deduplication"][label]
                - sum(row_label == label for _, row_label in prepared)
            )
    stats["after_subsampling"] = {
        label: sum(row_label == label for _, row_label in prepared)
        for label in (0, 1)
    }

    splits = split_rows(
        prepared,
        train_fraction,
        dev_fraction,
        seed,
    )
    write_splits(splits, output_dir)
    split_stats = {
        name: {
            label: sum(row_label == label for _, row_label in split)
            for label in (0, 1)
        }
        for name, split in splits.items()
    }
    result = {
        "dataset": dataset_name,
        "input": stats["input"],
        "removed": {
            "too_short": stats["too_short"],
            "duplicates": stats["duplicates"],
            "conflicting_duplicates": stats["conflicting_duplicates"],
            "subsampling": stats["subsampling"],
        },
        "after_deduplication": stats["after_deduplication"],
        "after_subsampling": stats["after_subsampling"],
        "splits": split_stats,
        "test_spam": split_stats["test"][1],
        "output_dir": str(output_dir),
    }
    print(f"\n{dataset_name}: preprocessing summary")
    print(
        f"  input rows: {sum(stats['input'].values())} "
        f"(ham={stats['input'][0]}, spam={stats['input'][1]})"
    )
    print("  rows removed by step (ham, spam):")
    for step, counts in result["removed"].items():
        print(f"    {step}: ham={counts[0]}, spam={counts[1]}")
    print("  final splits:")
    for name, counts in split_stats.items():
        print(
            f"    {name}: {sum(counts.values())} rows "
            f"(ham={counts[0]}, spam={counts[1]})"
        )
    print(f"  spam test examples for ASR: {result['test_spam']}")
    if result["test_spam"] < 200:
        print(
            f"  WARNING: test set has only {result['test_spam']} spam examples "
            "(fewer than 200)"
        )
    return result