"""
Forrest Leung
Last updated: 11/1/26

Converts TREC 2007 spam data into TSV format, which the Gan et al. pipeline expects.

In terminal, run:
python data_preprocess/trec_prep.py --input data/raw/trec2007/processed_data.csv --out data/clean_data/trec2007
"""
import argparse
from pathlib import Path

if __package__:
    from .spam_prep_common import (
        label_from_value,
        load_csv,
        load_labelled_folders,
        process_rows,
        read_email,
        strip_mime_headers,
    )
else:
    from spam_prep_common import (
        label_from_value,
        load_csv,
        load_labelled_folders,
        process_rows,
        read_email,
        strip_mime_headers,
    )

def _find_index(root):
    candidates = (root / "index", root / "full" / "index")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None

def _resolve_message(root, index_path, record_path):
    relative_path = Path(record_path.lstrip("/"))
    candidates = (
        index_path.parent / record_path,
        root / record_path,
        root / relative_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate TREC message {record_path!r} from {index_path}"
    )

def load_trec_index(root, index_path):
    rows = []
    with index_path.open("r", encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            fields = line.rsplit(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"{index_path}:{line_number}: expected path and label")
            record_path, raw_label = fields
            try:
                label = label_from_value(raw_label)
            except ValueError:
                reverse_label = label_from_value(record_path)
                record_path = raw_label
                label = reverse_label
            message_path = _resolve_message(root, index_path, record_path)
            rows.append((read_email(message_path), label))
    return rows

def load_input(path):
    if path.is_file():
        return load_csv(path, clean_body=strip_mime_headers)
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")

    index_path = _find_index(path)
    if index_path:
        return load_trec_index(path, index_path)

    rows = load_labelled_folders(path)
    if rows:
        return rows
    raise ValueError(
        f"No TREC index, labeled ham/spam folders, or CSV/TSV found at {path}"
    )

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/clean_data/trec2007"))
    parser.add_argument("--max_words", type=int, default=64)
    parser.add_argument("--min_words", type=int, default=3)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--dev_fraction", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=9620)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    rows = load_input(args.input)
    if not rows:
        parser.error(f"No emails were loaded from {args.input}")
    process_rows(
        rows,
        args.out,
        args.max_words,
        args.min_words,
        args.seed,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
        max_samples=args.max_samples,
    )

if __name__ == "__main__":
    main()