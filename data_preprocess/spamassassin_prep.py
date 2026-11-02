"""
Forrest Leung
Last updated: 11/1/26

Converts an extracted SpamAssassin public corpus into TSV format, which the Gan et al. pipeline expects.

In terminal, run:
python data_preprocess/spamassassin_prep.py --input_dir data/raw/spamassassin --out data/clean_data/spamassassin
"""
import argparse
from pathlib import Path

if __package__:
    from .spam_prep_common import load_labelled_folders, process_rows, strip_mime_headers
else:
    from spam_prep_common import load_labelled_folders, process_rows, strip_mime_headers

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/clean_data/spamassassin"))
    parser.add_argument("--max_words", type=int, default=64)
    parser.add_argument("--min_words", type=int, default=3)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--dev_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"--input_dir is not a directory: {args.input_dir}")

    rows = load_labelled_folders(args.input_dir, clean_body=strip_mime_headers)
    if not rows:
        parser.error(
            f"No emails found under recognized SpamAssassin folders in {args.input_dir}"
        )
    process_rows(
        rows,
        args.out,
        args.max_words,
        args.min_words,
        args.seed,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
    )

if __name__ == "__main__":
    main()