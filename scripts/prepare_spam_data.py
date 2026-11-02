"""
Forrest Leung
Last updated: 11/1/26

Prepare spam corpora as reproducible, stratified TSV splits.
    
In terminal, run:
python scripts/prepare_spam_data.py --dataset all --output-root data/clean_data
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_preprocess.enron_prep import ENRON_SPAM_REVISION, load_raw as load_enron
from data_preprocess.spam_prep_common import (
    load_labelled_folders,
    process_rows,
    strip_mime_headers,
)
from data_preprocess.trec_prep import load_input as load_trec

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "spamassassin", "trec07", "enron"),
        default="all",
    )
    parser.add_argument(
        "--spamassassin-dir",
        type=Path,
        default=Path("data/raw/spamassassin"),
    )
    parser.add_argument(
        "--trec07-input",
        type=Path,
        default=Path("data/raw/trec2007/processed_data.csv"),
    )
    parser.add_argument("--enron-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("data/clean_data"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/preprocessing_report.md"),
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--max-words", type=int, default=64)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=9620)
    subsample = parser.add_mutually_exclusive_group()
    subsample.add_argument(
        "--subsample",
        dest="subsample",
        action="store_true",
        help="Cap Enron and TREC at --max-samples (the default).",
    )
    subsample.add_argument(
        "--no-subsample",
        dest="subsample",
        action="store_false",
        help="Use every deduplicated row in every dataset.",
    )
    parser.set_defaults(subsample=True)
    return parser.parse_args()

def load_dataset_rows(name, args):
    if name == "spamassassin":
        if not args.spamassassin_dir.is_dir():
            raise FileNotFoundError(
                f"SpamAssassin input directory not found: {args.spamassassin_dir}"
            )
        return load_labelled_folders(args.spamassassin_dir, clean_body=strip_mime_headers)
    if name == "trec07":
        return load_trec(args.trec07_input)

    frame = load_enron(str(args.enron_csv) if args.enron_csv else None)
    return [
        (text, int(label))
        for text, label in
        frame[["sentence", "label"]].itertuples(index=False, name=None)
    ]

def output_name(dataset):
    return "trec2007" if dataset == "trec07" else dataset

def build_report(args, results):
    lines = [
        "# Spam dataset preprocessing report",
        "",
        f"- Seed: `{args.seed}`",
        f"- Split ratios: `{args.train_fraction:.3f}/{args.dev_fraction:.3f}/"
        f"{1 - args.train_fraction - args.dev_fraction:.3f}`",
        f"- Token limit: `{args.max_words}`",
        f"- Subsampling: `{'enabled' if args.subsample else 'disabled'}`",
    ]
    if args.subsample:
        lines.append(f"- Enron/TREC sample cap: `{args.max_samples}` rows")
    lines.extend([
        f"- SpamAssassin source: `{args.spamassassin_dir}`",
        f"- TREC 2007 source: `{args.trec07_input}`",
        f"- Enron source: `{args.enron_csv or f'SetFit/enron_spam @ {ENRON_SPAM_REVISION}'}`",
    ])
    lines.extend(["", "Labels: `0 = ham`, `1 = spam`.", ""])

    for result in results:
        lines.extend([
            f"## {result['dataset']}",
            "",
            f"Input: `{result['input_path']}`",
            f"Output: `{result['output_dir']}`",
            "",
            "| Preprocessing step | Ham rows removed | Spam rows removed |",
            "|---|---:|---:|",
        ])
        for step, removed in result["removed"].items():
            lines.append(
                f"| {step.replace('_', ' ')} | {removed[0]} | {removed[1]} |"
            )
        lines.extend([
            "",
            "| Split | Total rows | Ham | Spam |",
            "|---|---:|---:|---:|",
        ])
        for split, counts in result["splits"].items():
            lines.append(
                f"| {split} | {sum(counts.values())} | {counts[0]} | {counts[1]} |"
            )
        lines.extend([
            "",
            f"Spam examples in test set (ASR denominator): **{result['test_spam']}**.",
        ])
        if result["test_spam"] < 200:
            lines.append(
                f"**WARNING:** only {result['test_spam']} spam test examples "
                "(fewer than 200)."
            )
        lines.append("")

    return "\n".join(lines)

def main():
    args = parse_args()
    if args.max_words < 1 or args.min_words < 1:
        raise ValueError("--max-words and --min-words must be positive")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.train_fraction <= 0 or args.dev_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if args.train_fraction + args.dev_fraction >= 1:
        raise ValueError("Train and dev fractions must sum to less than 1")

    selected = (
        ("spamassassin", "trec07", "enron")
        if args.dataset == "all"
        else (args.dataset,)
    )
    results = []
    for dataset in selected:
        rows = load_dataset_rows(dataset, args)
        if not rows:
            raise ValueError(f"No rows loaded for {dataset}")
        max_samples = (
            args.max_samples
            if args.subsample and dataset in ("trec07", "enron")
            else None
        )
        output_dir = args.output_root / output_name(dataset)
        result = process_rows(
            rows,
            output_dir,
            max_words=args.max_words,
            min_words=args.min_words,
            seed=args.seed,
            train_fraction=args.train_fraction,
            dev_fraction=args.dev_fraction,
            max_samples=max_samples,
            dataset_name=dataset,
        )
        if dataset == "spamassassin":
            result["input_path"] = str(args.spamassassin_dir)
        elif dataset == "trec07":
            result["input_path"] = str(args.trec07_input)
        else:
            result["input_path"] = str(
                args.enron_csv or f"SetFit/enron_spam @ {ENRON_SPAM_REVISION}"
            )
        results.append(result)

    report = build_report(args, results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"\nPreprocessing report saved to {args.report}")

if __name__ == "__main__":
    main()