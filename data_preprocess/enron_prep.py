"""
Forrest Leung
10/25/26

Converts Enron-Spam dataset from Hugging Face into TSV format,
which the Gan et al. pipeline expects.

In terminal, run:
python prep_enron.py --out data/clean_data/enron
"""
import argparse
import csv
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw(input_csv):
    if input_csv:
        df = pd.read_csv(input_csv)
    else:
        from datasets import load_dataset
        d = load_dataset("SetFit/enron_spam")
        df = pd.concat([d[s].to_pandas() for s in d.keys()], ignore_index=True)

    cols = {c.lower(): c for c in df.columns}
    subject = df[cols["subject"]] if "subject" in cols else ""
    body = df[cols["message"]] if "message" in cols else df[cols["text"]]
    if "label" in cols:
        label = df[cols["label"]].astype(int)
    else:  # Kaggle version: "Spam/Ham" column with strings
        label = (df[cols["spam/ham"]].str.lower() == "spam").astype(int)

    text = subject.fillna("").astype(str) + " " + body.fillna("").astype(str) \
        if isinstance(subject, pd.Series) else body.fillna("").astype(str)
    return pd.DataFrame({"sentence": text, "label": label})

def clean(text, max_words):
    # Tabs/newlines would break the TSV; collapse all whitespace to single spaces.
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    return " ".join(words[:max_words])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", default=None)
    p.add_argument("--out", default="data/clean_data/enron")
    p.add_argument("--max_words", type=int, default=64)

    # Defaults match SST-2's split sizes so runtimes are comparable.
    p.add_argument("--n_train", type=int, default=6920)
    p.add_argument("--n_dev", type=int, default=872)
    p.add_argument("--n_test", type=int, default=1821)
    p.add_argument("--seed", type=int, default=1024)
    args = p.parse_args()

    df = load_raw(args.input_csv)
    df["sentence"] = df["sentence"].map(lambda t: clean(t, args.max_words))
    df = df[df["sentence"].str.split(" ").str.len() >= 3]

    # Dedupe after truncation so no email appears in two splits.
    df = df.drop_duplicates(subset="sentence").reset_index(drop=True)
    print(f"after cleaning: {len(df)} emails, spam rate {df.label.mean():.3f}")

    n_total = args.n_train + args.n_dev + args.n_test
    if n_total < len(df):
        df, _ = train_test_split(df, train_size=n_total, stratify=df.label, random_state=args.seed)
    train, rest = train_test_split(df, train_size=args.n_train, stratify=df.label, random_state=args.seed)
    dev, test = train_test_split(rest, train_size=args.n_dev, stratify=rest.label, random_state=args.seed)

    os.makedirs(args.out, exist_ok=True)
    for name, split in [("train", train), ("dev", dev), ("test", test)]:
        path = os.path.join(args.out, f"{name}.tsv")
        split[["sentence", "label"]].to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
        # Round-trip with the same call the repo's read_data() uses.
        back = pd.read_csv(path, sep="\t").values.tolist()
        assert len(back) == len(split), f"{name}: row count changed on reload"
        assert all(int(r[1]) in (0, 1) for r in back), f"{name}: bad label on reload"
        lens = split.sentence.str.split(" ").str.len()
        print(f"{name}: {len(split)} rows, spam {split.label.mean():.3f}, "
              f"words mean {lens.mean():.1f} / max {lens.max()}")

if __name__ == "__main__":
    main()