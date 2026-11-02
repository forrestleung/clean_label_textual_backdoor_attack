"""
Forrest Leung
Last updated: 11/1/26

Converts Enron-Spam dataset from Hugging Face into TSV format, which the Gan et al. pipeline expects.

In terminal, run:
python data_preprocess/enron_prep.py --out data/clean_data/enron
"""
import argparse

import pandas as pd

if __package__:
    from .spam_prep_common import process_rows, strip_mime_headers
else:
    from spam_prep_common import process_rows, strip_mime_headers

# Pin to the main branch commit, so re-running the script pulls the same rows
ENRON_SPAM_REVISION = "1916f66c89d52221ae33eb57d44498b4f3a5df22"

def load_raw(input_csv):
    if input_csv:
        df = pd.read_csv(input_csv)
    else:
        from datasets import load_dataset
        d = load_dataset("SetFit/enron_spam", revision=ENRON_SPAM_REVISION)
        df = pd.concat([d[s].to_pandas() for s in d.keys()], ignore_index=True)

    cols = {c.lower(): c for c in df.columns}
    subject = df[cols["subject"]] if "subject" in cols else ""
    body = df[cols["message"]] if "message" in cols else df[cols["text"]]
    if "label" in cols:
        label = df[cols["label"]].astype(int)
    else:
        label = (df[cols["spam/ham"]].str.lower() == "spam").astype(int)

    # Strip MIME headers + markers from body before subject is prepended
    body = body.fillna("").astype(str).map(strip_mime_headers)
    text = subject.fillna("").astype(str) + " " + body \
        if isinstance(subject, pd.Series) else body
    return pd.DataFrame({"sentence": text, "label": label})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", default=None)
    p.add_argument("--out", default="data/clean_data/enron")
    p.add_argument("--max_words", type=int, default=64)

    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--dev_fraction", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=9620)
    p.add_argument("--seed", type=int, default=1024)
    args = p.parse_args()

    df = load_raw(args.input_csv)
    rows = [
        (sentence, int(label))
        for sentence, label in
        df[["sentence", "label"]].itertuples(index=False, name=None)
    ]
    process_rows(
        rows,
        args.out,
        args.max_words,
        min_words=3,
        seed=args.seed,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
        max_samples=args.max_samples,
    )

if __name__ == "__main__":
    main()