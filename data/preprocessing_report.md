# Overview of Preprocessed Spam Datasets

- Seed: `1024`
- Split ratios: `80/10/10`
- Token limit: `64`
- Subsampling: `Enabled`
- Enron/TREC sample cap: `9620` rows
- SpamAssassin source: `data/raw/spamassassin`
- TREC 2007 source: `data/raw/trec2007/processed_data.csv`
- Enron source: `SetFit/enron_spam @ 1916f66c89d52221ae33eb57d44498b4f3a5df22`

Labels: `0 = ham`, `1 = spam`.

All three datasets (`spamassassin`, `trec07`/`trec2007`, `enron`) run
the same MIME-header-stripping pipeline in `spam_prep_common.strip_mime_headers()`, applied to the body before the subject is
concatenated.

## SpamAssassin

Input: `data/raw/spamassassin`,
Output: `data/clean_data/spamassassin`

| Preprocessing step | Ham rows removed | Spam rows removed |
|---|---:|---:|
| too short | 0 | 3 |
| duplicates | 49 | 144 |
| conflicting duplicates | 0 | 0 |
| subsampling | 0 | 0 |

| Split | Total rows | Ham | Spam |
|---|---:|---:|---:|
| train | 4684 | 3283 | 1401 |
| dev | 586 | 411 | 175 |
| test | 585 | 410 | 175 |

Spam examples in test set (ASR denominator): 175

## TREC 2007

Input: `data/raw/trec2007/processed_data.csv`,
Output: `data/clean_data/trec2007`

| Preprocessing step | Ham rows removed | Spam rows removed |
|---|---:|---:|
| too short | 40 | 169 |
| duplicates | 945 | 14417 |
| conflicting duplicates | 15 | 41 |
| subsampling | 20323 | 29849 |

| Split | Total rows | Ham | Spam |
|---|---:|---:|---:|
| train | 7696 | 3117 | 4579 |
| dev | 962 | 390 | 572 |
| test | 962 | 390 | 572 |

Spam examples in test set (ASR denominator): 572

## Enron

Input: `SetFit/enron_spam @ 1916f66c89d52221ae33eb57d44498b4f3a5df22`,
Output: `data/clean_data/enron`

| Preprocessing step | Ham rows removed | Spam rows removed |
|---|---:|---:|
| too short | 5 | 63 |
| duplicates | 1053 | 3075 |
| conflicting duplicates | 0 | 0 |
| subsampling | 10440 | 9460 |

| Split | Total rows | Ham | Spam |
|---|---:|---:|---:|
| train | 7696 | 4037 | 3659 |
| dev | 962 | 505 | 457 |
| test | 962 | 505 | 457 |

Spam examples in test set (ASR denominator): 457