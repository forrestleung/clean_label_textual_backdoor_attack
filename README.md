# Trigerless Backdoor Attack for NLP Tasks with Clean Labels

## Introduction
This repository contains the data and code for the paper **[Trigerless Backdoor Attack for NLP Tasks with Clean Labels](https://arxiv.org/abs/2111.07970)**.
<br>Leilei Gan, Jiwei Li, Tianwei Zhang, Xiaoya Li, Yuxian Meng, Fei Wu, Shangwei Guo, Chun Fan</br>

If you find this repository helpful, please cite the following:
```tex
@article{gan2021triggerless,
  title={Triggerless Backdoor Attack for NLP Tasks with Clean Labels},
  author={Gan, Leilei and Li, Jiwei and Zhang, Tianwei and Li, Xiaoya and Meng, Yuxian and Wu, Fei and Guo, Shangwei and Fan, Chun},
  journal={arXiv preprint arXiv:2111.07970},
  year={2021}
}
```

## Requirements
* Python == 3.7
* `pip install -r requirements.txt`

We also rely on some external resources, you can manually download them and put them into corresponding directories.

- Download [Counter-fitted word vectors](https://cdn.data.thunlp.org/TAADToolbox/counter-fitted-vectors.txt.zip), and put it into the ``data/AttackAssist.CounterFit`` directory.
- Download [Structure controlled paraphrasing model](https://cdn.data.thunlp.org/TAADToolbox/scpn.zip), and put it into the ``data/AttackAssist.SCPN`` directory.
- Download [Sentence tokenizer model](https://cdn.data.thunlp.org/TAADToolbox/punkt.english.pickle.zip), and put it into the ``data/TProcess.NLTKSentTokenizer`` directory.
- Download [Language Tool model](https://www.languagetool.org/download/LanguageTool-stable.zip) following the instruction in [Language Tool](https://github.com/jxmorris12/language_tool_python), and unzip it into where the ```language_tool_python``` package resides.

## Preparing spam-filtering datasets

Run the unified preprocessing pipeline with:

```shell
python scripts/prepare_spam_data.py --dataset all
```

Select one corpus with `--dataset spamassassin`, `--dataset trec07`, or
`--dataset enron`. Dataset paths, output/report paths, seed, split fractions,
token limits, and subsampling are configurable via `--help`. The defaults use
seed 1024, 80/10/10 stratified splits, 64 tokens, and a 9,620-row cap for
Enron/TREC; pass `--no-subsample` to use all deduplicated examples. The run
prints its preprocessing summary and writes `data/preprocessing_report.md`.

The pipeline produces `train.tsv`, `dev.tsv`, and `test.tsv` with
`sentence` and binary `label` columns (`0` = ham, `1` = spam). They normalize
whitespace, deduplicate after truncation, and make stratified splits. The
Enron, SpamAssassin, and TREC scripts share these cleanup, split, and TSV-writing
steps through `data_preprocess/spam_prep_common.py`. Cleanup collapses
whitespace, preserves email subject prefixes such as `Re:`, and uses NLTK's
Treebank tokenizer to separate punctuation consistently.
All three datasets use stratified 80/10/10 train/dev/test splits by default.
Deduplication uses a SHA-256 hash of lowercased, whitespace-normalized text
after model-length truncation. Conflicting labels for duplicate text are
excluded and reported. Each output directory contains `split_ids.json` with
stable content hashes and labels for the exact rows assigned to each split.
Enron and TREC are stratified-sampled to 9,620 emails by default (about 7,696
train and 962 each for dev/test); use `--max_samples 0` to disable the cap.
SpamAssassin uses all deduplicated examples; its current test split contains
175 spam emails for ASR evaluation.

SpamAssassin input defaults to the extracted folder layout (`spam`, `easy_ham`,
etc.); TREC defaults to the downloaded preprocessed CSV; Enron defaults to
`SetFit/enron_spam` from Hugging Face.

## Running the attack on a spam dataset

Every spam script takes `[dataset] [device]`: dataset is `enron` (default),
`spamassassin` or `trec2007`; device defaults to `cuda:0`. The attack targets a
spam test email (label 1) and perturbs ham training emails (label 0), which keep
their true ham label. It succeeds when the backdoored filter classifies the
target spam as ham.

```shell
bash scripts/run_spam_pipeline.sh enron cuda:0   # all three steps; finished steps are skipped
```

or step by step:

```shell
bash scripts/run_bert_spam_clean.sh enron cuda:0        # 1. clean victim -> checkpoints/clean_bert_base_enron_mlp0/
bash scripts/run_bert_spam_samples_gen.sh enron cuda:0  # 2. poisons -> data/clean_data/aux_files/enron/*.pkl
bash scripts/run_bert_spam_attack.sh enron cuda:0       # 3. filter poisons, train backdoors, print ASR / CACC
```

Logs go to `logs/<dataset>/`. Settings in `scripts/spam_common.sh` can be
overridden with environment variables (`ATTACK_NUM`, `TOP_BASE_NUM`, `POP_SIZE`,
`ITER_NUM`, `POISON_NUM`, `EPOCH`, `LR`). For example, a poisoning-budget sweep:

```shell
for n in 10 20 40 100 200; do POISON_NUM=$n bash scripts/run_bert_spam_attack.sh enron cuda:0; done
```

**Poison filter.** Gan et al. keep a poison only if its GPT-2 perplexity,
LanguageTool error count and BERTScore similarity pass fixed SST-2 thresholds.
Emails are much noisier than SST-2, so by default the spam runs set the PPL and
grammar thresholds to the 90th percentile of 500 genuine ham training emails:
a poison is kept if it is no less fluent than 90% of real ham. Those values are
cached next to the poison file (`*.ref_thresholds.p90.json`), and the filtered
poisons are cached as `*.filtered.ppl…`. To use fixed thresholds instead, pass
`--ppl_thresh`, `--gerr_thresh`, `--bertscore_thresh` or `--ref_percentile` after the
device. If no poison survives for a target, that target is skipped and counted
as a failure. The log reports ASR over all targets and over attacked targets only.

The first poison-generation run downloads the counter-fitted word vectors
(~65 MB) into `data/AttackAssist.CounterFit/`. The attack step needs Java for
LanguageTool and downloads GPT-2 and the LanguageTool server on first use.

**Settings matched to Gan et al. (2022).** Poison generation uses the paper's
hyperparameters: 300 candidate ham emails per target (`--top_base_num 300`),
K=60 masked-LM substitutes per word (set in `OpenAttack/attackers/genetic.py`),
and a 0.4 counter-fitted-vector antonym filter (`--cf_thresh 0.4`). The genetic
population (20) and iterations (15) follow the authors' released code. Backdoor
training freezes the pre-trained BERT, uses batch size 32, and injects
`POISON_NUM=40` poisons per target (the paper reports ~40 gives near-100% ASR on
SST-2); the budget sweep above varies this. `ATTACK_NUM=10` targets are attacked
by default (the paper uses 300; 10 is a smaller, faster subset — ASR then moves
in steps of 10%).

### Defending the spam backdoor (ONION)

After the attack has trained per-target backdoor models, run the ONION
perplexity defense (Gan et al. Table 5):

```shell
bash scripts/run_bert_spam_defend.sh enron cuda:0        # DEFENSE=ONION by default
```

ONION removes high-perplexity words from each input and re-checks whether the
target spam is still misclassified as ham; the log sweeps the perplexity bar and
prints ASR and clean accuracy per bar. It auto-discovers the clean checkpoint,
the backdoor-model directory for the current `POISON_NUM`, and the filtered
poison file. Targets whose poisons were all filtered out (no backdoor model) are
skipped. ONION needs only GPT-2. The `BT` (back-translation, needs `fairseq`)
and `SCPN` (needs OpenAttack's SCPN model) defenses can be selected with
`DEFENSE=BT` / `DEFENSE=SCPN` but have heavier dependencies. The same reproduces
on SST-2 with `bash scripts/run_bert_sst_defend.sh cuda:0`.

## Train the Clean Victim Model.
```shell
bash scripts/run_bert_sst_clean.sh
```

## Poisoned Sample Generation

```shell
bash scripts/run_bert_sst_samples_gen.sh
```


## Attack

```shell
bash scripts/run_bert_sst_attack.sh
```
Note that the PPL, GErr and BertScore constraints for the generated poisoned samples should be adjusted accordingly to achieve a balance between stealthiness and attacking success rates.

Table 1: Main attacking results. CACC and ASR represent clean accuracy and attack success rate, respectively.
<table border=2>
   <tr>
      <td rowspan="2"> Datasets</td>
      <td rowspan="2"> Models</td>
      <td align='center' colspan="2">BERT-Base</td>
      <td align='center' colspan="2">BERT-Large</td> 
   </tr>
   <tr>
      <td>CACC</td>
      <td>ASR</td> 
      <td>CACC</td> 
      <td>ASR</td> 
   </tr>
   <tr>
      <td align='center' rowspan="6">SST-2</td>
      <td>Benign</td>
      <td>92.3</td>
      <td>-</td>
      <td>93.1</td>
      <td>-</td>
   </tr>
   <tr>
      <td>BadNet</td>
      <td>90.9</td>
      <td>100</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>RIPPLES</td>
      <td>90.7</td>
      <td>100</td>
      <td>91.6</td>
      <td>100</td>
   </tr>
   <tr>
      <td>Syntactic</td>
      <td>90.9</td>
      <td>98.1</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>LWS</td>
      <td>88.6</td>
      <td>97.2</td>
      <td>90.0</td>
      <td>97.4</td>
   </tr>
   <tr>
      <td><b>Ours</b></td>
      <td>89.7</td>
      <td>98.0</td>
      <td>90.8</td>
      <td>99.1</td>
   </tr>

   <tr>
      <td align='center' rowspan="6">OLID</td>
      <td>Benign</td>
      <td>84.1</td>
      <td>-</td>
      <td>83.8</td>
      <td>-</td>
   </tr>
   <tr>
      <td>BadNet</td>
      <td>82.0</td>
      <td>100</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>RIPPLES</td>
      <td>83.3</td>
      <td>100</td>
      <td>83.7</td>
      <td>100</td>
   </tr>
   <tr>
      <td>Syntactic</td>
      <td>82.5</td>
      <td>99.1</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>LWS</td>
      <td>82.9</td>
      <td>97.1</td>
      <td>81.4</td>
      <td>97.9</td>
   </tr>
   <tr>
      <td><b>Ours</b></td>
      <td>83.1</td>
      <td>99.0</td>
      <td>82.5</td>
      <td>100</td>
   </tr>
   <tr>
      <td align='center' rowspan="6">AG's News</td>
      <td>Benign</td>
      <td>93.6</td>
      <td>-</td>
      <td>93.5</td>
      <td>-</td>
   </tr>
   <tr>
      <td>BadNet</td>
      <td>93.9</td>
      <td>100</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>RIPPLES</td>
      <td>92.3</td>
      <td>100</td>
      <td>91.6</td>
      <td>100</td>
   </tr>
   <tr>
      <td>Syntactic</td>
      <td>94.3</td>
      <td>100</td>
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>LWS</td>
      <td>92.0</td>
      <td>99.6</td>
      <td>92.6</td>
      <td>99.5</td>
   </tr>
   <tr>
      <td><b>Ours</b></td>
      <td>92.5</td>
      <td>92.8</td>
      <td>90.1</td>
      <td>96.7</td>
   </tr>
</table>

## Defend
Here, we test whether ONION, back-translation based paraphrasing defense and syntactically controlled paraphrasing defense can successfully defend our triggerless textual backdoor attack method.

  ```shell
  bash script/run_bert_sst_defend.sh 
  ```

Table 2.  Attacking results against three defense methods on the SST-2 dataset.
<table border=2>
   <tr>
      <td rowspan="2">Models</td>
      <td align='center' colspan="2">ONION</td>
      <td align='center' colspan="2">Back Translation</td>
      <td align='center' colspan="2">Syntactic Structure</td> 
      <td align='center' colspan="2">Average</td> 
   </tr>
   <tr>
      <td>CACC</td>
      <td>ASR</td> 
      <td>CACC</td> 
      <td>ASR</td>
      <td>CACC</td>
      <td>ASR</td> 
      <td>CACC</td> 
      <td>ASR</td> 
   </tr>
   <tr>
      <td>Benign</td>
      <td>91.32</td> 
      <td>-</td> 
      <td>89.79</td>
      <td>-</td>
      <td>82.02</td> 
      <td>-</td> 
      <td>87.71</td>
      <td>-</td>
   </tr>
   <tr>
      <td>BadNet</td>
      <td>89.95</td> 
      <td>40.30</td> 
      <td>84.78</td>
      <td>49.94</td>
      <td>81.86</td> 
      <td>58.27</td> 
      <td>85.31(↓ 3.4)</td>
      <td>49.50(↓ 50.50)</td>
   </tr>
   <tr>
      <td>RIPPLES</td>
      <td>88.90</td> 
      <td>17.80</td> 
      <td>-</td>
      <td>-</td>
      <td>-</td> 
      <td>-</td> 
      <td>-</td>
      <td>-</td>
   </tr>
   <tr>
      <td>Syntactic</td>
      <td>89.84</td> 
      <td>98.02</td> 
      <td>80.64</td>
      <td>91.64</td>
      <td>79.28</td> 
      <td>61.97</td> 
      <td>83.25(↓ 5.98)</td>
      <td>83.87(↓ 15.23)</td>
   </tr>
   <tr>
      <td>LWS</td>
      <td>87.30</td> 
      <td>92.90</td> 
      <td>86.00</td>
      <td>74.10</td>
      <td>77.90</td> 
      <td>75.77</td> 
      <td>83.73(↓ 4.10)</td>
      <td>80.92(↓ 17.08)</td>
   </tr>
   <tr>
      <td><br>Ours</br></td>
      <td>89.70</td> 
      <td>98.00</td> 
      <td>87.05</td>
      <td>88.00</td>
      <td>80.50</td> 
      <td>76.00</td> 
      <td>85.75(↓ 2.68)</td>
      <td>87.33(↓ 9.27)</td>
   </tr>
</table>

## Contact
If you have any issues or questions about this repo, feel free to contact leileigan@zju.edu.cn.

## License
[Apache License 2.0](./LICENSE) 
