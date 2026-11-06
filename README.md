# Telugu Automatic Text Correction for Low-Resource Languages

An IIIT Nagpur, CSE 5th-semester mini project for detecting and correcting Telugu spelling and orthographic errors. The project progresses from real-data preparation and deterministic correction baselines to an attention-based character Seq2Seq LSTM, with an mT5 fine-tuning workflow included for further work.

## Team

- Leela Venkata Sai Nerella — BT24CSA008
- Parthasaradhi Guda — BT24CSA061
- Dineswara Kumar Mangali — BT24CSA062

Supervisors: Ms. Dipti Pawade and Ms. Jaya Pathak

## Project highlights

- Prepared a real Telugu corpus from Telugu Wikipedia and processed Google Dakshina Telugu transliteration resources.
- Created controlled synthetic noisy-to-clean pairs for matra, consonant, deletion, swap, and word-boundary errors.
- Built an end-to-end pipeline with normalization, Tenglish-to-Telugu conversion, dictionary error detection, rule-based correction, and edit-distance correction.
- Implemented and compared four deterministic baselines.
- Trained an attention-based character Seq2Seq LSTM with checkpointing, early stopping, and conservative vocabulary-validated inference.
- Added an mT5-small fine-tuning notebook for a transformer-based next step.

## Recorded dataset statistics

| Artifact | Count |
| --- | ---: |
| Real Telugu corpus sentences | 97,154 |
| Telugu vocabulary entries | 100,000 |
| Dakshina word-level transliteration pairs | 69,980 |
| Dakshina full-sentence transliteration pairs | 10,000 |
| Synthetic training pairs | 60,000 |
| Synthetic validation pairs | 7,500 |
| Synthetic test pairs | 7,500 |

The corpus and synthetic datasets are stored in both `Notebooks/data/` and `text correctionfor telugu/data/` because the notebook workflow and modular pipeline use their own local relative paths.

## Recorded benchmark results

The following figures are recorded outputs from the 1,000-sentence synthetic test slice in `Notebooks/04_lstm_seq2seq_model.ipynb`. They measure controlled synthetic corruptions, so they should not be treated as a benchmark of naturally occurring user mistakes.

| Model | WER ↓ | CER ↓ | Exact match ↑ | BLEU ↑ | Token F1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncorrected baseline | 0.1838 | 0.0270 | 18.60% | 64.75 | 82.02% |
| Rule-based corrector | 0.1128 | 0.0180 | 33.90% | 77.53 | 89.15% |
| Dictionary + edit distance | 0.1269 | 0.0223 | 35.30% | 75.42 | 87.73% |
| Hybrid: rule + edit distance | 0.1212 | 0.0214 | 37.50% | 76.53 | 88.30% |
| Attention Seq2Seq LSTM | **0.0831** | **0.0113** | **53.60%** | **83.66** | **92.04%** |

Compared with the uncorrected input, the recorded Seq2Seq LSTM reduced WER by 54.8% and CER by 58.1% on that evaluation slice.

## Repository layout

```text
.
├── Notebooks/
│   ├── 01_data_extraction_and_synthetic_noise.ipynb
│   ├── 02_baseline_models_building.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_lstm_seq2seq_model.ipynb
│   ├── 05_mt5_small_telugu_gec_finetuning.ipynb
│   ├── data/                         # processed corpus and synthetic splits
│   └── models/                       # LSTM checkpoints
└── text correctionfor telugu/
    ├── src/                          # modular correction pipeline
    ├── scripts/                      # data, evaluation, and CLI demos
    ├── data/                         # pipeline-local data copy
    ├── requirements.txt
    └── test_setup.py
```

## Notebook guide

| Notebook | What it demonstrates |
| --- | --- |
| `01_data_extraction_and_synthetic_noise.ipynb` | Telugu corpus extraction, Unicode analysis, vocabulary construction, Dakshina exploration, synthetic noise generation, and split creation. |
| `02_baseline_models_building.ipynb` | Preprocessing, transliteration, error detection, rule-based, edit-distance, hybrid correction, and the unified pipeline. |
| `03_model_evaluation.ipynb` | WER, CER, BLEU, exact-match, precision, recall, F1, and comparison of the deterministic baselines. |
| `04_lstm_seq2seq_model.ipynb` | Attention-based character Seq2Seq LSTM training, checkpoints, evaluation, and visual comparison against baselines. |
| `05_mt5_small_telugu_gec_finetuning.ipynb` | Fine-tuning workflow for `google/mt5-small` on the synthetic correction pairs. |

`04_lstm_seq2seq_model copy.ipynb` is a saved training-resume copy and is retained as a project artifact.

## Modular pipeline

The `text correctionfor telugu/src/` package provides:

1. **Preprocessing** — NFC and IndicNLP normalization, cleanup, script detection, word/syllable tokenization, and Tenglish conversion using Dakshina mappings.
2. **Detection** — vocabulary-based out-of-vocabulary detection and a rule-based detector.
3. **Correction** — curated Telugu spelling fixes, phonological substitution rules, frequency-ranked edit-distance candidates, and a transformer-correction placeholder.
4. **Evaluation** — WER, CER, BLEU, exact match, token precision/recall/F1, and error-type analysis.

## Setup and use

Create a Python environment, install the project dependencies, then work from the modular-project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "text correctionfor telugu/requirements.txt"
cd "text correctionfor telugu"
python test_setup.py
python scripts/demo.py
```

Run the notebooks from inside `Notebooks/` so their relative `data/` and `models/` paths resolve correctly:

```powershell
cd Notebooks
jupyter notebook
```

## Checkpoints

The project includes three PyTorch artifacts:

- `Notebooks/LSTM_MODEL.pth` — exported LSTM model state dictionary (3.06 MiB).
- `Notebooks/models/telugu_attentive_lstm_best.pt` — best LSTM checkpoint with character mappings and configuration (6.51 MiB).
- `Notebooks/models/telugu_attentive_lstm_latest_checkpoint.pt` — resumable training checkpoint at recorded epoch 21 (12.25 MiB).

## Current validation status

- Notebook 1 has been executed successfully during the project workflow to create the real processed corpus and synthetic splits.
- The recorded baseline and LSTM results above come from completed notebook outputs.
- The mT5 notebook contains a complete training/evaluation workflow but does not contain a recorded completed run or saved mT5 checkpoint in this repository.
- `text correctionfor telugu/test_setup.py` currently stops at `EditDistanceCorrector(vocab=vocab)` because the current constructor accepts `vocab_path`, not `vocab`. The correction modules and notebook results are retained unchanged; this smoke-test script needs that argument updated before it can complete.

## Limitations and next steps

- Synthetic noise enables controlled experiments but does not replace a human-annotated Telugu grammar-correction dataset.
- Dictionary and edit-distance methods can struggle with context-dependent real-word errors, multi-word edits, and rare vocabulary.
- A stronger evaluation should use a sentence-disjoint, human-reviewed test set and report mT5 results after a completed fine-tuning run.
