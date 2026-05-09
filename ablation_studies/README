# Pixels to Predictions — Ablation Studies Directory

This README maps every ablation study run directory to the corresponding trial numbers in **Table 3** (`tab:experiments`) of the report.

---

## Directory Structure

```
experiments/
├── run_0/    # Ablation sweep         → T0–T22
├── run_2/    # img_size supplementary → T19–T22 (additional metrics)
├── run_3/    # Improved baseline      → T24, T25, T26
├── run_4/    # Frozen encoder, prompt → T22 (frozen), T23, context ablation
├── run_5/    # Long training          → T27
├── run_7/    # img_size=512           → T28
├── run_8/    # Train+val retrain      → T29
└── run_9/    # Context at inference   → T30
```

---

## Trial-to-Directory Mapping

### `run_0/` — Baseline (T0)

| Trial | Directory | Key Change | Val Acc | Kaggle |
|---|---|---|---|---|
| T0  | `run_0/`  | Baseline: epoch=1, r=8, α=16, lr=2e-4, dropout=0.05, warmup=0.05, targets=q/k/v/o | 0.4476 | 0.560 |


---

### `run_1/` — Baseline Epochs (T1)

| Trial | Directory | Key Change | Val Acc | Kaggle |
|---|---|---|---|---|
| T1  | `run_1/`  | epochs=5 | 0.5524 | 0.660 |

---

### `run_2/` — Baseline Ablation Sweep (T24–T26)

| Trial | Directory | Key Change | Val Acc | Kaggle |
|---|---|---|---|---|
| T2  | `run_2/`  | r=4, α=8 (1.04M trainable params) | 0.4095 | — |
| T3  | `run_2/`  | r=16, α=32 (4.16M trainable params) | 0.5333 | — |
| T4  | `run_2/`  | r=32, α=64 — **exceeds 5M param limit, not trained** | N/A | — |
| T5  | `run_2/`  | lr=2e-2 | 0.3048 | — |
| T6  | `run_2/`  | lr=2e-3 | 0.4952 | — |
| T7  | `run_2/`  | lr=2e-5 | 0.3905 | — |
| T8  | `run_2/`  | lr=8e-5 | 0.3333 | — |
| T9  | `run_2/`  | lr=2e-6 | 0.3714 | — |
| T10 | `run_2/` | lr=5e-6 | 0.4000 | — |
| T11 | `run_2/` | dropout=0.075 | 0.4286 | — |
| T12 | `run_2/` | dropout=0.1 | 0.4190 | — |
| T13 | `run_2/` | warmup_ratio=0.075 | 0.4381 | — |
| T14 | `run_2/` | warmup_ratio=0.1 | 0.4095 | — |
| T15 | `run_2/` | targets: q, v only | 0.4000 | — |
| T16 | `run_2/` | targets: q,v,k,o,gate,up,down (comprehensive) | 0.5238 | — |
| T17 | `run_2/` | targets: q,v,gate,up — balanced (2.92M params) | 0.4286 | — |
| T18 | `run_2/` | img_size=128, train_batch=2, eval_batch=4 | 0.5143 | — |
| T19 | `run_2/` | img_size=256, train_batch=2, eval_batch=4 | 0.5048 | — |
| T20 | `run_2/` | img_size=384, train_batch=2, eval_batch=4 | 0.5619 | — |
| T21 | `run_2/` | img_size=512, train_batch=2, eval_batch=4 | 0.5429 | — |

> T21: OOM risk during validation; training time ~6 min vs ~4 min standard.  
> Additional config and metrics JSONs for T2–T22 are also stored in `run_2/`.

---

### `run_3/` — Improved Baseline + Full Dataset (T24–T26)

| Trial | Directory | Key Change | Val Acc | Kaggle |
|---|---|---|---|---|
| T24 | `run_3_0/` | Improved baseline: epoch=5, all targets, img_size=384 | 0.6095 | 0.6680 |
| T25 | `run_3_2/` | Direct generation inference — same weights as T24 | 0.6095 | 0.6620 |
| T26 | `run_3_1/` | Full training set (train_frac=1.0) | 0.7815 | 0.8089 |

> T25 uses the `run_3/0/best_checkpoint/` weights with a different inference strategy only — no retraining.  
> T26 used an A100 GPU for inference (completed in ~2 min).

---

### `run_4/` — Frozen Encoder, Prompt Format, Context Ablation (T22 frozen, T23)

| Trial | Directory | Key Change | Val Acc | Kaggle |
|---|---|---|---|---|
| T22 (frozen) | `run_4/` | Frozen vision encoder, epoch=1 | 0.5238 | — |
| T23 | `run_4_2/` | Native chat template + cleaned prompt, epoch=5 | 0.5333 | — |
| Context ablation | `run_4/3/` | Inference only on `run_3/0/` checkpoint | N/A | — |

> T23: very low val loss (0.1349) with no accuracy improvement — rapid overfitting on the 10% subset.  
> Context ablation (`run_4_3/`) contains no trained weights; outputs stored in `context_ablation.json`.

---

### `run_5/` — Full Data, Long Training (T27)

| Trial | Directory | Key Change | Val Acc | Kaggle |
|---|---|---|---|---|
| T27 | `run_5/` | max_seq_len=2048, epoch=10, train_batch=8, eval_batch=16, all targets | 0.8053 | 0.8209 |

> Model still improving at end of epoch 10; ~30 GB GPU RAM used.

---

### `run_7/` — img_size=512, Best HPs (T28)

| Trial | Directory | Key Change | Val Acc | Kaggle |
|---|---|---|---|---|
| T28 | `run_7/` | img_size=512, epoch=10, train_batch=4, eval_batch=16 (T27 HPs) | 0.8330 | 0.8491 |

---

### `run_8/` — Final Retrain on Train+Val Combined (T29)

| Trial | Directory | Key Change | Val Acc | Kaggle |
|---|---|---|---|---|
| T29 | `run_8/` | Retrain from T28 best checkpoint with combined train+val | 0.8330 | 0.8511 |

> Retrains from `run_7/0/best_checkpoint/`. No validation loop during retraining.

---

### `run_9/` — Context at Inference (T30)

| Trial | Directory | Key Change | Val Acc | Kaggle |
|---|---|---|---|---|
| T30 | `run_9/` | Subject-conditional context strategy at inference on T29 weights | N/A | 0.8511 |

> Inference only — no retraining. Uses `run_8/0/final_checkpoint/`.

---

## Key Checkpoints

| Checkpoint | Path | Used by |
|---|---|---|
| Baseline best | `run_0/0/best_checkpoint/` | T0 inference |
| Improved baseline | `run_3/0/best_checkpoint/` | T24, T25, context ablation |
| Full data best | `run_3/1/best_checkpoint/` | T26 submission |
| Long training best | `run_5/0/best_checkpoint/` | T27, seeds T28 |
| 512px best | `run_7/0/best_checkpoint/` | T28, base for T29 |
| Combined train+val | `run_8/0/final_checkpoint/` | T29, T30 inference |

---

## Artefacts Per Run

```
run_X/Y/
├── config.json                      # Hyperparameters
├── train_metrics.json               # Per-epoch train/val loss and accuracy
├── environment.json                 # Library versions for reproducibility
├── submission_YYYYMMDD_HHMMSS.csv   # Kaggle submission file
├── submission_metrics_*.json        # Inference time, answer distribution
├── best_checkpoint/                 # Saved at peak val accuracy
└── final_checkpoint/                # After train+val retraining (runs 8, 9 only)
```

---

# Setup Guide — Paths to Update Before Running

This lists every path and run identifier that must be updated in `baseline.ipynb` before running or reproducing an environment.

---

## Summary of All Variables to Change

| Cell | Variable | Default | Change when |
|---|---|---|---|
| 2 | `RUN_ID` | `"run_ID"` | Starting a new experiment run ID (MANDATORY) |
| 2 | `SUB_RUN_ID` | `"sub trial ID"` | Appending the sub trial ID suffix (MANDATORY) |
| 7 | `DRIVE_DIR_PATH` | `Colab Notebooks/{RUN_ID}/` | Confirm since Drive folder structure might be different (MANDATORY) |
| 7 | `DRIVE_DATA_PATH` | `Colab Notebooks/pixels-to-predictions.zip` | Data zip must be stored on Drive (MANDATORY) |
| 44 | `CUSTOM_RUN_ID` | `"run_7"` | If loading a different best checkpoint for retraining (OPTIONAL) |
| 48 | `CUSTOM_RUN_ID` | `"run_9"` | If loading a different final checkpoint for inference (OPTIONAL) |

---

## Output Locations

All outputs are written to `DRIVE_DIR_PATH` on Google Drive:

```
/content/drive/MyDrive/Colab Notebooks/{RUN_ID}/
├── config.json
├── environment.json
├── train_metrics.json
├── best_checkpoint/
├── final_checkpoint/
├── submission_YYYYMMDD_HHMMSS.csv
└── submission_metrics_YYYYMMDD_HHMMSS.json
```

---

## Step 1 — Set the Run ID (Cell 2) MANDATORY

```python
RUN_ID     = "run_9"          # ← name of the output folder in Google Drive
SUB_RUN_ID = "include_solution"  # ← suffix appended to submission filenames; set to "" to disable
```

`RUN_ID` controls where all outputs (checkpoints, metrics, submission CSVs) are saved under `DRIVE_DIR_PATH`. Change it for each new experiment to avoid overwriting previous runs.

---

## Step 2 — Set Google Drive Paths (Cell 7) MANDATORY

```python
DRIVE_DIR_PATH  = '/content/drive/MyDrive/Colab Notebooks/' + RUN_ID + '/'
DRIVE_DATA_PATH = '/content/drive/MyDrive/Colab Notebooks/pixels-to-predictions.zip'
```

| Variable | Purpose | What to change |
|---|---|---|
| `DRIVE_DIR_PATH` | Output folder for checkpoints, metrics, and submissions | Change the base path if your Drive root is different from `Colab Notebooks/` |
| `DRIVE_DATA_PATH` | Location of the competition zip file on Drive | Update to wherever you uploaded `pixels-to-predictions.zip` |

The data is unzipped to `/content/data/pixels-to-predictions/` on local disk for faster access. This is handled automatically if the directory does not already exist.

---

## Step 3 — Set the Checkpoint Source for Final Retraining (Cell 44) OPTIONAL

```python
CUSTOM_RUN_ID  = "run_7"   # ← run folder containing the best_checkpoint to load
MODEL_DIR_PATH = '/content/drive/MyDrive/Colab Notebooks/' + CUSTOM_RUN_ID + '/'
```

This cell copies `best_checkpoint` and `train_metrics.json` from a **previous run** into the current Colab session before final retraining. Set `CUSTOM_RUN_ID` to the run folder that contains the checkpoint you want to retrain from.

---

## Step 4 — Set the Checkpoint Source for Inference (Cell 48) OPTIONAL

```python
CUSTOM_RUN_ID  = "run_9"   # ← run folder containing the final_checkpoint to load
MODEL_DIR_PATH = '/content/drive/MyDrive/Colab Notebooks/' + CUSTOM_RUN_ID + '/'
```

This cell copies `final_checkpoint` from Drive into the Colab session before inference. Set `CUSTOM_RUN_ID` to the run whose final trained model you want to submit.




