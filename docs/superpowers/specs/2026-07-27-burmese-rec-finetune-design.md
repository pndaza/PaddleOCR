# Fine-tune a PP-OCRv6 recognition model for Burmese

**Date:** 2026-07-27
**Status:** Approved (pre-implementation)
**Working dir:** `/Users/pndaza/Projects/playground/clones/PaddleOCR`

## Goal & Scope

Produce a PP-OCRv6 **small** recognition model (`PPLCNetV4` + `LightSVTR` + CTC/NRTR
MultiHead) fine-tuned for Burmese/Pali, that pairs with the existing PP-OCRv6
detector (`models/PP-OCRv6_small_det_infer`) and exports to a deployable PaddleOCR
inference model.

**Source data:** the existing Kraken arrow dataset at
`/Users/pndaza/Projects/playground/ocr/kraken-burmese/manifests/burmese_kraken_1m_clean.arrow`
— 897,883 line crops (train 718,515 / val 89,625 / test 89,743), 1-bit PNG,
height 48px, with a 118-char alphabet embedded in the file's schema metadata.

**In scope:** data extraction (arrow → PaddleOCR format), Burmese dict, config,
training (smoke + full), eval on held-out test split, export to inference model.

**Out of scope:** retraining detection, changing the Kraken pipeline, building a UI,
changing the existing detection scripts (`detect_baselines.py`, `polygonize_kraken.py`,
etc.).

### Pipeline overview

```
burmese_kraken_1m_clean.arrow
        │
        ▼  scripts/build_paddle_dataset.py  (extract PNGs + rec_gt_*.txt)
train_data/burmese_rec/
   ├── burmese_dict.txt        (118 chars, one per line)
   ├── train_list.txt          path<TAB>text
   ├── train_list_smoke.txt    (subset for smoke run)
   ├── val_list.txt
   ├── val_list_smoke.txt
   ├── test_list.txt
   ├── train/                  PNG line crops
   └── val/
        │
        ▼  python tools/train.py -c configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml
   output/burmese_PP-OCRv6_small_rec/best_accuracy.pdparams
        │
        ▼  python tools/export_model.py ...
   models/burmese_PP-OCRv6_small_rec_infer/   (deliverable: inference.pdmodel + .pdiparams)
```

## Key data findings (drove the config decisions)

Measured from the arrow file (every 2000th row, n=449):

| Metric | Value | Implication |
|---|---|---|
| Text length: median | 53 chars | Stock `max_text_length: 25` drops **67%** of data |
| Text length: p90 / p99 / max | 83 / 93 / 104 chars | `max_text_length: 100` keeps ~99.7% |
| Image width: median | 557px | Stock width `320` distorts 67% of crops |
| Image width: p90 / p95 | 1035 / 1158 px | Width `640` keeps ~88% undistorted |
| Image height | 48px (all rows) | Matches PP-OCRv6 `[3,48,W]` exactly — no vertical resampling |
| Alphabet (from metadata) | 118 chars | No Burmese chars exist in shipped `ppocrv6_dict.txt` (18,708 chars); custom dict mandatory |

## Section 1 — Data extraction (arrow → PaddleOCR format)

New script: `scripts/build_paddle_dataset.py`.

**Outputs** (under `train_data/burmese_rec/`):

```
train_data/burmese_rec/
├── burmese_dict.txt        # 118 chars, one per line, sorted by Unicode codepoint
├── train_list.txt          # train/000000001.png\t<text>
├── train_list_smoke.txt    # first N train rows (for smoke run)
├── val_list.txt
├── val_list_smoke.txt
├── test_list.txt           # held-out test split (not used in training)
├── train/000000001.png ... # 1-bit PNG written as grayscale (L mode)
└── val/000000001.png ...
```

**Design decisions:**

1. **Splits come from the arrow's `train`/`validation`/`test` boolean columns** — not
   random shuffling. Keeps the held-out test set identical to the one Kraken models
   were measured against, so CER is directly comparable.
2. **Image mode: `1` (bi-level) → `L` (grayscale) on write.** PaddleOCR's
   `DecodeImage` works with 1-bit but grayscale is safer and matches how the eval
   pipeline extracts crops.
3. **Dict extracted from arrow metadata** — the 118-char alphabet is in the schema
   metadata under the `lines` key as JSON (`{"alphabet": {"က": 33674, ...}}`).
   Script sorts by Unicode codepoint and writes one char per line. No manual
   dict authoring.
4. **Subset support via `--max-per-split N` flag.** First run extracts e.g.
   50K train / 5K val for the smoke test; later run extracts the full set. Same
   script, same outputs, just a cap. Produces both `<split>_list.txt` (full) and
   `<split>_list_smoke.txt` (subset).
5. **Filename = arrow row order**, zero-padded `000000001` (matches the
   convention in `burmese_ocr_1m_README.md`). Deterministic and re-runnable.

**Built-in validation:** after writing, the script re-reads `train_list.txt`,
confirms every listed file exists and every transcription char is in
`burmese_dict.txt`, and prints a coverage report. Any OOV char → error (CTC
cannot emit it).

**Disk footprint:** full extraction ~1.5GB PNGs; smoke subset ~100MB for 50K.

## Section 2 — Config (clone of PP-OCRv6 small rec, tuned for Burmese)

New file: `configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml`. Cloned from
`configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml` with these changes:

| Field | Stock v6 small | Burmese | Why |
|---|---|---|---|
| `character_dict_path` | `ppocr/utils/dict/ppocrv6_dict.txt` (18,708 chars, no Burmese) | `train_data/burmese_rec/burmese_dict.txt` (118 chars) | Custom dict mandatory; head FC auto-resizes to 118 + blank + space = 120. |
| `max_text_length` (anchor `&max_text_length`) | `25` | `100` | Stock value drops 67% of data via `NRTRLabelEncode`; `100` keeps ~99.7%. |
| `d2s_train_image_shape` | `[3, 48, 320]` | `[3, 48, 640]` | Stock width distorts 67% of crops; 640 keeps ~88% undistorted. |
| `Eval ... RecResizeImg.image_shape` | `[3, 48, 320]` | `[3, 48, 640]` | Match train. |
| `Train.sampler.scales` | `[[320,32],[320,48],[320,64]]` | `[[640,48],[640,64],[640,80]]` | Width 640; min height 48 (data is all 48px — drop the 32 scale). |
| `RecConAug.image_shape` | `[48, 320, 3]` | `[48, 640, 3]` | Keep augmentation canvas consistent. |
| `Optimizer.lr.learning_rate` | `0.0005` | `0.0001` | Stock is from-scratch 8-GPU (batch 1024); 5× lower for single-GPU fine-tune. |
| `Global.epoch_num` | `100` | `5` (smoke) / `30` (full), via CLI | Smoke validates fast; full gives cosine schedule room. |
| `Global.pretrained_model` | (empty) | `./PP-OCRv6_small_rec_pretrained` | Download `PP-OCRv6_small_rec_pretrained.pdparams` from official URL. |
| `save_model_dir` | `./output/PP-OCRv6_small_rec` | `./output/burmese_PP-OCRv6_small_rec` | Keep separate from stock run. |

**Unchanged (intentionally):** backbone (`PPLCNetV4 small`), head (`MultiHead`
CTC+NRTR), loss (`MultiLoss`), optimizer (`Adam` + Cosine, warmup 5),
`RecConAug`/`RecAug` augmentations, `MultiScaleSampler`, `use_space_char: true`.

**Pretrained checkpoint URL:**
`https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams`
(found in `docs/version3.x/module_usage/text_recognition.en.md`).

**Expected behavior at iter 0:** accuracy = 0. The head FC reinitializes because
the vocab changed (18,708 → 118). The pretrained backbone+neck weights load and
accelerate convergence. This is expected and documented in `finetune.en.md`.

**Open knob (decide after smoke test):** whether to widen image further. At width
640, ~12% of crops still exceed it and compress. Could go 800 (keeps 95%) at ~25%
more memory/compute. Revisit after the smoke test measures actual GPU memory
headroom.

## Section 3 — Training & export workflow

New orchestrator: `scripts/train_burmese.sh` with `smoke` / `full` modes.

### Smoke run (validate pipeline, ~30–60 min, subset data)

```bash
python tools/train.py \
  -c configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml \
  -o Global.epoch_num=1 \
     Train.dataset.label_file_list=["./train_data/burmese_rec/train_list_smoke.txt"] \
     Eval.dataset.label_file_list=["./train_data/burmese_rec/val_list_smoke.txt"]
```

**Success criteria (all must pass before scaling up):**

1. No crash through ≥2000 iterations (gets past `eval_batch_step: [0, 2000]` → first eval fires).
2. Loss decreases — both `CTCLoss` and `NRTRLoss` trending down, not NaN/inf.
3. First eval completes and reports non-zero acc (even 5–20% confirms head is learning).
4. GPU memory headroom noted — this is where we decide whether to widen to 800.
5. Dict/head wiring verified — `out_channels` in the log equals `118 + 1 (blank) + 1 (space) = 120`.

### Full run (production model, single GPU, multi-hour)

```bash
python tools/train.py \
  -c configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml \
  -o Global.epoch_num=30 \
     Train.dataset.label_file_list=["./train_data/burmese_rec/train_list.txt"] \
     Eval.dataset.label_file_list=["./train_data/burmese_rec/val_list.txt"]
```

PaddleOCR autosaves `best_accuracy.pdparams` (highest val acc) and periodic
checkpoints to `output/burmese_PP-OCRv6_small_rec/`. Cosine LR over 30 epochs
with 5-epoch warmup gives a smooth decay tail.

### Export (checkpoint → deployable inference model)

```bash
python tools/export_model.py \
  -c configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml \
  -o Global.checkpoints=./output/burmese_PP-OCRv6_small_rec/best_accuracy.pdparams \
     Global.save_inference_dir=./models/burmese_PP-OCRv6_small_rec_infer
```

**Deliverable:** `models/burmese_PP-OCRv6_small_rec_infer/` containing
`inference.pdmodel` + `inference.pdiparams` — drop-in compatible with the
existing detection pipeline.

### Final quality check (held-out test split)

```bash
python tools/eval.py \
  -c configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml \
  -o Eval.dataset.label_file_list=["./train_data/burmese_rec/test_list.txt"] \
     Global.checkpoints=./output/burmese_PP-OCRv6_small_rec/best_accuracy.pdparams
```

Reports line-level accuracy and CER on the test split (same images Kraken models
were measured on — directly comparable).

## Deliverables

| Artifact | Path |
|---|---|
| Dataset extraction script | `scripts/build_paddle_dataset.py` |
| Burmese dict | `train_data/burmese_rec/burmese_dict.txt` |
| Label files | `train_data/burmese_rec/{train,val,test}_list.txt` (+ `_smoke`) |
| Line crops | `train_data/burmese_rec/{train,val}/*.png` |
| Config | `configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml` |
| Training orchestrator | `scripts/train_burmese.sh` |
| Best checkpoint | `output/burmese_PP-OCRv6_small_rec/best_accuracy.pdparams` |
| **Inference model (final)** | `models/burmese_PP-OCRv6_small_rec_infer/` |

## Risks & mitigations

- **Iter-0 acc = 0.** Expected (vocab mismatch reinitializes head FC). Not a bug.
- **Width 640 may still compress ~12% of long lines.** Mitigation: revisit after
  smoke test measures GPU memory; bump to 800 if headroom allows.
- **Long Burmese lines near `max_text_length: 100`.** Only 0.7% of lines exceed
  100 chars and are dropped — acceptable.
- **Pretrained checkpoint availability.** URL verified in
  `docs/version3.x/module_usage/text_recognition.en.md`. If download fails, fall
  back to training from scratch (slower convergence) — fallback noted in the plan.
