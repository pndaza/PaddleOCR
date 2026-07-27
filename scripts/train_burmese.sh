#!/usr/bin/env bash
# Fine-tune PP-OCRv6 small rec for Burmese.
#
# Usage:
#   bash scripts/train_burmese.sh smoke       # 1 epoch, subset — GPU pipeline check
#   bash scripts/train_burmese.sh cpu-smoke   # 1 epoch, subset, use_gpu:false — CPU check
#   bash scripts/train_burmese.sh cpu-train   # N epochs on subset, use_gpu:false — CPU training
#   bash scripts/train_burmese.sh full        # 30 epochs, full data — GPU production
#   bash scripts/train_burmese.sh export      # checkpoint -> inference model
#   bash scripts/train_burmese.sh eval        # final CER on held-out test split
#
# Run from repo root. Uses .venv/bin/python (paddle 3.3.1, CPU-only wheel on macOS).
set -euo pipefail

# Resolve python: prefer the repo's .venv (local dev), fall back to system python
# (Colab VM has no .venv — python is at /usr/local/bin/python).
if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
else
  PY=python
fi
CFG=configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml
OUT=output/burmese_PP-OCRv6_small_rec

case "${1:-}" in
  smoke)
    "$PY" tools/train.py -c "$CFG" \
      -o Global.epoch_num=1 \
         Train.dataset.label_file_list=["./train_data/burmese_rec/train_list_smoke.txt"] \
         Eval.dataset.label_file_list=["./train_data/burmese_rec/val_list_smoke.txt"]
    ;;
  cpu-smoke)
    # CPU pipeline check: use_gpu:false, smaller batch (128 won't fit in CPU RAM at 640 wide).
    "$PY" tools/train.py -c "$CFG" \
      -o Global.use_gpu=false \
         Global.epoch_num=1 \
         Train.loader.batch_size_per_card=32 \
         Eval.loader.batch_size_per_card=32 \
         Train.sampler.first_bs=32 \
         Train.dataset.label_file_list=["./train_data/burmese_rec/train_list_smoke.txt"] \
         Eval.dataset.label_file_list=["./train_data/burmese_rec/val_list_smoke.txt"]
    ;;
  cpu-train)
    # CPU training on the capped subset (50K train). Set EPOCHS env to override.
    : "${EPOCHS:=10}"
    "$PY" tools/train.py -c "$CFG" \
      -o Global.use_gpu=false \
         Global.epoch_num="$EPOCHS" \
         Train.loader.batch_size_per_card=32 \
         Eval.loader.batch_size_per_card=32 \
         Train.sampler.first_bs=32 \
         Train.dataset.label_file_list=["./train_data/burmese_rec/train_list.txt"] \
         Eval.dataset.label_file_list=["./train_data/burmese_rec/val_list.txt"]
    ;;
  full)
    "$PY" tools/train.py -c "$CFG"
    ;;
  export)
    "$PY" tools/export_model.py -c "$CFG" \
      -o Global.checkpoints=./${OUT}/best_accuracy.pdparams \
         Global.save_inference_dir=./models/burmese_PP-OCRv6_small_rec_infer
    ;;
  eval)
    "$PY" tools/eval.py -c "$CFG" \
      -o Eval.dataset.label_file_list=["./train_data/burmese_rec/test_list.txt"] \
         Global.checkpoints=./${OUT}/best_accuracy.pdparams
    ;;
  *)
    echo "Usage: bash scripts/train_burmese.sh {smoke|cpu-smoke|cpu-train|full|export|eval}" >&2
    echo "  cpu-train accepts EPOCHS env var (default 10)" >&2
    exit 1
    ;;
esac
