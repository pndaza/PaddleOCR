#!/usr/bin/env bash
# Fine-tune PP-OCRv6 small rec for Burmese.
#
# Usage:
#   bash scripts/train_burmese.sh smoke     # 1 epoch, subset — validates pipeline
#   bash scripts/train_burmese.sh full      # 30 epochs, full data — production
#   bash scripts/train_burmese.sh export    # checkpoint -> inference model
#   bash scripts/train_burmese.sh eval      # final CER on held-out test split
#
# Run from repo root. Uses .venv/bin/python (paddle 3.3.1).
set -euo pipefail

PY=.venv/bin/python
CFG=configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml
OUT=output/burmese_PP-OCRv6_small_rec

case "${1:-}" in
  smoke)
    "$PY" tools/train.py -c "$CFG" \
      -o Global.epoch_num=1 \
         Train.dataset.label_file_list=["./train_data/burmese_rec/train_list_smoke.txt"] \
         Eval.dataset.label_file_list=["./train_data/burmese_rec/val_list_smoke.txt"]
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
    echo "Usage: bash scripts/train_burmese.sh {smoke|full|export|eval}" >&2
    exit 1
    ;;
esac
