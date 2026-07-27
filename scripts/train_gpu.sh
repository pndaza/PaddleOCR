#!/usr/bin/env bash
#
# train_gpu.sh — Set up and run Burmese PP-OCRv6 rec training on a rented GPU box
# (vast.ai or any SSH-accessible Linux server with an NVIDIA GPU).
#
# Vast.ai notes applied (from kraken-burmese/docs/cloud_training_notes.md):
#   - Ubuntu 24.04 images block pip → use --break-system-packages
#   - Use tmux for the long training run (live progress, survives disconnects)
#   - Dropbox direct links need ?dl=1 / &dl=1
#
# Unlike train_colab.sh, this runs as a NORMAL synchronous bash script on the
# box itself — no colab CLI, no Jupyter boundary, no sentinel polling. Exit
# codes work normally. Just ssh in and run it.
#
# ---------------------------------------------------------------------------
# USAGE (run ON the GPU box, after ssh'ing in):
#
#   bash /workspace/PaddleOCR/scripts/train_gpu.sh setup    # one-time env setup
#   bash /workspace/PaddleOCR/scripts/train_gpu.sh train    # full training (in tmux)
#   bash /workspace/PaddleOCR/scripts/train_gpu.sh export   # checkpoint → inference model
#   bash /workspace/PaddleOCR/scripts/train_gpu.sh eval     # final test-split CER
#   bash /workspace/PaddleOCR/scripts/train_gpu.sh all      # setup → train → export → eval
#
# Run each phase separately so you can inspect between them, or use 'all'.
# The 'train' phase launches in tmux so it survives SSH disconnects.
#
# Env vars (override on the CLI, e.g. EPOCHS=10 bash train_gpu.sh train):
#   EPOCHS (30), BATCH_SIZE (64 — RTX 5060 has 16GB; see note below),
#   FORK_URL, FORK_BRANCH,
#   PADDLE_INDEX, PADDLE_PKG, PADDLE_WHEEL_URL (empty = use Baidu index),
#   DATASET_ZIP_URL.

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
WORKDIR="${WORKDIR:-/workspace}"
REPO="${REPO:-${WORKDIR}/PaddleOCR}"
FORK_URL="${FORK_URL:-https://github.com/pndaza/PaddleOCR.git}"
FORK_BRANCH="${FORK_BRANCH:-burmese-rec-finetune}"

# Paddle GPU install source. Two options:
#   1. (default) PADDLE_INDEX — Baidu's PEP 503 simple index. pip resolves the
#      correct wheel for whatever Python the box runs (cp39/310/311/312/313).
#      Direct .whl URLs on Baidu's CDN return empty bodies, so the index route
#      is required. Paddle 3.x is NOT on PyPI (only 2.6.x).
#   2. PADDLE_WHEEL_URL — a direct .whl URL (e.g. a Dropbox mirror). Use ONLY if
#      it matches the box's Python tag exactly (cp312 wheel won't install on 3.10).
PADDLE_INDEX="${PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"
PADDLE_PKG="${PADDLE_PKG:-paddlepaddle-gpu==3.2.0}"
PADDLE_WHEEL_URL="${PADDLE_WHEEL_URL:-}"

# Pre-built dataset zip (Dropbox mirror of train_data/burmese_rec/, 1.3 GB).
DATASET_ZIP_URL="${DATASET_ZIP_URL:-https://www.dropbox.com/scl/fi/ibvp1cjeoi6jlnjp34icm/burmese_rec_dataset.zip?rlkey=00iknrubirk4ubcfayiji8bn6&st=v3175nwf&dl=1}"

# Training knobs.
EPOCHS="${EPOCHS:-30}"
# RTX 5060 has 16 GB VRAM. At width 640 (Burmese lines are wide), batch 128
# risks OOM. Start at 64; bump to 96/128 if headroom allows. MultiScaleSampler
# with fix_bs:false self-throttles taller heights (actual ~64/48/38).
BATCH_SIZE="${BATCH_SIZE:-64}"

CFG="${REPO}/configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml"
OUT="${REPO}/output/burmese_PP-OCRv6_small_rec"

# Detect pip flag for Ubuntu 24.04 (externally-managed-environment restriction).
PIP_INSTALL=(python3 -m pip install)
if python3 -c "import sysconfig, os; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_path('stdlib'), 'EXTERNALLY-MANAGED')) else 1)" 2>/dev/null; then
  PIP_INSTALL+=(--break-system-packages)
fi

# Ensure pip exists (minimal vast.ai images ship neither pip nor ensurepip).
ensure_pip() {
  if python3 -m pip --version >/dev/null 2>&1; then return 0; fi
  log "pip not found — bootstrapping via get-pip.py"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  python3 /tmp/get-pip.py
  rm -f /tmp/get-pip.py
  python3 -m pip --version >/dev/null 2>&1 || { log "ERROR: pip bootstrap failed"; exit 1; }
  log "pip bootstrapped: $(python3 -m pip --version)"
}

log() { printf '\n[setup] %s\n' "$*"; }

# Resolve a python interpreter. Minimal images ship only `python3`, not `python`;
# venvs ship `.venv/bin/python`. Prefer a venv if present, else fall back.
if [[ -x "${REPO}/.venv/bin/python" ]]; then PY="${REPO}/.venv/bin/python";
elif command -v python >/dev/null 2>&1; then PY=python;
else PY=python3; fi

# ---------------------------------------------------------------------------
# PHASE: setup — clone repo, install paddle + deps, fetch data + weights
# ---------------------------------------------------------------------------
do_setup() {
  cd "$WORKDIR"

  # 1. Clone the fork (skip if already present).
  if [[ ! -d "$REPO/.git" ]]; then
    log "cloning fork ${FORK_BRANCH} from ${FORK_URL}"
    git clone --branch "$FORK_BRANCH" --depth 1 "$FORK_URL" "$REPO"
    # Remove the remote so no push to fork/upstream is possible from this box.
    git -C "$REPO" remote remove origin || true
    log "removed 'origin' remote (push safeguard)"
  else
    log "repo already cloned at $REPO"
  fi

  # 2. Install paddle (skip if a CUDA-built paddle already imports).
  log "checking paddle"
  if python3 -c "import paddle; assert paddle.device.is_compiled_with_cuda()" 2>/dev/null; then
    log "paddle $(python3 -c 'import paddle; print(paddle.__version__)') already CUDA-built — skipping install"
  else
    ensure_pip
    if [[ -n "$PADDLE_WHEEL_URL" ]]; then
      log "installing paddle from wheel: $PADDLE_WHEEL_URL"
      # Match the wheel's Python tag to avoid silent cp-mismatch failures.
      wheel="/tmp/paddlepaddle_gpu.whl"
      curl -L --fail -C - -o "$wheel" "$PADDLE_WHEEL_URL"
      "${PIP_INSTALL[@]}" "$wheel"
    else
      log "installing ${PADDLE_PKG} from Baidu index ${PADDLE_INDEX} (pip picks the cp tag for this Python)"
      "${PIP_INSTALL[@]}" --timeout 300 --retries 5 \
        -i "$PADDLE_INDEX" "$PADDLE_PKG"
    fi
    python3 -c "import paddle; assert paddle.device.is_compiled_with_cuda(), 'paddle NOT compiled with CUDA — wrong wheel'"
    log "OK: paddle $(python3 -c 'import paddle; print(paddle.__version__)') installed, CUDA compiled"
  fi

  # 3. Install PaddleOCR repo requirements.
  log "installing repo requirements.txt"
  ensure_pip
  "${PIP_INSTALL[@]}" -r "$REPO/requirements.txt"

  # 3b. System libs that opencv (cv2) needs at import time. Minimal images lack
  # these; without them `import albumentations` → ImportError: libGL/libxcb.
  if ! python3 -c "import cv2" 2>/dev/null; then
    log "cv2 won't import — installing system libs (libgl1, libglib2.0-0, libxcb1)"
    apt-get update -qq
    apt-get install -y -qq libgl1 libglib2.0-0 libxcb1
  fi

  # 4. Download the pre-built dataset zip + extract (Python zipfile — no `unzip` dep).
  if [[ -f "$REPO/train_data/burmese_rec/burmese_dict.txt" ]]; then
    log "dataset already present at $REPO/train_data/burmese_rec/ — skipping"
  else
    log "downloading dataset zip from Dropbox (~1.3 GB)"
    zip="/tmp/burmese_rec_dataset.zip"
    curl -L --fail -C - -o "$zip" "$DATASET_ZIP_URL"
    mkdir -p "$REPO/train_data"
    log "extracting dataset (python zipfile)"
    # Minimal vast.ai images have no `unzip`; use Python's stdlib instead.
    python3 - "$zip" "$REPO/train_data" <<'PY'
import sys, zipfile, os
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(src) as z:
    z.extractall(dst)
print(f"extracted {len(os.listdir(dst))} top-level entries to {dst}")
PY
    rm -f "$zip"
    # Verify key outputs.
    for f in burmese_dict.txt train_list.txt val_list.txt test_list.txt; do
      [[ -f "$REPO/train_data/burmese_rec/$f" ]] || { log "ERROR: $f missing after unzip"; exit 1; }
    done
    n_train=$(ls "$REPO/train_data/burmese_rec/train" | wc -l)
    log "OK: dataset ready ($n_train train crops)"
  fi

  # 5. Download pretrained weights.
  pretrain="$REPO/PP-OCRv6_small_rec_pretrained.pdparams"
  if [[ -f "$pretrain" ]]; then
    log "pretrained weights already present"
  else
    log "downloading pretrained weights (PP-OCRv6 small rec, 119 MB)"
    curl -L --fail -o "$pretrain" \
      "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams"
  fi

  # 6. Verify GPU is visible.
  log "GPU check"
  python3 -c "
import paddle
n = paddle.device.cuda.device_count()
print(f'paddle CUDA device count: {n}, version: {paddle.version.cuda()}')
assert n >= 1, 'no CUDA GPU visible to paddle'
print('OK: GPU available')
"

  log "SETUP COMPLETE. Next: bash $0 train   (or 'all' to continue automatically)"
}

# ---------------------------------------------------------------------------
# PHASE: train — full training, launched in tmux so it survives SSH disconnects
# ---------------------------------------------------------------------------
do_train() {
  cd "$REPO"
  log "launching training in tmux session 'train' (survives SSH disconnects)"
  log "  epochs: ${EPOCHS}, batch: ${BATCH_SIZE}, cfg: $(basename "$CFG")"
  log "  to watch live:    tmux attach -t train   (Ctrl-B then D to detach)"
  log "  to tail the log:  tail -f ${OUT}/train.log"
  log "  to check GPU:     nvidia-smi"

  # Build the training command (runs inside tmux, logs to file as a backup).
  # Note: --use_gpu is implicit from the config; we just override epochs + batch.
  tmux new-session -d -s train "cd '$REPO' && \
    '$PY' tools/train.py \
      -c '$CFG' \
      -o Global.epoch_num=${EPOCHS} \
         Train.loader.batch_size_per_card=${BATCH_SIZE} \
         Eval.loader.batch_size_per_card=${BATCH_SIZE} \
         Train.sampler.first_bs=${BATCH_SIZE} \
    2>&1 | tee '${OUT}/train.log'"
  log "training launched in tmux. Detach with Ctrl-B then D."
  log "when training finishes, run: bash $0 export"
}

# ---------------------------------------------------------------------------
# PHASE: export — checkpoint → deployable inference model
# ---------------------------------------------------------------------------
do_export() {
  cd "$REPO"
  local ckpt="${OUT}/best_accuracy.pdparams"
  [[ -f "$ckpt" ]] || { log "ERROR: checkpoint $ckpt not found — did training finish?"; exit 1; }
  log "exporting inference model from $ckpt"
  "$PY" tools/export_model.py \
    -c "$CFG" \
    -o Global.checkpoints="$ckpt" \
       Global.save_inference_dir="${REPO}/models/burmese_PP-OCRv6_small_rec_infer"
  # Verify outputs.
  for f in inference.pdmodel inference.pdiparams; do
    [[ -f "${REPO}/models/burmese_PP-OCRv6_small_rec_infer/$f" ]] \
      || { log "ERROR: $f missing after export"; exit 1; }
  done
  log "OK: inference model exported to ${REPO}/models/burmese_PP-OCRv6_small_rec_infer/"
}

# ---------------------------------------------------------------------------
# PHASE: eval — final CER on the held-out test split (comparable to Kraken)
# ---------------------------------------------------------------------------
do_eval() {
  cd "$REPO"
  local ckpt="${OUT}/best_accuracy.pdparams"
  [[ -f "$ckpt" ]] || { log "ERROR: checkpoint $ckpt not found"; exit 1; }
  log "running final eval on held-out test split"
  "$PY" tools/eval.py \
    -c "$CFG" \
    -o 'Eval.dataset.label_file_list=["./train_data/burmese_rec/test_list.txt"]' \
       Global.checkpoints="$ckpt"
  log "eval complete (see acc + norm_edit_dis above)"
}

# ---------------------------------------------------------------------------
# DISPATCH
# ---------------------------------------------------------------------------
case "${1:-}" in
  setup)  do_setup ;;
  train)  do_train ;;
  export) do_export ;;
  eval)   do_eval ;;
  all)
    do_setup
    do_train
    log "training is running in tmux. When it finishes, run:"
    log "  bash $0 export && bash $0 eval"
    log "(can't auto-continue past training since it's in tmux — re-run export/eval after.)"
    ;;
  *)
    cat >&2 <<EOF
Usage: bash $0 {setup|train|export|eval|all}

  setup   one-time env setup (clone, paddle, deps, data, weights, GPU check)
  train   launch full training in tmux (survives SSH disconnects)
  export  checkpoint → deployable inference model
  eval    final CER on held-out test split
  all     setup + train (then manually run export + eval after training)

Env vars: EPOCHS (30), BATCH_SIZE (64), FORK_URL, FORK_BRANCH,
          PADDLE_INDEX, PADDLE_PKG, PADDLE_WHEEL_URL, DATASET_ZIP_URL,
          WORKDIR, REPO
EOF
    exit 1
    ;;
esac
