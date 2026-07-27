#!/usr/bin/env bash
#
# train_colab.sh — Fine-tune the Burmese PP-OCRv6 rec model on a free Colab T4.
#
# Orchestrates the full lifecycle via the official google-colab-cli:
#   provision T4 → clone fork → install deps → download data → extract →
#   smoke-gate → train → export → eval → zip → download → stop
#
# Data comes from HuggingFace; code comes from your GitHub fork. Nothing is
# pushed from the VM (the upstream PaddlePaddle/PaddleOCR is unreachable:
# the cloned repo's only remote is removed right after clone).
#
# ---------------------------------------------------------------------------
# PREREQUISITES (one-time setup)
# ---------------------------------------------------------------------------
#   1. Push the branch to your fork (the VM clones it):
#        git remote add fork https://github.com/<you>/PaddleOCR.git
#        git push fork burmese-rec-finetune
#   2. Colab CLI installed + authenticated (the correction repo's venv has it):
#        pip install google-colab-cli && colab auth
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   bash scripts/train_colab.sh                       # all defaults (30 epochs, full data)
#   EPOCHS=5 bash scripts/train_colab.sh              # fewer epochs
#   MAX_PER_SPLIT=50000 bash scripts/train_colab.sh   # subset run
#   SMOKE_FIRST=0 bash scripts/train_colab.sh         # skip the smoke gate
#   KEEP_VM=1 bash scripts/train_colab.sh             # don't tear down on exit (debug)
#   BATCH_SIZE=96 bash scripts/train_colab.sh         # push batch if T4 has headroom
#
# All knobs are env vars (see CONFIGURATION below).

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIGURATION (override via env vars)
# ---------------------------------------------------------------------------
SESSION="${SESSION:-burmese-rec-train}"
GPU="${GPU:-T4}"
FORK_URL="${FORK_URL:-https://github.com/pndaza/PaddleOCR.git}"
FORK_BRANCH="${FORK_BRANCH:-burmese-rec-finetune}"

# Data: HuggingFace dataset + file. The published arrow is the 80/20 clean+degraded
# version (897K rows, 118-char alphabet) — same schema as the local _clean variant,
# so build_paddle_dataset.py runs unchanged.
HF_DATASET="${HF_DATASET:-pndaza/burmese-kraken-1m}"
HF_FILE="${HF_FILE:-burmese_kraken_1m.arrow}"

# Training knobs.
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"        # T4 16GB VRAM safety at width 640 (stock 128 risks OOM)
MAX_PER_SPLIT="${MAX_PER_SPLIT:-}"    # empty = full 718K; set N for a subset run
SMOKE_FIRST="${SMOKE_FIRST:-1}"       # 1 = gate full training behind a passing 5K smoke
KEEP_VM="${KEEP_VM:-0}"               # 1 = leave session running after exit (debug)
DRIVE_BACKUP="${DRIVE_BACKUP:-1}"     # 1 = mount Drive and copy the output zip there

# Pretrained rec weights (official PP-OCRv6 small rec).
PRETRAIN_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams"

# Paddle GPU wheel — Baidu cu126 index (bundles CUDA 12.6 + cuDNN; works on Colab
# regardless of host CUDA, per docs/version3.x/paddlepaddle_installation.en.md).
PADDLE_PKG="paddlepaddle-gpu==3.2.0"
PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"

# Derived paths (VM-local).
VM_REPO="/content/PaddleOCR"
VM_DATA="/content/data"
VM_ARROW="${VM_DATA}/${HF_FILE}"
VM_PRETRAIN="/content/PP-OCRv6_small_rec_pretrained"   # path WITHOUT .pdparams (paddle convention)
VM_INFER="${VM_REPO}/models/burmese_PP-OCRv6_small_rec_infer"
RUN_ID="colab-run-$(date +%Y%m%d-%H%M%S)"
LOCAL_DIR="./models/${RUN_ID}"
DRIVE_BASE="${DRIVE_BASE:-/content/drive/MyDrive/burmese-paddleocr}"
REMOTE_ZIP="${DRIVE_BASE}/${RUN_ID}.zip"

# ---------------------------------------------------------------------------
# SAFETY: always tear down the VM on exit (unless KEEP_VM=1) so compute units
# aren't burned on an idle session.
# ---------------------------------------------------------------------------
COLAB=""  # resolved in preflight
cleanup() {
  local exit_code=$?
  if [[ "$KEEP_VM" == "1" ]]; then
    echo ""
    echo "[script] KEEP_VM=1: leaving session '${SESSION}' running (exit_code=${exit_code})."
    echo "[script] Inspect: ${COLAB} status -s ${SESSION}   then: ${COLAB} stop -s ${SESSION}"
    return
  fi
  echo ""
  echo "[script] tearing down session '${SESSION}' (exit_code=${exit_code})"
  "$COLAB" stop -s "$SESSION" >/dev/null 2>&1 || true
}
trap cleanup EXIT

log() { printf '\n[script] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# PREFLIGHT
# ---------------------------------------------------------------------------
log "preflight checks"

# Resolve the colab binary: explicit path → known venv → PATH.
if [[ -n "${COLAB_BIN:-}" ]] && [[ -x "$COLAB_BIN" ]]; then
  COLAB="$COLAB_BIN"
elif [[ -x /Users/pndaza/Projects/playground/ocr/burmese-ocr-correction/.venv/bin/colab ]]; then
  COLAB=/Users/pndaza/Projects/playground/ocr/burmese-ocr-correction/.venv/bin/colab
elif command -v colab >/dev/null 2>&1; then
  COLAB="$(command -v colab)"
else
  echo "ERROR: 'colab' CLI not found." >&2
  echo "       Install with:  pip install google-colab-cli" >&2
  echo "       (or set COLAB_BIN to point at it)" >&2
  exit 1
fi
echo "[script] using colab: ${COLAB} ($("${COLAB}" version 2>&1 | head -1))"

log "configuration:"
echo "    session:       ${SESSION}"
echo "    gpu:           ${GPU}"
echo "    fork:          ${FORK_URL} @ ${FORK_BRANCH}"
echo "    hf dataset:    ${HF_DATASET} / ${HF_FILE}"
echo "    epochs:        ${EPOCHS}"
echo "    batch_size:    ${BATCH_SIZE} (T4-safe at width 640)"
echo "    max_per_split: ${MAX_PER_SPLIT:-<full>}"
echo "    smoke_first:   ${SMOKE_FIRST}"
echo "    keep_vm:       ${KEEP_VM}"
echo "    drive_backup:  ${DRIVE_BACKUP}"
echo "    local output:  ${LOCAL_DIR}"

# ---------------------------------------------------------------------------
# 1. PROVISION THE VM
# ---------------------------------------------------------------------------
log "provisioning ${GPU} session '${SESSION}' (this can take ~30s)"
"$COLAB" new -s "$SESSION" --gpu "$GPU"

# ---------------------------------------------------------------------------
# 2. CLONE THE FORK + REMOVE ITS REMOTE (upstream-push safeguard)
# ---------------------------------------------------------------------------
log "cloning fork ${FORK_BRANCH} from ${FORK_URL}"
"$COLAB" exec -s "$SESSION" --timeout 300 <<EOF
import subprocess, sys
r = subprocess.run(
    ["git", "clone", "--branch", "${FORK_BRANCH}", "--depth", "1",
     "${FORK_URL}", "${VM_REPO}"],
    capture_output=True, text=True)
print(r.stdout, end="")
if r.returncode != 0:
    print(r.stderr, file=sys.stderr)
    raise SystemExit(r.returncode)
# Remove the only remote so NO push (to fork or upstream) is possible from the VM.
sub = subprocess.run(["git", "-C", "${VM_REPO}", "remote", "remove", "origin"],
                     capture_output=True, text=True)
print("removed 'origin' remote (upstream-push safeguard):", sub.stdout.strip() or sub.stderr.strip())
# Sanity: no remotes remain.
chk = subprocess.run(["git", "-C", "${VM_REPO}", "remote", "-v"],
                     capture_output=True, text=True)
print("remaining remotes:", repr(chk.stdout.strip()) or "(none)")
EOF

# ---------------------------------------------------------------------------
# 3. INSTALL PADDLE (GPU) + REPO REQUIREMENTS
# ---------------------------------------------------------------------------
log "installing ${PADDLE_PKG} from Baidu cu126 index (this can take a few minutes)"
"$COLAB" install -s "$SESSION" "${PADDLE_PKG}" -i "${PADDLE_INDEX}" --extra-index-url "${PADDLE_INDEX}"

log "installing repo requirements.txt"
"$COLAB" install -s "$SESSION" -r "${VM_REPO}/requirements.txt"

# ---------------------------------------------------------------------------
# 4. DOWNLOAD DATA (HuggingFace) + PRETRAINED WEIGHTS
# ---------------------------------------------------------------------------
log "downloading ${HF_FILE} from HuggingFace ${HF_DATASET} (~1.1 GB)"
"$COLAB" exec -s "$SESSION" --timeout 600 <<EOF
import subprocess, os, sys
os.makedirs("${VM_DATA}", exist_ok=True)
if os.path.exists("${VM_ARROW}"):
    print(f"data already present at ${VM_ARROW} ({os.path.getsize('${VM_ARROW}')//1024//1024} MB)")
else:
    r = subprocess.run(
        ["huggingface-cli", "download", "${HF_DATASET}", "${HF_FILE}",
         "--repo-type", "dataset", "--local-dir", "${VM_DATA}"],
        capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr); raise SystemExit(r.returncode)
mb = os.path.getsize("${VM_ARROW}") / (1024*1024)
print(f"OK: data file is {mb:.0f} MB at ${VM_ARROW}")
EOF

log "downloading pretrained weights (PP-OCRv6 small rec)"
"$COLAB" exec -s "$SESSION" --timeout 300 <<EOF
import subprocess, os, sys
url = "${PRETRAIN_URL}"
dest = "${VM_PRETRAIN}.pdparams"   # paddle wants the path without extension; file has it
if os.path.exists(dest):
    print(f"pretrained weights already present ({os.path.getsize(dest)//1024//1024} MB)")
else:
    r = subprocess.run(["curl", "-L", "--fail", "-o", dest, url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr); raise SystemExit(r.returncode)
print(f"OK: pretrained weights at {dest} ({os.path.getsize(dest)//1024//1024} MB)")
EOF

# ---------------------------------------------------------------------------
# 5. EXTRACT: arrow -> crops + dict + label lists
# ---------------------------------------------------------------------------
log "extracting dataset (arrow -> PaddleOCR format)"
# Build the CLI args as a shell string (tokenized by the VM's shell via shell=True),
# NOT as a Python list — avoids the "shell fragment into Python list" quoting bug.
EXTRACT_ARGS="--smoke-per-split 5000"   # always keep the 5K smoke subsets for the gate
if [[ -n "$MAX_PER_SPLIT" ]]; then
  EXTRACT_ARGS="--max-per-split ${MAX_PER_SPLIT} ${EXTRACT_ARGS}"
fi
"$COLAB" exec -s "$SESSION" --timeout 1800 <<EOF
import subprocess, sys, os
os.chdir("${VM_REPO}")
cmd = "python scripts/build_paddle_dataset.py --arrow ${VM_ARROW} --out train_data/burmese_rec ${EXTRACT_ARGS}"
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(r.stdout, end="")
if r.returncode != 0:
    print(r.stderr, file=sys.stderr); raise SystemExit(r.returncode)
EOF

# ---------------------------------------------------------------------------
# 6. VERIFY GPU + (OPTIONAL) SMOKE GATE
# ---------------------------------------------------------------------------
log "verifying a CUDA GPU is available to paddle"
"$COLAB" exec -s "$SESSION" --timeout 120 <<EOF
import paddle
n = paddle.device.cuda.device_count()
print(f"paddle CUDA device count: {n}")
print(f"paddle CUDA version: {paddle.version.cuda()}")
assert n >= 1, "ERROR: no CUDA GPU visible to paddle — aborting before wasting a full run."
print("OK: GPU available")
EOF

if [[ "$SMOKE_FIRST" == "1" ]]; then
  log "SMOKE GATE: 1-epoch run on 5K subset (aborts if it fails)"
  "$COLAB" exec -s "$SESSION" --timeout 3600 <<EOF
import subprocess, sys, os
os.chdir("${VM_REPO}")
r = subprocess.run(
    ["bash", "scripts/train_burmese.sh", "smoke"],
    capture_output=True, text=True)
# Print the tail so we see acc/loss even on success.
out = r.stdout.decode(errors="replace") if isinstance(r.stdout, bytes) else r.stdout
print(out[-4000:])
if r.returncode != 0:
    err = r.stderr.decode(errors="replace") if isinstance(r.stderr, bytes) else r.stderr
    print(err[-2000:], file=sys.stderr)
    raise SystemExit(r.returncode)
print("OK: smoke run completed")
EOF
fi

# ---------------------------------------------------------------------------
# 7. FULL TRAIN (batch overridden to BATCH_SIZE for T4 VRAM safety)
# ---------------------------------------------------------------------------
log "STARTING FULL TRAINING → ${VM_REPO}/output/burmese_PP-OCRv6_small_rec"
"$COLAB" exec -s "$SESSION" --timeout 86400 <<EOF
import subprocess, sys, os
os.chdir("${VM_REPO}")
r = subprocess.run(
    ["python", "tools/train.py",
     "-c", "configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml",
     "-o", "Global.epoch_num=${EPOCHS}",
            "Train.loader.batch_size_per_card=${BATCH_SIZE}",
            "Eval.loader.batch_size_per_card=${BATCH_SIZE}",
            "Train.sampler.first_bs=${BATCH_SIZE}"],
    capture_output=True, text=True)
out = r.stdout if isinstance(r.stdout, str) else r.stdout.decode(errors="replace")
print(out[-6000:])
if r.returncode != 0:
    err = r.stderr if isinstance(r.stderr, str) else r.stderr.decode(errors="replace")
    print(err[-3000:], file=sys.stderr)
    raise SystemExit(r.returncode)
print("OK: full training completed")
EOF

# ---------------------------------------------------------------------------
# 8. EXPORT (checkpoint -> inference model) + EVAL (final test-split CER)
# ---------------------------------------------------------------------------
log "exporting inference model"
"$COLAB" exec -s "$SESSION" --timeout 600 <<EOF
import subprocess, sys, os
os.chdir("${VM_REPO}")
r = subprocess.run(
    ["python", "tools/export_model.py",
     "-c", "configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml",
     "-o", "Global.checkpoints=./output/burmese_PP-OCRv6_small_rec/best_accuracy.pdparams",
            "Global.save_inference_dir=./models/burmese_PP-OCRv6_small_rec_infer"],
    capture_output=True, text=True)
print(r.stdout[-2000:] if isinstance(r.stdout, str) else r.stdout.decode(errors="replace")[-2000:])
if r.returncode != 0:
    print(r.stderr if isinstance(r.stderr, str) else r.stderr.decode(errors="replace"), file=sys.stderr)
    raise SystemExit(r.returncode)
print("OK: inference model exported to ${VM_INFER}")
EOF

log "final eval on held-out test split (CER comparable to Kraken benchmarks)"
"$COLAB" exec -s "$SESSION" --timeout 1800 <<EOF
import subprocess, sys, os
os.chdir("${VM_REPO}")
r = subprocess.run(
    ["python", "tools/eval.py",
     "-c", "configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml",
     "-o", "Eval.dataset.label_file_list=[\"./train_data/burmese_rec/test_list.txt\"]",
            "Global.checkpoints=./output/burmese_PP-OCRv6_small_rec/best_accuracy.pdparams"],
    capture_output=True, text=True)
print(r.stdout[-2000:] if isinstance(r.stdout, str) else r.stdout.decode(errors="replace")[-2000:])
if r.returncode != 0:
    print(r.stderr if isinstance(r.stderr, str) else r.stderr.decode(errors="replace"), file=sys.stderr)
    raise SystemExit(r.returncode)
EOF

# ---------------------------------------------------------------------------
# 9. ZIP THE INFERENCE MODEL, DOWNLOAD LOCALLY, (OPTIONAL) DRIVE BACKUP
# ---------------------------------------------------------------------------
log "zipping inference model"
"$COLAB" exec -s "$SESSION" --timeout 600 <<EOF
import shutil, os
src = "${VM_INFER}"
dst_zip = "/content/burmese_infer.zip"
if not os.path.isdir(src):
    raise SystemExit(f"ERROR: inference dir {src} does not exist")
shutil.make_archive(dst_zip[:-4], "zip", root_dir=os.path.dirname(src),
                    base_dir=os.path.basename(src))
print(f"OK: archive {os.path.getsize(dst_zip)//1024} KB at {dst_zip}")
EOF

log "downloading inference zip → ${LOCAL_DIR}/"
mkdir -p "$LOCAL_DIR"
if "$COLAB" download -s "$SESSION" "/content/burmese_infer.zip" "${LOCAL_DIR}/burmese_infer.zip" 2>/dev/null; then
  echo "[script] downloaded ${LOCAL_DIR}/burmese_infer.zip"
  echo "[script] unzip with: cd ${LOCAL_DIR} && unzip burmese_infer.zip"
else
  echo "[script] WARNING: local download failed." >&2
fi

if [[ "$DRIVE_BACKUP" == "1" ]]; then
  log "mounting Drive for output backup → ${REMOTE_ZIP}"
  "$COLAB" drivemount -s "$SESSION" || true
  "$COLAB" exec -s "$SESSION" --timeout 300 <<EOF
import shutil, os
os.makedirs("${DRIVE_BASE}", exist_ok=True)
shutil.copy("/content/burmese_infer.zip", "${REMOTE_ZIP}")
print(f"OK: Drive backup at ${REMOTE_ZIP}")
EOF
fi

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------
log "DONE."
echo ""
echo "  Inference model (local):"
echo "    ${LOCAL_DIR}/burmese_infer.zip"
[[ "$DRIVE_BACKUP" == "1" ]] && echo "  Drive backup: ${REMOTE_ZIP}"
echo ""
echo "  Unzip and point your detection pipeline at it:"
echo "    cd ${LOCAL_DIR} && unzip burmese_infer.zip"
echo ""
echo "  (Session '${SESSION}' torn down automatically on exit.)"
