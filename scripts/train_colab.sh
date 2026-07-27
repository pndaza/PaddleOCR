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

# Paddle GPU wheel. Paddle 3.x is NOT published to PyPI (only 2.6.x is); the
# only publisher is the Baidu cu126 index (bundles CUDA 12.6 + cuDNN). That
# CDN is slow/unreliable from Colab, so prefer a direct wheel URL when set.
#
# Recommended: pre-download the wheel from
#   https://paddle-whl.bj.bcebos.com/stable/cu126/paddlepaddle-gpu/paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl
# (fast on a home connection), upload to Dropbox/Drive, and point the VM at it:
#   PADDLE_WHEEL_URL='https://www.dropbox.com/s/<id>/paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl?dl=1' bash scripts/train_colab.sh
# (Dropbox: append ?dl=1 to the share link for direct download.)
# When PADDLE_WHEEL_URL is empty, falls back to pip install from PADDLE_INDEX.
PADDLE_PKG="${PADDLE_PKG:-paddlepaddle-gpu==3.2.0}"
PADDLE_INDEX="${PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"
PADDLE_WHEEL_URL="${PADDLE_WHEEL_URL:-}"   # empty = use PADDLE_INDEX; set = curl the wheel directly

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
# RELIABLE ERROR CHECKING across the colab-exec boundary.
#
# `colab exec` runs the cell in a Jupyter kernel, which SWALLOWS SystemExit
# (IPython prints the traceback but the process keeps running). As a result
# `colab exec` returns 0 even when the inner Python raised SystemExit(1) —
# confirmed empirically. So `set -e` and `$?` checks are USELESS here.
#
# Workaround: each step writes a sentinel file at the END of its happy path.
# run_step checks for the sentinel; if missing, the step failed and we abort.
# Sentinels are VM-local (on /content), queried via a tiny colab exec.
# ---------------------------------------------------------------------------
SENTINEL_DIR="/content/.steps"

# run_step <name>: verify the named step's sentinel exists on the VM.
# Usage: run_step clone   # after a `colab exec` block that should touch /content/.steps/clone.ok
run_step() {
  local name="$1"
  local out
  out="$("$COLAB" exec -s "$SESSION" --timeout 30 <<EOF
import os, sys
ok = os.path.exists("${SENTINEL_DIR}/${name}.ok")
print("SENTINEL_PRESENT" if ok else "SENTINEL_MISSING")
EOF
)"
  if echo "$out" | grep -q "SENTINEL_PRESENT"; then
    echo "[script] ✓ step '${name}' OK"
  else
    echo "[script] ✗ step '${name}' FAILED — aborting (see traceback above)." >&2
    echo "[script]   (colab exec masks Python failures; sentinel check caught it.)" >&2
    exit 1
  fi
}

# Inside each step's colab-exec heredoc, the happy path ends with:
#   import os; os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/<name>.ok","w").close()
# That sentinel is what run_step checks. If the step raises before reaching it,
# the sentinel is never written and run_step aborts the whole script.


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
import subprocess, sys, os
r = subprocess.run(
    ["git", "clone", "--branch", "${FORK_BRANCH}", "--depth", "1",
     "${FORK_URL}", "${VM_REPO}"],
    capture_output=True, text=True)
print(r.stdout, end="")
if r.returncode != 0:
    print(r.stderr, file=sys.stderr)
    sys.exit("__CLONE_FAILED__")   # signal failure (sentinel won't be written)
# Remove the only remote so NO push (to fork or upstream) is possible from the VM.
sub = subprocess.run(["git", "-C", "${VM_REPO}", "remote", "remove", "origin"],
                     capture_output=True, text=True)
print("removed 'origin' remote (upstream-push safeguard):", sub.stdout.strip() or sub.stderr.strip())
chk = subprocess.run(["git", "-C", "${VM_REPO}", "remote", "-v"], capture_output=True, text=True)
print("remaining remotes:", repr(chk.stdout.strip()) or "(none)")
# SUCCESS: write sentinel LAST.
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/clone.ok","w").close()
EOF
run_step clone

# ---------------------------------------------------------------------------
# 3. INSTALL PADDLE (GPU) + REPO REQUIREMENTS
# ---------------------------------------------------------------------------
if [[ -n "$PADDLE_WHEEL_URL" ]]; then
  log "installing paddle from wheel: ${PADDLE_WHEEL_URL}"
else
  log "installing ${PADDLE_PKG} from Baidu index ${PADDLE_INDEX}"
fi
"$COLAB" exec -s "$SESSION" --timeout 1800 <<EOF
import subprocess, sys, os
wheel_url = "${PADDLE_WHEEL_URL}"
if wheel_url:
    # Download the wheel directly (Dropbox/Drive direct link), then pip install
    # the local file. Faster and resumable (curl -C -) than pip from Baidu CDN.
    dest = "/content/paddle.whl"
    r = subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", dest, wheel_url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr); sys.exit("__PADDLE_FAILED__")
    cmd = [sys.executable, "-m", "pip", "install", dest]
else:
    cmd = [sys.executable, "-m", "pip", "install",
           "--timeout", "300", "--retries", "5",
           "-i", "${PADDLE_INDEX}", "${PADDLE_PKG}"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-2000:])
if r.returncode != 0:
    print(r.stderr[-2000:], file=sys.stderr); sys.exit("__PADDLE_FAILED__")
import paddle
assert paddle.device.is_compiled_with_cuda(), "paddle NOT compiled with CUDA — wrong wheel"
print(f"OK: paddle {paddle.__version__}, cuda compiled: {paddle.device.is_compiled_with_cuda()}")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/paddle.ok","w").close()
EOF
run_step paddle

log "installing repo requirements.txt"
"$COLAB" exec -s "$SESSION" --timeout 600 <<EOF
import subprocess, sys, os
r = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "${VM_REPO}/requirements.txt"],
                   capture_output=True, text=True)
print(r.stdout[-1500:])
if r.returncode != 0:
    print(r.stderr[-1500:], file=sys.stderr); sys.exit("__REQS_FAILED__")
print("OK: repo requirements installed")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/reqs.ok","w").close()
EOF
run_step reqs

# ---------------------------------------------------------------------------
# 4. DOWNLOAD DATA (HuggingFace) + PRETRAINED WEIGHTS
# ---------------------------------------------------------------------------
# NOTE: `huggingface-cli` is DEPRECATED on Colab (exits 1, no-op). Use `hf`.
log "downloading ${HF_FILE} from HuggingFace ${HF_DATASET} (~1.1 GB)"
"$COLAB" exec -s "$SESSION" --timeout 900 <<EOF
import subprocess, os, sys
os.makedirs("${VM_DATA}", exist_ok=True)
if os.path.exists("${VM_ARROW}"):
    print(f"data already present at ${VM_ARROW} ({os.path.getsize('${VM_ARROW}')//1024//1024} MB)")
else:
    r = subprocess.run(
        ["hf", "download", "${HF_DATASET}", "${HF_FILE}",
         "--repo-type", "dataset", "--local-dir", "${VM_DATA}"],
        capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr); sys.exit("__HF_FAILED__")
if not os.path.exists("${VM_ARROW}"):
    print(f"ERROR: download reported success but ${VM_ARROW} missing", file=sys.stderr)
    sys.exit("__HF_MISSING__")
mb = os.path.getsize("${VM_ARROW}") / (1024*1024)
print(f"OK: data file is {mb:.0f} MB at ${VM_ARROW}")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/data.ok","w").close()
EOF
run_step data

log "downloading pretrained weights (PP-OCRv6 small rec)"
"$COLAB" exec -s "$SESSION" --timeout 300 <<EOF
import subprocess, os, sys
dest = "${VM_PRETRAIN}.pdparams"
if os.path.exists(dest):
    print(f"pretrained weights already present ({os.path.getsize(dest)//1024//1024} MB)")
else:
    r = subprocess.run(["curl", "-L", "--fail", "-o", dest, "${PRETRAIN_URL}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr); sys.exit("__PRETRAIN_FAILED__")
print(f"OK: pretrained weights at {dest} ({os.path.getsize(dest)//1024//1024} MB)")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/pretrain.ok","w").close()
EOF
run_step pretrain

# ---------------------------------------------------------------------------
# 5. EXTRACT: arrow -> crops + dict + label lists
# ---------------------------------------------------------------------------
log "extracting dataset (arrow -> PaddleOCR format)"
EXTRACT_ARGS="--smoke-per-split 5000"
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
    print(r.stderr, file=sys.stderr); sys.exit("__EXTRACT_FAILED__")
# Verify the expected outputs exist before claiming success.
for f in ["burmese_dict.txt", "train_list.txt", "val_list.txt", "train_list_smoke.txt"]:
    p = os.path.join("train_data/burmese_rec", f)
    if not os.path.exists(p):
        print(f"ERROR: extraction reported success but {p} missing", file=sys.stderr)
        sys.exit("__EXTRACT_MISSING__")
print("OK: extraction produced dict + label lists")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/extract.ok","w").close()
EOF
run_step extract

# ---------------------------------------------------------------------------
# 6. VERIFY GPU + (OPTIONAL) SMOKE GATE
# ---------------------------------------------------------------------------
log "verifying a CUDA GPU is available to paddle"
"$COLAB" exec -s "$SESSION" --timeout 120 <<EOF
import paddle, os, sys
n = paddle.device.cuda.device_count()
print(f"paddle CUDA device count: {n}, version: {paddle.version.cuda()}")
if n < 1:
    print("ERROR: no CUDA GPU visible to paddle", file=sys.stderr); sys.exit("__NO_GPU__")
print("OK: GPU available")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/gpu.ok","w").close()
EOF
run_step gpu

if [[ "$SMOKE_FIRST" == "1" ]]; then
  log "SMOKE GATE: 1-epoch run on 5K subset (aborts if it fails)"
  "$COLAB" exec -s "$SESSION" --timeout 3600 <<EOF
import subprocess, sys, os
os.chdir("${VM_REPO}")
r = subprocess.run(["bash", "scripts/train_burmese.sh", "smoke"],
                   capture_output=True, text=True)
out = r.stdout if isinstance(r.stdout, str) else r.stdout.decode(errors="replace")
print(out[-4000:])
if r.returncode != 0:
    err = r.stderr if isinstance(r.stderr, str) else r.stderr.decode(errors="replace")
    print(err[-2000:], file=sys.stderr); sys.exit("__SMOKE_FAILED__")
print("OK: smoke run completed")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/smoke.ok","w").close()
EOF
  run_step smoke
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
    print(err[-3000:], file=sys.stderr); sys.exit("__TRAIN_FAILED__")
# Verify best_accuracy checkpoint exists.
ckpt = "./output/burmese_PP-OCRv6_small_rec/best_accuracy.pdparams"
if not os.path.exists(ckpt):
    print(f"ERROR: training reported success but {ckpt} missing", file=sys.stderr)
    sys.exit("__TRAIN_NO_CKPT__")
print("OK: full training completed, best_accuracy saved")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/train.ok","w").close()
EOF
run_step train

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
out = r.stdout if isinstance(r.stdout, str) else r.stdout.decode(errors="replace")
print(out[-2000:])
if r.returncode != 0:
    err = r.stderr if isinstance(r.stderr, str) else r.stderr.decode(errors="replace")
    print(err[-2000:], file=sys.stderr); sys.exit("__EXPORT_FAILED__")
# Verify the inference model files exist.
for f in ["inference.pdmodel", "inference.pdiparams"]:
    p = os.path.join("./models/burmese_PP-OCRv6_small_rec_infer", f)
    if not os.path.exists(p):
        print(f"ERROR: export reported success but {p} missing", file=sys.stderr)
        sys.exit("__EXPORT_MISSING__")
print("OK: inference model exported to ${VM_INFER}")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/export.ok","w").close()
EOF
run_step export

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
out = r.stdout if isinstance(r.stdout, str) else r.stdout.decode(errors="replace")
print(out[-2500:])
if r.returncode != 0:
    err = r.stderr if isinstance(r.stderr, str) else r.stderr.decode(errors="replace")
    print(err[-2000:], file=sys.stderr); sys.exit("__EVAL_FAILED__")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/eval.ok","w").close()
EOF
run_step eval

# ---------------------------------------------------------------------------
# 9. ZIP THE INFERENCE MODEL, DOWNLOAD LOCALLY, (OPTIONAL) DRIVE BACKUP
# ---------------------------------------------------------------------------
log "zipping inference model"
"$COLAB" exec -s "$SESSION" --timeout 600 <<EOF
import shutil, os, sys
src = "${VM_INFER}"
dst_zip = "/content/burmese_infer.zip"
if not os.path.isdir(src):
    print(f"ERROR: inference dir {src} does not exist", file=sys.stderr)
    sys.exit("__ZIP_NO_SRC__")
shutil.make_archive(dst_zip[:-4], "zip", root_dir=os.path.dirname(src),
                    base_dir=os.path.basename(src))
print(f"OK: archive {os.path.getsize(dst_zip)//1024} KB at {dst_zip}")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/zip.ok","w").close()
EOF
run_step zip

log "downloading inference zip → ${LOCAL_DIR}/"
mkdir -p "$LOCAL_DIR"
if "$COLAB" download -s "$SESSION" "/content/burmese_infer.zip" "${LOCAL_DIR}/burmese_infer.zip"; then
  echo "[script] downloaded ${LOCAL_DIR}/burmese_infer.zip"
  echo "[script] unzip with: cd ${LOCAL_DIR} && unzip burmese_infer.zip"
else
  echo "[script] WARNING: local download failed — check Drive backup if enabled." >&2
fi

if [[ "$DRIVE_BACKUP" == "1" ]]; then
  log "mounting Drive for output backup → ${REMOTE_ZIP}"
  "$COLAB" drivemount -s "$SESSION" || true
  "$COLAB" exec -s "$SESSION" --timeout 300 <<EOF
import shutil, os, sys
os.makedirs("${DRIVE_BASE}", exist_ok=True)
try:
    shutil.copy("/content/burmese_infer.zip", "${REMOTE_ZIP}")
    print(f"OK: Drive backup at ${REMOTE_ZIP}")
except Exception as e:
    print(f"WARNING: Drive backup failed: {e}", file=sys.stderr)
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
