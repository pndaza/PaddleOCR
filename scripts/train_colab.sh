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

# Pre-built PaddleOCR dataset zip (train_data/burmese_rec/ contents: dict, label
# lists, smoke subsets, train/val/test PNG crops). Built locally once, uploaded
# to Dropbox — the VM downloads + unzips instead of downloading the 1.1 GB arrow
# from HuggingFace AND extracting on the VM (which was slow + colab-exec flaky).
# Dropbox: append ?dl=1 to the share link for direct download.
DATASET_ZIP_URL="${DATASET_ZIP_URL:-https://www.dropbox.com/scl/fi/ibvp1cjeoi6jlnjp34icm/burmese_rec_dataset.zip?rlkey=00iknrubirk4ubcfayiji8bn6&st=v3175nwf&dl=1}"

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
# Default wheel source: a Dropbox mirror of paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl
# (downloaded once from Baidu, uploaded to Dropbox — fast CDN from Colab, no per-session
# 1.9 GB download from paddle-whl.bj.bcebos.com). Override with any other direct (?dl=1) URL,
# or set PADDLE_WHEEL_URL='' to fall back to the Baidu pip index.
PADDLE_WHEEL_URL="${PADDLE_WHEEL_URL:-https://www.dropbox.com/scl/fi/71xe2qpsxd31e5wpe4k17/paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl?rlkey=qz4u9xupt42x9kdc6561l1fia&st=7wx7lezq&dl=1}"

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
# RELIABLE STEP EXECUTION across the colab-exec boundary.
#
# Two problems with `colab exec <long-cell>`:
#   1. It HANGS on long-running cells — the cell completes and the kernel goes
#      IDLE, but the CLI never returns control to bash (confirmed: extraction
#      finished, sentinel written, but the script was wedged for 10+ min).
#   2. It returns exit code 0 even when the inner Python raised SystemExit(1)
#      (Jupyter swallows it). $? and `set -e` are useless.
#
# Solution: split "launch work" from "wait for it".
#   - bg_step <name> <script> : writes <script> to a VM file, launches it
#     DETACHED (nohup, start_new_session) so colab exec returns immediately,
#     then POLLS the sentinel in a bash loop with timeout. While polling it
#     tails the work's log so you see live progress. On success or timeout,
#     it returns 0 / aborts.
#   - quick_step <name> <script> : for short (<60s) steps where the hang risk
#     is low; runs synchronously and checks the sentinel.
#
# Both write /content/.steps/<name>.ok (sentinel) + /content/.steps/<name>.log
# (output). Skip-if-done: if the sentinel already exists, the step is skipped.
# ---------------------------------------------------------------------------
SENTINEL_DIR="/content/.steps"
STEP_TIMEOUT_DEFAULT=3600   # 1h default per step; override per-call

# _check_sentinel <name>: prints PRESENT or MISSING, exits 0 always.
_check_sentinel() {
  local name="$1"
  local out
  out="$("$COLAB" exec -s "$SESSION" --timeout 30 <<EOF
import os
print("SENTINEL_PRESENT" if os.path.exists("${SENTINEL_DIR}/${name}.ok") else "SENTINEL_MISSING")
EOF
)" 2>/dev/null
  if echo "$out" | grep -q "SENTINEL_PRESENT"; then echo PRESENT; else echo MISSING; fi
}

# quick_step <name> [timeout]: run a synchronous colab-exec that should write
# the sentinel. The cell body is read from stdin (heredoc). Short steps only.
quick_step() {
  local name="$1"; local tmo="${2:-120}"
  if [[ "$(_check_sentinel "$name")" == "PRESENT" ]]; then
    echo "[script] ✓ step '${name}' already done (sentinel present) — skipping"
    return 0
  fi
  "$COLAB" exec -s "$SESSION" --timeout "$tmo"
  if [[ "$(_check_sentinel "$name")" == "PRESENT" ]]; then
    echo "[script] ✓ step '${name}' OK"
  else
    echo "[script] ✗ step '${name}' FAILED — aborting." >&2
    exit 1
  fi
}

# bg_step <name> <timeout_sec>: reads a Python script from stdin, writes it to
# /content/.steps/<name>.py on the VM, launches it detached, then polls the
# sentinel (tailing the log) until present or timeout. For LONG steps only.
bg_step() {
  local name="$1"; local tmo="${2:-$STEP_TIMEOUT_DEFAULT}"
  if [[ "$(_check_sentinel "$name")" == "PRESENT" ]]; then
    echo "[script] ✓ step '${name}' already done (sentinel present) — skipping"
    return 0
  fi
  # Read the script body from stdin into a local var (heredoc passed by caller).
  local body
  body="$(cat)"
  # Write the script to the VM + launch it detached.
  # The body is base64-encoded to survive shell/heredoc quoting faithfully.
  local b64
  b64="$(printf '%s' "$body" | base64 | tr -d '\n')"
  "$COLAB" exec -s "$SESSION" --timeout 60 <<EOF
import os, subprocess, base64, sys
os.makedirs("${SENTINEL_DIR}", exist_ok=True)
script = base64.b64decode("${b64}").decode("utf-8")
path = "${SENTINEL_DIR}/${name}.py"
open(path, "w").write(script)
# Launch detached: nohup, new session, output to the step's log.
log = "${SENTINEL_DIR}/${name}.log"
subprocess.Popen(["nohup", sys.executable, "-u", path],
                 stdout=open(log, "w"), stderr=subprocess.STDOUT,
                 start_new_session=True)
print("launched ${name} in background; polling for sentinel")
EOF
  echo "[script] launched '${name}' in background; polling (timeout ${tmo}s)"
  # Poll loop: check sentinel every 15s, tail the log for live progress.
  # Also detect a DEAD background process — if it's no longer running AND the
  # sentinel isn't written, the step crashed (fail fast instead of waiting the
  # full timeout blind).
  local start=$(( $(date +%s) ))
  local last_size=0
  # Give it a 5s grace period before checking liveness (process may not have
  # started yet when we first poll).
  sleep 5
  while :; do
    if [[ "$(_check_sentinel "$name")" == "PRESENT" ]]; then
      echo "[script] ✓ step '${name}' OK"
      _tail_log "$name" "$last_size"
      return 0
    fi
    # is the background process still alive?
    if [[ "$(_check_alive "$name")" != "ALIVE" ]]; then
      echo "[script] ✗ step '${name}' process exited without writing sentinel — FAILED." >&2
      echo "[script]   tail of log:" >&2
      _tail_log "$name" "$last_size" >&2
      exit 1
    fi
    # check for timeout
    local now=$(( $(date +%s) ))
    if (( now - start > tmo )); then
      echo "[script] ✗ step '${name}' TIMED OUT after ${tmo}s — aborting." >&2
      _tail_log "$name" "$last_size" >&2
      exit 1
    fi
    # stream any new log output
    _tail_log "$name" "$last_size"
    last_size=$(_log_size "$name")
    sleep 15
  done
}

# _check_alive <name>: is the background python for this step still running?
# Looks for a process running "<name>.py". Prints ALIVE or DEAD.
_check_alive() {
  local name="$1"
  local out
  out="$("$COLAB" exec -s "$SESSION" --timeout 20 <<EOF
import subprocess
# pgrep for the step script file. -f matches the full command line.
r = subprocess.run(["pgrep", "-f", "${SENTINEL_DIR}/${name}.py"],
                   capture_output=True, text=True)
print("ALIVE" if r.returncode == 0 else "DEAD")
EOF
)" 2>/dev/null
  if echo "$out" | grep -q "ALIVE"; then echo ALIVE; else echo DEAD; fi
}

# _log_size / _tail_log: helpers to stream the VM log file incrementally.
_log_size() {
  "$COLAB" exec -s "$SESSION" --timeout 20 <<EOF
import os
p = "${SENTINEL_DIR}/$1.log"
print(os.path.getsize(p) if os.path.exists(p) else 0)
EOF
}

_tail_log() {
  local name="$1"; local since="${2:-0}"
  "$COLAB" exec -s "$SESSION" --timeout 20 <<EOF
import os
p = "${SENTINEL_DIR}/${name}.log"
if os.path.exists(p):
    size = os.path.getsize(p)
    if size > ${since}:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            f.seek(${since})
            print(f.read(), end="")
EOF
}



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
quick_step clone 120 <<EOF
import subprocess, sys, os
r = subprocess.run(
    ["git", "clone", "--branch", "${FORK_BRANCH}", "--depth", "1",
     "${FORK_URL}", "${VM_REPO}"],
    capture_output=True, text=True)
print(r.stdout, end="")
if r.returncode != 0:
    print(r.stderr, file=sys.stderr); sys.exit(1)
sub = subprocess.run(["git", "-C", "${VM_REPO}", "remote", "remove", "origin"],
                     capture_output=True, text=True)
print("removed 'origin' remote (upstream-push safeguard):", sub.stdout.strip() or sub.stderr.strip())
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/clone.ok","w").close()
EOF

# ---------------------------------------------------------------------------
# 3. INSTALL PADDLE (GPU) + REPO REQUIREMENTS
# ---------------------------------------------------------------------------
if [[ -n "$PADDLE_WHEEL_URL" ]]; then
  log "installing paddle from wheel: ${PADDLE_WHEEL_URL}"
else
  log "installing ${PADDLE_PKG} from Baidu index ${PADDLE_INDEX}"
fi
# Paddle install can take a few minutes (1.9 GB wheel) — background it.
bg_step paddle 900 <<EOF
import subprocess, sys, os
wheel_url = "${PADDLE_WHEEL_URL}"
if wheel_url:
    dest = "/content/paddlepaddle_gpu-3.2.0-cp312-cp312-linux_x86_64.whl"
    r = subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", dest, wheel_url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr); sys.exit(1)
    cmd = [sys.executable, "-m", "pip", "install", dest]
else:
    cmd = [sys.executable, "-m", "pip", "install",
           "--timeout", "300", "--retries", "5",
           "-i", "${PADDLE_INDEX}", "${PADDLE_PKG}"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-2000:])
if r.returncode != 0:
    print(r.stderr[-2000:], file=sys.stderr); sys.exit(1)
import paddle
assert paddle.device.is_compiled_with_cuda(), "paddle NOT compiled with CUDA — wrong wheel"
print(f"OK: paddle {paddle.__version__}, cuda compiled: {paddle.device.is_compiled_with_cuda()}")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/paddle.ok","w").close()
EOF

log "installing repo requirements.txt"
quick_step reqs 300 <<EOF
import subprocess, sys, os
r = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "${VM_REPO}/requirements.txt"],
                   capture_output=True, text=True)
print(r.stdout[-1500:])
if r.returncode != 0:
    print(r.stderr[-1500:], file=sys.stderr); sys.exit(1)
print("OK: repo requirements installed")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/reqs.ok","w").close()
EOF

# ---------------------------------------------------------------------------
# 4. GET THE DATASET  (pre-built zip preferred; fall back to HF arrow + extract)
# ---------------------------------------------------------------------------
if [[ -n "$DATASET_ZIP_URL" ]]; then
  # Preferred: download a pre-built train_data/burmese_rec/ zip from Dropbox and
  # unzip it on the VM. Avoids both the 1.1 GB HuggingFace arrow download AND the
  # slow on-VM extraction (which was also colab-exec flaky).
  log "downloading pre-built dataset zip from Dropbox"
  bg_step dataset 1200 <<EOF
import subprocess, os, sys
dest = "/content/burmese_rec_dataset.zip"
if os.path.exists(dest):
    print(f"zip already present ({os.path.getsize(dest)//1024//1024} MB)")
else:
    r = subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", dest, "${DATASET_ZIP_URL}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr); sys.exit(1)
    print(f"downloaded {os.path.getsize(dest)//1024//1024} MB")
# Unzip into the repo (zip contains burmese_rec/ at its root).
os.chdir("${VM_REPO}")
os.makedirs("train_data", exist_ok=True)
r = subprocess.run(["unzip", "-o", "-q", dest, "-d", "train_data"],
                   capture_output=True, text=True)
print(r.stdout[-500:])
if r.returncode != 0:
    print(r.stderr[-500:], file=sys.stderr); sys.exit(1)
# Verify the expected contents landed.
for f in ["burmese_dict.txt", "train_list.txt", "val_list.txt", "test_list.txt", "train_list_smoke.txt"]:
    p = os.path.join("train_data", "burmese_rec", f)
    if not os.path.exists(p):
        print(f"ERROR: unzipped but {p} missing", file=sys.stderr); sys.exit(1)
n_train = len(os.listdir(os.path.join("train_data","burmese_rec","train")))
print(f"OK: dataset ready ({n_train} train crops)")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/dataset.ok","w").close()
EOF
else
  # Fallback: HF arrow download + on-VM extraction (slower, was colab-exec flaky).
  log "DATASET_ZIP_URL not set — falling back to HF arrow + on-VM extract"
  bg_step data 600 <<EOF
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
        print(r.stderr, file=sys.stderr); sys.exit(1)
if not os.path.exists("${VM_ARROW}"):
    print(f"ERROR: download reported success but ${VM_ARROW} missing", file=sys.stderr)
    sys.exit(1)
mb = os.path.getsize("${VM_ARROW}") / (1024*1024)
print(f"OK: data file is {mb:.0f} MB at ${VM_ARROW}")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/data.ok","w").close()
EOF

  log "extracting dataset (arrow -> PaddleOCR format)"
  EXTRACT_ARGS="--smoke-per-split 5000"
  if [[ -n "$MAX_PER_SPLIT" ]]; then
    EXTRACT_ARGS="--max-per-split ${MAX_PER_SPLIT} ${EXTRACT_ARGS}"
  fi
  bg_step extract 1800 <<EOF
import subprocess, sys, os
os.chdir("${VM_REPO}")
cmd = "python scripts/build_paddle_dataset.py --arrow ${VM_ARROW} --out train_data/burmese_rec ${EXTRACT_ARGS}"
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(r.stdout, end="")
if r.returncode != 0:
    print(r.stderr, file=sys.stderr); sys.exit(1)
for f in ["burmese_dict.txt", "train_list.txt", "val_list.txt", "train_list_smoke.txt"]:
    p = os.path.join("train_data", "burmese_rec", f)
    if not os.path.exists(p):
        print(f"ERROR: extraction reported success but {p} missing", file=sys.stderr)
        sys.exit(1)
print("OK: extraction produced dict + label lists")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/extract.ok","w").close()
EOF
fi

log "downloading pretrained weights (PP-OCRv6 small rec)"
quick_step pretrain 300 <<EOF
import subprocess, os, sys
dest = "${VM_PRETRAIN}.pdparams"
if os.path.exists(dest):
    print(f"pretrained weights already present ({os.path.getsize(dest)//1024//1024} MB)")
else:
    r = subprocess.run(["curl", "-L", "--fail", "-o", dest, "${PRETRAIN_URL}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr); sys.exit(1)
print(f"OK: pretrained weights at {dest} ({os.path.getsize(dest)//1024//1024} MB)")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/pretrain.ok","w").close()
EOF

# ---------------------------------------------------------------------------
# 6. VERIFY GPU + (OPTIONAL) SMOKE GATE
# ---------------------------------------------------------------------------
log "verifying a CUDA GPU is available to paddle"
quick_step gpu 60 <<EOF
import paddle, os, sys
n = paddle.device.cuda.device_count()
print(f"paddle CUDA device count: {n}, version: {paddle.version.cuda()}")
if n < 1:
    print("ERROR: no CUDA GPU visible to paddle", file=sys.stderr); sys.exit(1)
print("OK: GPU available")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/gpu.ok","w").close()
EOF

if [[ "$SMOKE_FIRST" == "1" ]]; then
  log "SMOKE GATE: 1-epoch run on 5K subset (aborts if it fails)"
  bg_step smoke 3600 <<EOF
import subprocess, sys, os
os.chdir("${VM_REPO}")
r = subprocess.run(["bash", "scripts/train_burmese.sh", "smoke"],
                   capture_output=True, text=True)
out = r.stdout if isinstance(r.stdout, str) else r.stdout.decode(errors="replace")
print(out[-4000:])
if r.returncode != 0:
    sys.exit(1)
print("OK: smoke run completed")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/smoke.ok","w").close()
EOF
fi

# ---------------------------------------------------------------------------
# 7. FULL TRAIN (batch overridden to BATCH_SIZE for T4 VRAM safety)  (LONG)
# ---------------------------------------------------------------------------
log "STARTING FULL TRAINING → ${VM_REPO}/output/burmese_PP-OCRv6_small_rec"
bg_step train 86400 <<EOF
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
    sys.exit(1)
ckpt = os.path.join("output", "burmese_PP-OCRv6_small_rec", "best_accuracy.pdparams")
if not os.path.exists(ckpt):
    print(f"ERROR: training reported success but {ckpt} missing", file=sys.stderr)
    sys.exit(1)
print("OK: full training completed, best_accuracy saved")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/train.ok","w").close()
EOF

# ---------------------------------------------------------------------------
# 8. EXPORT (checkpoint -> inference model) + EVAL (final test-split CER)
# ---------------------------------------------------------------------------
log "exporting inference model"
quick_step export 600 <<EOF
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
    sys.exit(1)
for f in ["inference.pdmodel", "inference.pdiparams"]:
    p = os.path.join("models", "burmese_PP-OCRv6_small_rec_infer", f)
    if not os.path.exists(p):
        print(f"ERROR: export reported success but {p} missing", file=sys.stderr)
        sys.exit(1)
print("OK: inference model exported")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/export.ok","w").close()
EOF

log "final eval on held-out test split (CER comparable to Kraken benchmarks)"
bg_step eval 3600 <<EOF
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
    sys.exit(1)
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/eval.ok","w").close()
EOF

# ---------------------------------------------------------------------------
# 9. ZIP THE INFERENCE MODEL, DOWNLOAD LOCALLY, (OPTIONAL) DRIVE BACKUP
# ---------------------------------------------------------------------------
log "zipping inference model"
quick_step zip 300 <<EOF
import shutil, os, sys
src = "${VM_INFER}"
dst_zip = "/content/burmese_infer.zip"
if not os.path.isdir(src):
    print(f"ERROR: inference dir {src} does not exist", file=sys.stderr)
    sys.exit(1)
shutil.make_archive(dst_zip[:-4], "zip", root_dir=os.path.dirname(src),
                    base_dir=os.path.basename(src))
print(f"OK: archive {os.path.getsize(dst_zip)//1024} KB at {dst_zip}")
os.makedirs("${SENTINEL_DIR}", exist_ok=True); open("${SENTINEL_DIR}/zip.ok","w").close()
EOF

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
