# Burmese PP-OCRv6 Recognition Fine-tune — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the Kraken arrow dataset (897K Burmese/Pali line crops) into PaddleOCR format and fine-tune a PP-OCRv6 small recognition model that exports to a deployable inference model.

**Architecture:** Clone `configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml`, retune geometry/lr for Burmese line lengths (measured median 53 chars / 557px wide vs stock 25 chars / 320px), build a 118-char Burmese dict from the arrow's embedded alphabet, fine-tune from the official PP-OCRv6 small rec pretrained checkpoint, export to inference format.

**Tech Stack:** PaddlePaddle 3.3.1 (in `.venv`), PaddleOCR training tools (`tools/train.py`, `tools/eval.py`, `tools/export_model.py`), pyarrow (to install), Pillow (already present), pytest.

**Spec:** `docs/superpowers/specs/2026-07-27-burmese-rec-finetune-design.md`

**Reference facts (verified, do not re-derive):**
- Test framework: pytest; load modules via `importlib.util` to avoid the paddlex import chain. Example: `tests/tools/test_program_safe_yaml.py`.
- CLI overrides parsed by `yaml.load` — lists MUST use bracket syntax: `-o Train.dataset.label_file_list=["./a.txt"]`.
- `pretrained_model` / `checkpoints` take a path WITHOUT the `.pdparams` extension.
- Image paths in label files are relative to `data_dir` (joined via `os.path.join`). Format: `rel/path\ttranscription` (literal tab).
- Arrow `lines` metadata JSON has key `alphabet`: a `{char: count}` dict of len 118.
- OOV chars are silently dropped by `CTCLabelEncode`/`NRTRLabelEncode` (warning commented out) → silent label corruption. Validate before training.
- Python entrypoints: `.venv/bin/python` for everything (after installing pyarrow).

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/build_paddle_dataset.py` | Read Kraken arrow → write PNG crops, `burmese_dict.txt`, `{train,val,test}_list.txt` (+ smoke subsets), validate OOV. Pure stdlib + pyarrow + PIL. No paddle import. |
| `scripts/build_paddle_dataset_test.py` | pytest unit tests for the extraction script (dict building, label-line formatting, OOV detection, path-relativization). |
| `configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml` | Burmese-tuned clone of `PP-OCRv6_small_rec.yml`. |
| `scripts/train_burmese.sh` | Orchestrator: `smoke` / `full` / `export` / `eval` modes. |

**Generated at runtime (gitignored — large/regeneratable):**
- `train_data/burmese_rec/{burmese_dict.txt, *_list.txt, *_list_smoke.txt}`
- `train_data/burmese_rec/{train,val}/*.png`
- `output/burmese_PP-OCRv6_small_rec/` (checkpoints)
- `models/burmese_PP-OCRv6_small_rec_infer/` (final deliverable)
- `PP-OCRv6_small_rec_pretrained.pdparams` (downloaded pretrained weights)

---

## Task 1: Add `pyarrow` to the PaddleOCR venv

**Files:**
- Modify: `.venv/` (install only; no source change)

- [ ] **Step 1: Confirm pyarrow is missing**

Run: `.venv/bin/python -c "import pyarrow"`
Expected: `ModuleNotFoundError: No module named 'pyarrow'`

- [ ] **Step 2: Install pyarrow via uv**

The `.venv` was created by `uv` (no pip module). Run:

```bash
uv pip install pyarrow --python .venv/bin/python
```

Expected: installs pyarrow into `.venv`.

- [ ] **Step 3: Verify both deps present**

Run: `.venv/bin/python -c "import pyarrow, PIL; print('pyarrow', pyarrow.__version__, 'PIL', PIL.__version__)"`
Expected: prints both versions, no error.

- [ ] **Step 4: Commit**

No source change in this task (venv only). Skip commit — `.venv/` is gitignored.

---

## Task 2: Create the extraction script scaffold + dict-building tests

The script is a single focused file. We build it test-first, starting with dict extraction from arrow metadata.

**Files:**
- Create: `scripts/build_paddle_dataset.py`
- Create: `scripts/build_paddle_dataset_test.py`

- [ ] **Step 1: Write the failing test for dict building**

Create `scripts/build_paddle_dataset_test.py`:

```python
"""Tests for build_paddle_dataset. Loaded via importlib to avoid the paddleocr import chain."""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_paddle_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_paddle_dataset", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_paddle_dataset"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_extract_alphabet_orders_by_codepoint():
    mod = _load_module()
    fake_meta = {
        b"lines": (
            b'{"type":"kraken_recognition_bbox","alphabet":'
            b'{"\u1040":1,"\u1000":2,"\u1021":3}}'
        )
    }
    chars = mod.extract_alphabet_chars(fake_meta)
    # sorted by Unicode codepoint: U+1000, U+1021, U+1040
    assert chars == ["\u1000", "\u1021", "\u1040"]


def test_extract_alphabet_char_count():
    mod = _load_module()
    fake_meta = {
        b"lines": (
            b'{"type":"kraken_recognition_bbox","alphabet":'
            b'{"\u1000":2,"\u1001":5}}'
        )
    }
    chars = mod.extract_alphabet_chars(fake_meta)
    assert len(chars) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest scripts/build_paddle_dataset_test.py -v`
Expected: FAIL — `ModuleNotFoundError` / file does not exist.

- [ ] **Step 3: Create the script with the dict function**

Create `scripts/build_paddle_dataset.py`:

```python
#!/usr/bin/env python3
"""Convert the Kraken Burmese arrow dataset to PaddleOCR recognition format.

Reads:  a Kraken binary arrow file (schema: lines struct<text, im> + train/val/test bools).
Writes: train_data/<out>/{burmese_dict.txt, *_list.txt, *_list_smoke.txt, train/*.png, val/*.png}

Usage:
    .venv/bin/python scripts/build_paddle_dataset.py \\
        --arrow /path/to/burmese_kraken_1m_clean.arrow \\
        --out train_data/burmese_rec \\
        [--max-per-split 50000]
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import pyarrow.ipc as ipc
from PIL import Image


# Default source — the user's existing dataset.
DEFAULT_ARROW = (
    "/Users/pndaza/Projects/playground/ocr/kraken-burmese/"
    "manifests/burmese_kraken_1m_clean.arrow"
)


def extract_alphabet_chars(metadata: dict) -> list[str]:
    """Return the alphabet chars from arrow schema metadata, sorted by codepoint.

    `metadata` is the dict returned by `pyarrow.Schema.metadata` (bytes keys).
    The `lines` key holds JSON with an `alphabet` field: {char: count}.
    """
    raw = metadata[b"lines"]
    data = json.loads(raw)
    alphabet = data["alphabet"]  # {char: count}
    return sorted(alphabet.keys(), key=lambda c: ord(c))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest scripts/build_paddle_dataset_test.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_paddle_dataset.py scripts/build_paddle_dataset_test.py
git commit -m "feat(rec-burmese): extract alphabet chars from arrow metadata

Skeleton for the Kraken arrow -> PaddleOCR dataset converter. First function
extracts and codepoint-sorts the 118-char Burmese alphabet from the arrow
schema's `lines` metadata."
```

---

## Task 3: Label-line formatting + path-relativization tests

**Files:**
- Modify: `scripts/build_paddle_dataset.py`
- Modify: `scripts/build_paddle_dataset_test.py`

- [ ] **Step 1: Write failing tests for label-line + dict-line formatting**

Append to `scripts/build_paddle_dataset_test.py`:

```python
def test_format_label_line_uses_tab_separator():
    mod = _load_module()
    line = mod.format_label_line("train/000000001.png", "မင်္ဂလာပါ")
    assert line == "train/000000001.png\tမင်္ဂလာပါ\n"


def test_format_label_line_preserves_special_chars():
    mod = _load_module()
    line = mod.format_label_line("train/000000002.png", "ဈ = ၅")
    assert line == "train/000000002.png\tဈ = ၅\n"


def test_dict_line_is_one_char_plus_newline():
    mod = _load_module()
    assert mod.format_dict_line("\u1000") == "\u1000\n"
    assert mod.format_dict_line(" ") == " \n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest scripts/build_paddle_dataset_test.py -v`
Expected: 3 new tests FAIL with AttributeError (`format_label_line` / `format_dict_line` undefined).

- [ ] **Step 3: Implement the formatters**

Add to `scripts/build_paddle_dataset.py`:

```python
def format_label_line(rel_path: str, text: str) -> str:
    """One label-file row: `<rel_path><TAB><text><newline>`.

    PaddleOCR's SimpleDataSet splits on a literal tab; any other separator
    errors. Paths are relative to the config's `data_dir`.
    """
    return f"{rel_path}\t{text}\n"


def format_dict_line(char: str) -> str:
    """One dict-file row: the single char plus a newline.

    PaddleOCR appends the CTC blank and (if use_space_char) a space token
    automatically — do NOT add them here.
    """
    return f"{char}\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest scripts/build_paddle_dataset_test.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_paddle_dataset.py scripts/build_paddle_dataset_test.py
git commit -m "feat(rec-burmese): label-line and dict-line formatters"
```

---

## Task 4: OOV detection tests

OOV chars are silently dropped by CTCLabelEncode (warning commented out), producing silently corrupted labels. We detect and filter OOV before writing.

**Files:**
- Modify: `scripts/build_paddle_dataset.py`
- Modify: `scripts/build_paddle_dataset_test.py`

- [ ] **Step 1: Write failing tests for OOV detection**

Append to `scripts/build_paddle_dataset_test.py`:

```python
def test_find_oov_chars_returns_oov_set():
    mod = _load_module()
    oov = mod.find_oov_chars("ကခgaza", alphabet={"က", "ခ"})
    assert oov == {"g", "a", "z"}


def test_find_oov_chars_empty_when_all_in_alphabet():
    mod = _load_module()
    assert mod.find_oov_chars("ကခ", alphabet={"က", "ခ"}) == set()


def test_is_clean_returns_true_when_no_oov():
    mod = _load_module()
    assert mod.is_clean("ကခ", alphabet={"က", "ခ"}) is True


def test_is_clean_returns_false_when_oov():
    mod = _load_module()
    assert mod.is_clean("ကx", alphabet={"က"}) is False


def test_is_clean_empty_text_is_not_clean():
    """Empty transcriptions become empty labels — drop them."""
    mod = _load_module()
    assert mod.is_clean("", alphabet={"က"}) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest scripts/build_paddle_dataset_test.py -v`
Expected: 5 new tests FAIL with AttributeError.

- [ ] **Step 3: Implement OOV functions**

Add to `scripts/build_paddle_dataset.py`:

```python
def find_oov_chars(text: str, alphabet: set[str]) -> set[str]:
    """Return the set of chars in `text` not present in `alphabet`."""
    return {c for c in text if c not in alphabet}


def is_clean(text: str, alphabet: set[str]) -> bool:
    """True if text is non-empty and contains only in-alphabet chars.

    Empty texts are rejected (they would yield empty labels and be dropped
    downstream anyway, but we drop them explicitly with a count).
    """
    return len(text) > 0 and not find_oov_chars(text, alphabet)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest scripts/build_paddle_dataset_test.py -v`
Expected: 10 passed (5 prior + 5 new).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_paddle_dataset.py scripts/build_paddle_dataset_test.py
git commit -m "feat(rec-burmese): OOV detection to prevent silent label corruption

CTCLabelEncode silently drops out-of-vocab chars (its warning is commented
out), which corrupts labels with no signal. These helpers flag any
transcription containing chars outside the 118-char alphabet."
```

---

## Task 5: Full extraction pipeline (main) + end-to-end test on a tiny synthetic arrow

**Files:**
- Modify: `scripts/build_paddle_dataset.py`
- Modify: `scripts/build_paddle_dataset_test.py`

- [ ] **Step 1: Write an end-to-end test using a synthetic arrow file**

Append to `scripts/build_paddle_dataset_test.py`:

```python
def test_build_dataset_end_to_end(tmp_path):
    """Build a tiny synthetic arrow, run the full pipeline, assert outputs."""
    mod = _load_module()

    arrow_path = tmp_path / "synthetic.arrow"
    _write_synthetic_arrow(arrow_path)

    out_dir = tmp_path / "out"
    mod.build_dataset(
        arrow_path=str(arrow_path),
        out_dir=str(out_dir),
        max_per_split=10,
        smoke_per_split=2,
    )

    # Dict file: 3 chars (က ခ ဂ), one per line, codepoint-sorted.
    dict_text = (out_dir / "burmese_dict.txt").read_text(encoding="utf-8")
    assert dict_text == "\u1000\n\u1001\n\u1002\n"

    # train_list.txt: path<TAB>text, sorted by filename, relative to out_dir.
    train_list = (out_dir / "train_list.txt").read_text(encoding="utf-8")
    assert "train/000000001.png\tကခဂ\n" in train_list

    # PNG crop exists and is readable, grayscale.
    from PIL import Image
    img = Image.open(out_dir / "train" / "000000001.png")
    assert img.mode == "L"

    # Smoke subset is smaller than full.
    train_smoke = (out_dir / "train_list_smoke.txt").read_text(encoding="utf-8")
    assert len(train_smoke.splitlines()) <= len(train_list.splitlines())

    # OOV report written.
    oov_report = (out_dir / "oov_report.txt").read_text(encoding="utf-8")
    # The OOV sample 'X' should appear in the report.
    assert "X" in oov_report or "0 OOV" in oov_report


def _write_synthetic_arrow(path):
    """Write a minimal Kraken-format arrow for testing."""
    import pyarrow as pa
    from PIL import Image
    import io

    # 3 train rows (one with OOV char 'X'), 1 val, 1 test.
    def png_bytes():
        buf = io.BytesIO()
        Image.new("1", (10, 4), 1).save(buf, format="PNG")
        return buf.getvalue()

    rows = {
        "lines": [
            {"text": "\u1000\u1001\u1002", "im": png_bytes()},  # train clean
            {"text": "\u1000\u1000", "im": png_bytes()},         # train clean
            {"text": "\u1000X", "im": png_bytes()},              # train OOV ('X')
            {"text": "\u1001\u1002", "im": png_bytes()},         # val
            {"text": "\u1002", "im": png_bytes()},               # test
        ],
        "train":      [True, True, True, False, False],
        "validation": [False, False, False, True, False],
        "test":       [False, False, False, False, True],
    }
    table = pa.table(rows)
    # Attach alphabet metadata matching the real Kraken layout.
    meta = {
        b"lines": (
            b'{"type":"kraken_recognition_bbox","alphabet":'
            b'{"\u1000":1,"\u1001":1,"\u1002":1}}'
        )
    }
    table = table.replace_schema_metadata(meta)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest scripts/build_paddle_dataset_test.py::test_build_dataset_end_to_end -v`
Expected: FAIL — `build_dataset` undefined.

- [ ] **Step 3: Implement `build_dataset`**

Add to `scripts/build_paddle_dataset.py`:

```python
SPLITS = [("train", "train"), ("validation", "val"), ("test", "test")]


def build_dataset(arrow_path: str, out_dir: str, max_per_split: int | None = None,
                  smoke_per_split: int | None = None) -> None:
    """Read arrow, write dict + label files + PNG crops + OOV report.

    Splits come from the arrow's train/validation/test bool columns (not
    shuffled) so the held-out test set matches the Kraken benchmarks.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reader = ipc.open_file(arrow_path)
    table = reader.read_all()
    metadata = reader.schema.metadata

    chars = extract_alphabet_chars(metadata)
    alphabet = set(chars)

    # 1. Write dict file.
    (out / "burmese_dict.txt").write_text(
        "".join(format_dict_line(c) for c in chars), encoding="utf-8"
    )

    # 2. Iterate rows, split by bool column, write crops + label lists.
    lines_col = table.column("lines").to_pylist()
    train_mask = table.column("train").to_pylist()
    val_mask = table.column("validation").to_pylist()
    test_mask = table.column("test").to_pylist()

    split_masks = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }
    split_dirs = {"train": "train", "val": "val", "test": "test"}

    oov_counts: dict[str, int] = {}
    dropped_counts: dict[str, int] = {}

    for split_name in ("train", "val", "test"):
        # Iterate rows in arrow order; filter by this split's mask.
        idx_in_split = 0
        full_lines: list[str] = []
        smoke_lines: list[str] = []
        n_full = 0
        n_smoke = 0

        for row_idx, in_split in enumerate(split_masks[split_name]):
            if not in_split:
                continue
            if max_per_split is not None and n_full >= max_per_split:
                break

            text = lines_col[row_idx]["text"]
            img_bytes = lines_col[row_idx]["im"]

            if not is_clean(text, alphabet):
                oov = find_oov_chars(text, alphabet)
                for c in oov:
                    oov_counts[c] = oov_counts.get(c, 0) + 1
                dropped_counts[split_name] = dropped_counts.get(split_name, 0) + 1
                continue

            # Zero-padded filename, deterministic.
            fname = f"{idx_in_split + 1:09d}.png"

            # Write crop for all splits (train/val for training; test for eval).
            subdir = split_dirs[split_name]
            crop_path = out / subdir / fname
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            _write_grayscale_png(img_bytes, crop_path)
            rel_path = f"{subdir}/{fname}"

            line = format_label_line(rel_path, text)
            full_lines.append(line)
            n_full += 1

            if smoke_per_split is not None and n_smoke < smoke_per_split:
                smoke_lines.append(line)
                n_smoke += 1

            idx_in_split += 1

        (out / f"{split_name}_list.txt").write_text(
            "".join(full_lines), encoding="utf-8"
        )
        if smoke_per_split is not None:
            (out / f"{split_name}_list_smoke.txt").write_text(
                "".join(smoke_lines), encoding="utf-8"
            )

    # 3. OOV report.
    _write_oov_report(out, oov_counts, dropped_counts, len(lines_col))

    # 4. Summary to stdout.
    print(f"Dict: {len(chars)} chars -> {out / 'burmese_dict.txt'}")
    for split_name in ("train", "val", "test"):
        n = len((out / f"{split_name}_list.txt").read_text(encoding="utf-8").splitlines())
        print(f"  {split_name}: {n} lines")
    print(f"  dropped (OOV/empty): {sum(dropped_counts.values())}")


def _write_grayscale_png(png_bytes: bytes, dest: Path) -> None:
    """Decode the 1-bit Kraken PNG and re-encode as 8-bit grayscale (L mode)."""
    with Image.open(io.BytesIO(png_bytes)) as img:
        img.convert("L").save(dest, format="PNG")


def _write_oov_report(out: Path, oov_counts: dict[str, int],
                      dropped_counts: dict[str, int], total_rows: int) -> None:
    """Write a human-readable report of dropped samples and OOV char frequencies."""
    lines = [f"OOV / dropped-sample report (out of {total_rows} arrow rows)\n",
             f"total dropped: {sum(dropped_counts.values())}\n"]
    for split, n in sorted(dropped_counts.items()):
        lines.append(f"  {split}: {n} dropped\n")
    lines.append("\nOOV char frequencies (char -> count across dropped samples):\n")
    for c, n in sorted(oov_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {c!r} -> {n}\n")
    if not oov_counts:
        lines.append("  (none — all transcriptions are in-alphabet)\n")
    (out / "oov_report.txt").write_text("".join(lines), encoding="utf-8")
```

Also add the `__main__` entry point at the bottom of `scripts/build_paddle_dataset.py`:

```python
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arrow", default=DEFAULT_ARROW, help="Kraken .arrow file path")
    p.add_argument("--out", default="train_data/burmese_rec", help="output dir")
    p.add_argument("--max-per-split", type=int, default=None,
                   help="cap rows per split (for smoke extraction)")
    p.add_argument("--smoke-per-split", type=int, default=None,
                   help="also write *_list_smoke.txt with this many rows per split")
    args = p.parse_args()
    build_dataset(args.arrow, args.out, args.max_per_split, args.smoke_per_split)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest scripts/build_paddle_dataset_test.py -v`
Expected: all tests pass, including the end-to-end test.

- [ ] **Step 5: Run the script for real on a tiny smoke subset**

Smoke-extract 100 train / 20 val rows to confirm it works on the real arrow:

```bash
.venv/bin/python scripts/build_paddle_dataset.py \
    --max-per-split 100 --smoke-per-split 20
```

Expected: prints a summary; `train_data/burmese_rec/burmese_dict.txt` has 118 lines; `train_list.txt` has ≤100 lines; a few PNGs exist under `train/` and `val/`; `oov_report.txt` shows few or zero drops.

Spot-check one label line manually:

```bash
head -1 train_data/burmese_rec/train_list.txt | cat -A | head -1
```

Expected output contains a literal tab (shown as `^I` by `cat -A`) between path and Burmese text.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_paddle_dataset.py scripts/build_paddle_dataset_test.py
git commit -m "feat(rec-burmese): full arrow->PaddleOCR extraction pipeline

Reads the Kraken arrow, preserves train/val/test split columns, writes
grayscale PNG crops + label lists + 118-char dict + OOV report. Supports
--max-per-split for smoke extraction."
```

---

## Task 6: Write the Burmese-tuned config

Clone `PP-OCRv6_small_rec.yml` and apply the six geometry/LR/dict/pretrained changes from the spec. No tests (config is declarative) — verified by the smoke run in Task 8.

**Files:**
- Create: `configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml`

- [ ] **Step 1: Read the stock config to clone from**

Run: `cat configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml`
(Reference only — copy structure into the new file in Step 2.)

- [ ] **Step 2: Create the Burmese config**

Create `configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml`:

```yaml
Global:
  model_name: burmese_PP-OCRv6_small_rec
  debug: false
  use_gpu: true
  epoch_num: 30
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output/burmese_PP-OCRv6_small_rec
  save_epoch_step: 5
  eval_batch_step: [0, 2000]
  cal_metric_during_train: true
  pretrained_model: ./PP-OCRv6_small_rec_pretrained
  checkpoints:
  save_inference_dir:
  use_visualdl: false
  infer_img: doc/imgs_words/ch/word_1.jpg
  character_dict_path: train_data/burmese_rec/burmese_dict.txt
  max_text_length: &max_text_length 100
  infer_mode: false
  use_space_char: true
  distributed: true
  save_res_path: ./output/rec/predicts_burmese_ppocrv6_small.txt
  d2s_train_image_shape: [3, 48, 640]


Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0001
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 3.0e-05


Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: PPLCNetV4
    model_size: small
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: lightsvtr
            dims: 120
            depth: 2
            mlp_ratio: 2.0
            local_kernel: 7
          Head:
            fc_decay: 0.00001
      - NRTRHead:
          nrtr_dim: 384
          max_text_length: *max_text_length

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - NRTRLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: MultiScaleDataSet
    ds_width: false
    data_dir: ./train_data/burmese_rec/
    ext_op_transform_idx: 1
    label_file_list:
    - ./train_data/burmese_rec/train_list.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - RecConAug:
        prob: 0.5
        ext_data_num: 2
        image_shape: [48, 640, 3]
        max_text_length: *max_text_length
    - RecAug:
    - MultiLabelEncode:
        gtc_encode: NRTRLabelEncode
    - KeepKeys:
        keep_keys:
        - image
        - label_ctc
        - label_gtc
        - length
        - valid_ratio
  sampler:
    name: MultiScaleSampler
    scales: [[640, 48], [640, 64], [640, 80]]
    first_bs: &bs 128
    fix_bs: false
    divided_factor: [8, 16]
    is_training: True
  loader:
    shuffle: true
    batch_size_per_card: *bs
    drop_last: true
    num_workers: 8

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/burmese_rec
    label_file_list:
    - ./train_data/burmese_rec/val_list.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - MultiLabelEncode:
        gtc_encode: NRTRLabelEncode
    - RecResizeImg:
        image_shape: [3, 48, 640]
    - KeepKeys:
        keep_keys:
        - image
        - label_ctc
        - label_gtc
        - length
        - valid_ratio
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 128
    num_workers: 4
```

- [ ] **Step 3: Diff-check against the stock config**

Run:

```bash
diff configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
     configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml
```

Expected: only the planned fields differ (`character_dict_path`, `max_text_length`, `d2s_train_image_shape`, `image_shape` in RecConAug/RecResizeImg, `scales`, `learning_rate`, `epoch_num`, `save_model_dir`, `save_res_path`, `pretrained_model`, `model_name`, `save_epoch_step`, label_file_list / data_dir paths). No structural changes.

- [ ] **Step 4: Commit**

```bash
git add configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml
git commit -m "feat(rec-burmese): PP-OCRv6 small config tuned for Burmese

Clone of PP-OCRv6_small_rec.yml with: custom 118-char dict, max_text_length
100 (median line 53 chars), image width 640 (median 557px), LR 1e-4 for
single-GPU fine-tune, official v6 small rec pretrained weights."
```

---

## Task 7: Write the training orchestrator

**Files:**
- Create: `scripts/train_burmese.sh`

- [ ] **Step 1: Create the orchestrator script**

Create `scripts/train_burmese.sh`:

```bash
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
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/train_burmese.sh`
Expected: no output.

- [ ] **Step 3: Sanity-check usage message**

Run: `bash scripts/train_burmese.sh`
Expected: prints `Usage: bash scripts/train_burmese.sh {smoke|full|export|eval}` to stderr and exits 1.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_burmese.sh
git commit -m "feat(rec-burmese): training orchestrator (smoke/full/export/eval)"
```

---

## Task 8: Download pretrained weights, extract smoke data, run the smoke test

This task validates the entire pipeline end-to-end on the real arrow + real config before committing GPU hours to the full run.

**Files:** none (operational only)

- [ ] **Step 1: Download the official PP-OCRv6 small rec pretrained checkpoint**

```bash
curl -L -o PP-OCRv6_small_rec_pretrained.pdparams \
  https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams
```

Expected: file `PP-OCRv6_small_rec_pretrained.pdparams` (~tens of MB) at repo root. (This path matches `Global.pretrained_model: ./PP-OCRv6_small_rec_pretrained` in the config.)

- [ ] **Step 2: Add the pretrained file to gitignore**

Append to `.gitignore`:

```
# Burmese rec fine-tune artifacts (large / regeneratable)
PP-OCRv6_small_rec_pretrained.pdparams
train_data/burmese_rec/
output/burmese_PP-OCRv6_small_rec/
models/burmese_PP-OCRv6_small_rec_infer/
```

- [ ] **Step 3: Extract the smoke subset (50K train / 5K val)**

```bash
.venv/bin/python scripts/build_paddle_dataset.py \
    --max-per-split 50000 --smoke-per-split 5000
```

Expected: prints a summary; `train_list.txt` has ≤50,000 lines (capped, full set extracted later in Task 9); `train_list_smoke.txt` has 5,000. `burmese_dict.txt` has 118 lines. OOV report shows few/no drops (the `_clean` dataset was pre-filtered by Kraken).

Note on extraction semantics: `--max-per-split` caps **all** splits (train, val, test), so this command also caps val/test at 50K. That's fine — the smoke *training* run only uses the `_smoke` files (`val_list_smoke.txt`, 5K). The full run in Task 9 re-runs extraction WITHOUT `--max-per-split` to get the uncapped val/test for production eval.

- [ ] **Step 4: Run the smoke training**

```bash
bash scripts/train_burmese.sh smoke
```

Watch the console for these **five success criteria** (from the spec). All must pass before scaling up:

1. **No crash through ≥2000 iterations** — gets past `eval_batch_step: [0, 2000]`, the first eval fires.
2. **Loss decreases** — both `CTCLoss` and `NRTRLoss` components trending down, never NaN/inf.
3. **First eval completes and reports non-zero acc** (5–20% is fine — confirms the head is learning, not stuck at iter-0 zero).
4. **GPU memory headroom noted** — record peak memory; this decides whether to widen to 800 (see Step 6).
5. **Dict/head wiring verified** — the log's `out_channels` equals `118 + 1 (blank) + 1 (space) = 120`.

- [ ] **Step 5: If any criterion fails, debug before proceeding**

Common failures and fixes:
- **Crash on "length of dir should not be equal to 0"** → `train_list_smoke.txt` paths don't resolve under `data_dir`. Verify `train/000000001.png` exists relative to `./train_data/burmese_rec/`.
- **acc stuck at 0 past eval** → pretrained weights didn't load; check the log for "not in pretrained model" warnings and confirm `PP-OCRv6_small_rec_pretrained.pdparams` is at repo root.
- **`out_channels` != 120** → dict has wrong line count; run `wc -l train_data/burmese_rec/burmese_dict.txt` (must be 118).
- **NaN loss** → LR too high or mixed-precision issue; retry with `learning_rate: 0.00005`.

- [ ] **Step 6: Decide on image width (spec's open knob)**

Based on the GPU memory headroom observed in Step 4:
- If peak memory is well under the GPU's limit (e.g. <70% on an 8GB card, <60% on larger), bump `d2s_train_image_shape` / `RecResizeImg.image_shape` / `RecConAug.image_shape` to `[3,48,800]` and `scales` widths to 800. Re-run smoke. This keeps ~95% of crops undistorted.
- If memory is tight, keep 640 and note the tradeoff in the final report.

This is a judgment call at execution time — record the decision and the memory number.

- [ ] **Step 7: Commit the gitignore change**

```bash
git add .gitignore
git commit -m "chore(rec-burmese): gitignore pretrained weights + dataset artifacts"
```

---

## Task 9: Run the full training

**Files:** none (operational)

- [ ] **Step 1: Extract the full dataset**

```bash
.venv/bin/python scripts/build_paddle_dataset.py --smoke-per-split 5000
```

(No `--max-per-split` → full 718,515 train / 89,625 val / 89,743 test. `--smoke-per-split` keeps the smoke files for any re-runs.) Expected: `train_list.txt` ~718K lines, val ~89K, test ~89K. Disk ~1.9GB PNGs (train+val+test crops).

- [ ] **Step 2: Launch full training**

```bash
bash scripts/train_burmese.sh full
```

Expected: 30 epochs of Cosine LR (warmup 5), periodic checkpoints + `best_accuracy.pdparams` saved to `output/burmese_PP-OCRv6_small_rec/`. Eval acc should climb steadily past the smoke-run number.

- [ ] **Step 3: Monitor and record final val acc**

When training completes, note the final `best_accuracy` from the eval logs. This is the production val-set accuracy.

---

## Task 10: Export and final eval

**Files:** none (operational; produces the deliverable)

- [ ] **Step 1: Export the best checkpoint to an inference model**

```bash
bash scripts/train_burmese.sh export
```

Expected: writes `models/burmese_PP-OCRv6_small_rec_infer/inference.pdmodel` and `inference.pdiparams`. Verify:

```bash
ls -la models/burmese_PP-OCRv6_small_rec_infer/
```

Expected: both files present, non-zero size.

- [ ] **Step 2: Run final eval on the held-out test split**

```bash
bash scripts/train_burmese.sh eval
```

Expected: prints final line-level accuracy and CER on the test split (89,743 images, never seen in training). This number is directly comparable to the Kraken benchmarks since the splits come from the same arrow `test` column.

- [ ] **Step 3: Record the result**

Note the test-split accuracy and CER. The deliverable is `models/burmese_PP-OCRv6_small_rec_infer/` — drop-in compatible with the existing PP-OCRv6 detector.

---

## Done criteria

- [ ] `scripts/build_paddle_dataset.py` + tests pass, real extraction produces 118-char dict + ≤718K train lines.
- [ ] `configs/rec/PP-OCRv6/burmese_PP-OCRv6_small_rec.yml` present and diff-clean against stock.
- [ ] `scripts/train_burmese.sh` runs all four modes.
- [ ] Smoke run passes all five success criteria.
- [ ] Full training completes; `best_accuracy.pdparams` saved.
- [ ] Inference model exported to `models/burmese_PP-OCRv6_small_rec_infer/`.
- [ ] Test-split CER recorded.
