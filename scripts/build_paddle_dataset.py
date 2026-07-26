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


def find_oov_chars(text: str, alphabet: set[str]) -> set[str]:
    """Return the set of chars in `text` not present in `alphabet`."""
    return {c for c in text if c not in alphabet}


def is_clean(text: str, alphabet: set[str]) -> bool:
    """True if text is non-empty and contains only in-alphabet chars.

    Empty texts are rejected (they would yield empty labels and be dropped
    downstream anyway, but we drop them explicitly with a count).
    """
    return len(text) > 0 and not find_oov_chars(text, alphabet)


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
