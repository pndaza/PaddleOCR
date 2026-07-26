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
