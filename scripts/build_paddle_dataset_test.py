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
