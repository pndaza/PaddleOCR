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
            rb'{"type":"kraken_recognition_bbox","alphabet":'
            rb'{"\u1040":1,"\u1000":2,"\u1021":3}}'
        )
    }
    chars = mod.extract_alphabet_chars(fake_meta)
    # sorted by Unicode codepoint: U+1000, U+1021, U+1040
    assert chars == ["\u1000", "\u1021", "\u1040"]


def test_extract_alphabet_char_count():
    mod = _load_module()
    fake_meta = {
        b"lines": (
            rb'{"type":"kraken_recognition_bbox","alphabet":'
            rb'{"\u1000":2,"\u1001":5}}'
        )
    }
    chars = mod.extract_alphabet_chars(fake_meta)
    assert len(chars) == 2


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
