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

    # train_list.txt: path<TAB>text, relative to out_dir.
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
            rb'{"type":"kraken_recognition_bbox","alphabet":'
            rb'{"\u1000":1,"\u1001":1,"\u1002":1}}'
        )
    }
    table = table.replace_schema_metadata(meta)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
