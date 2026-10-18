import codecs
from types import SimpleNamespace

import pytest

from praxis_agent.tools import edit_file, read_file, write_file


def modify(workspace, operation, before, after):
    if operation == "edit":
        return edit_file(workspace, "notes.txt", [{"old_text": before, "new_text": after}])
    return write_file(workspace, "notes.txt", after)


@pytest.mark.parametrize("operation", ["edit", "write"])
@pytest.mark.parametrize("encoding,bom", [
    ("utf-8", b""), ("utf-8", codecs.BOM_UTF8),
    ("utf-16-le", codecs.BOM_UTF16_LE), ("utf-16-be", codecs.BOM_UTF16_BE),
    ("utf-32-le", codecs.BOM_UTF32_LE), ("utf-32-be", codecs.BOM_UTF32_BE),
])
def test_edit_and_write_preserve_encoding_bom_and_crlf(tmp_path, operation, encoding, bom):
    original = "标题\r\n原始内容\r\n最后一行"
    updated = "标题\n修改内容\n最后一行"
    path = tmp_path / "notes.txt"
    path.write_bytes(bom + original.encode(encoding))

    assert read_file(tmp_path, "notes.txt")["text"] == original
    change = modify(tmp_path, operation, original, updated)

    assert path.read_bytes() == bom + updated.replace("\n", "\r\n").encode(encoding)
    assert change.path == "notes.txt"
    assert "-原始内容" in change.diff
    assert "+修改内容" in change.diff


@pytest.mark.parametrize("operation", ["edit", "write"])
@pytest.mark.parametrize("encoding,original", [
    ("gbk", "文件编码测试：你好，世界。\r\n原始内容。\r\n第二行内容。\r\n"),
    ("big5", "檔案編碼測試：你好，世界。\r\n原始內容。\r\n第二行內容。\r\n"),
])
def test_edit_and_write_detect_legacy_chinese_encodings(tmp_path, operation, encoding, original):
    path = tmp_path / "notes.txt"
    path.write_bytes(original.encode(encoding))
    updated = original.replace("原始", "修改")

    assert read_file(tmp_path, "notes.txt")["text"] == original
    modify(tmp_path, operation, original, updated)

    assert path.read_bytes() == updated.encode(encoding)


@pytest.mark.parametrize("operation", ["edit", "write"])
@pytest.mark.parametrize("original,expected", [
    ("prefix\nold\nend\n", "prefix\nnew\nextra\nend\n"),
    ("prefix\r\nold\r\nend\r\n", "prefix\r\nnew\r\nextra\r\nend\r\n"),
    ("prefix\r\nold\nend\r\n", "prefix\nnew\nextra\nend\n"),
    ("prefix\nold\r\nend\n", "prefix\nnew\nextra\nend\n"),
])
def test_newline_policy_applies_to_matching_replacements_and_the_whole_file(tmp_path, operation, original, expected):
    path = tmp_path / "notes.txt"
    path.write_bytes(original.encode("utf-8"))

    if operation == "edit":
        edit_file(tmp_path, "notes.txt", [{"old_text": "old\r\n", "new_text": "new\r\nextra\n"}])
    else:
        write_file(tmp_path, "notes.txt", "prefix\r\nnew\nextra\r\nend\n")

    assert path.read_bytes() == expected.encode("utf-8")


@pytest.mark.parametrize("operation", ["edit", "write"])
@pytest.mark.parametrize("original", ["old", "prefix\nold", "prefix\r\nold"])
def test_edit_and_write_do_not_add_a_final_newline(tmp_path, operation, original):
    path = tmp_path / "notes.txt"
    path.write_bytes(original.encode("utf-8"))
    updated = original.replace("old", "new")

    modify(tmp_path, operation, original, updated)

    assert path.read_bytes() == updated.encode("utf-8")


@pytest.mark.parametrize("operation", ["edit", "write"])
def test_ascii_files_can_receive_unicode_without_changing_format(tmp_path, operation):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"old\r\n")

    modify(tmp_path, operation, "old\n", "中文 😀\n")

    assert path.read_bytes() == "中文 😀\r\n".encode("utf-8")


@pytest.mark.parametrize("operation", ["edit", "write"])
def test_utf8_character_crossing_the_detection_sample_boundary_is_preserved(tmp_path, operation):
    original = "a" * 4095 + "你好\nold"
    path = tmp_path / "notes.txt"
    path.write_bytes(original.encode("utf-8"))
    updated = original.replace("old", "修改")

    modify(tmp_path, operation, original, updated)

    assert path.read_bytes() == updated.encode("utf-8")


@pytest.mark.parametrize("content,expected", [
    ("你好\r\n世界\r\n", "你好\n世界\n"), ("content", "content"), ("", ""),
])
def test_new_files_use_utf8_without_bom_and_lf(tmp_path, content, expected):
    write_file(tmp_path, "nested/notes.txt", content)
    assert (tmp_path / "nested/notes.txt").read_bytes() == expected.encode("utf-8")


def test_empty_existing_file_uses_utf8_and_lf(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"")

    write_file(tmp_path, "notes.txt", "你好\r\n")

    assert path.read_bytes() == "你好\n".encode("utf-8")


@pytest.mark.parametrize("operation", ["edit", "write"])
@pytest.mark.parametrize("original", [
    codecs.BOM_UTF8 + b"a" * 5000 + b"\xff",
    codecs.BOM_UTF16_LE + b"a",
])
def test_invalid_encoded_files_are_not_modified(tmp_path, operation, original):
    path = tmp_path / "notes.txt"
    path.write_bytes(original)

    with pytest.raises(UnicodeDecodeError):
        modify(tmp_path, operation, "a", "new")

    assert path.read_bytes() == original


@pytest.mark.parametrize("operation", ["edit", "write"])
def test_unencodable_replacements_do_not_truncate_the_original(tmp_path, monkeypatch, operation):
    monkeypatch.setattr("praxis_agent.tools.text_file.from_bytes",
                        lambda _: SimpleNamespace(best=lambda: SimpleNamespace(encoding="gbk")))
    original = "原始内容\r\n".encode("gbk")
    path = tmp_path / "notes.txt"
    path.write_bytes(original)

    with pytest.raises(UnicodeEncodeError):
        modify(tmp_path, operation, "原始内容\n", "新内容 😀\n")

    assert path.read_bytes() == original


def test_unknown_encoding_falls_back_to_utf8_without_replacing_invalid_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr("praxis_agent.tools.text_file.from_bytes", lambda _: SimpleNamespace(best=lambda: None))
    path = tmp_path / "notes.txt"
    original = b"\xfforiginal"
    path.write_bytes(original)

    with pytest.raises(UnicodeDecodeError):
        write_file(tmp_path, "notes.txt", "new")

    assert path.read_bytes() == original


def test_encoding_detection_samples_the_prefix_but_decodes_the_complete_file(tmp_path, monkeypatch):
    samples = []

    def detect(sample):
        samples.append(sample)
        return SimpleNamespace(best=lambda: SimpleNamespace(encoding="gbk"))

    monkeypatch.setattr("praxis_agent.tools.text_file.from_bytes", detect)
    original = "中文\r\n" + "a" * 5000 + "\r\n末尾"
    raw = original.encode("gbk")
    (tmp_path / "notes.txt").write_bytes(raw)

    result = read_file(tmp_path, "notes.txt", character_limit=None)

    assert samples == [raw[:4096]]
    assert result["text"] == original


def test_a_failed_later_edit_keeps_original_bytes_including_mixed_newlines(tmp_path):
    raw = codecs.BOM_UTF16_BE + "one\r\ntwo\nthree".encode("utf-16-be")
    path = tmp_path / "notes.txt"
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="edit 2 old_text was not found"):
        edit_file(tmp_path, "notes.txt", [
            {"old_text": "one\n", "new_text": "first\n"},
            {"old_text": "absent", "new_text": "last"},
        ])

    assert path.read_bytes() == raw


def test_each_edit_reads_current_format_instead_of_caching_it(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(codecs.BOM_UTF16_LE + "one\r\n".encode("utf-16-le"))
    edit_file(tmp_path, "notes.txt", [{"old_text": "one", "new_text": "two"}])
    assert path.read_bytes() == codecs.BOM_UTF16_LE + "two\r\n".encode("utf-16-le")

    path.write_bytes(b"two\n")
    edit_file(tmp_path, "notes.txt", [{"old_text": "two", "new_text": "three"}])
    assert path.read_bytes() == b"three\n"
