from pathlib import Path

import pytest

from praxis_agent.tools.code_output import CodeOutput


@pytest.mark.parametrize("size", [0, 5999, 6000, 6001, 29999, 30000, 30001, 100000])
@pytest.mark.parametrize("chunk_size", [137, 100000])
def test_preview_and_original_match_for_unicode_and_chunk_boundaries(tmp_path, size, chunk_size):
    text = ("中文🙂\r\n0123456789" * 10000)[:size]
    output = CodeOutput(tmp_path / "logs")
    for offset in range(0, len(text), chunk_size):
        output.append("stdout", text[offset:offset + chunk_size])
    output.close()
    output.close()
    fields = output.fields()
    assert fields["stderr"] == ""
    if size <= 30000:
        assert fields["stdout"] == text
        assert fields["output_files"] == {}
        assert not (tmp_path / "logs").exists()
    else:
        assert fields["stdout"] == text[:6000] + f"\n... [{size - 30000} characters omitted] ...\n" + text[-24000:]
        reference = fields["output_files"]["stdout"]
        assert reference["total_chars"] == size
        assert reference["omitted_chars"] == size - 30000
        assert reference["complete"] is True
        assert Path(reference["path"]).read_bytes().decode("utf-8") == text


def test_close_failure_marks_log_incomplete(tmp_path, monkeypatch):
    original_open = Path.open
    class FailingClose:
        def __init__(self, file):
            self.file = file
        def write(self, text):
            return self.file.write(text)
        def close(self):
            self.file.close()
            raise OSError("disk flush failed")

    monkeypatch.setattr(Path, "open", lambda path, *args, **kwargs:
                        FailingClose(original_open(path, *args, **kwargs)))
    output = CodeOutput(tmp_path)
    output.append("stdout", "a" * 30001)
    output.append("stderr", "b" * 30001)
    output.close()
    fields = output.fields()
    assert set(fields["output_errors"]) == {"stdout", "stderr"}
    assert all(not ref["complete"] for ref in fields["output_files"].values())
