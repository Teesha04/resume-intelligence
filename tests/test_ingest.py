"""Ingestion reliability: isolation, dedupe, unsupported types."""

from pathlib import Path

from src.ingest import discover_files, load_documents, parse_file
from src.schemas import ParseStatus


def test_parse_txt(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("Jane Doe\nPython, RAG", encoding="utf-8")
    doc = parse_file(f)
    assert doc.parse_status == ParseStatus.OK
    assert "Python" in doc.text
    assert doc.content_hash


def test_missing_file_does_not_raise(tmp_path: Path):
    doc = parse_file(tmp_path / "ghost.pdf")
    assert doc.parse_status == ParseStatus.FAILED
    assert doc.error


def test_unsupported_extension_is_failed(tmp_path: Path):
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG")
    assert parse_file(f).parse_status == ParseStatus.FAILED


def test_corrupt_pdf_isolated(tmp_path: Path):
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"%PDF-1.4 not actually a pdf \x00\x01")
    doc = parse_file(f)  # must not raise
    assert doc.parse_status == ParseStatus.FAILED


def test_empty_file_marked_empty(tmp_path: Path):
    f = tmp_path / "blank.txt"
    f.write_text("   \n\n", encoding="utf-8")
    assert parse_file(f).parse_status == ParseStatus.EMPTY


def test_duplicate_content_is_detected(tmp_path: Path):
    (tmp_path / "one.txt").write_text("Same content here", encoding="utf-8")
    (tmp_path / "two.txt").write_text("Same content here", encoding="utf-8")
    docs, duplicates = load_documents(tmp_path)
    assert duplicates == 1
    assert sum(1 for d in docs if d.parse_status == ParseStatus.DUPLICATE) == 1


def test_discover_skips_unsupported_and_hidden(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("hi")
    (tmp_path / "skip.png").write_bytes(b"x")
    (tmp_path / ".DS_Store").write_bytes(b"x")
    files, skipped = discover_files(tmp_path)
    names = {f.name for f in files}
    assert names == {"ok.txt"}
    assert "skip.png" in skipped


def test_missing_input_dir_raises_for_caller():
    import pytest

    with pytest.raises(FileNotFoundError):
        discover_files(Path("/no/such/dir/xyz"))
