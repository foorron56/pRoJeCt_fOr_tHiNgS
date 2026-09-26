"""Tests cho scripts/pick_video.py - bo chon video ngau nhien cho menu run.bat.

Run: pytest -q
"""
from __future__ import annotations

import os

from scripts import pick_video


def _make_data(tmp_path, names):
    d = tmp_path / "data"
    d.mkdir()
    for n in names:
        (d / n).write_bytes(b"")
    return d


def test_list_clips_skips_browser_duplicates(tmp_path, monkeypatch):
    """Ban sao "abc (1).mp4" cua trinh duyet phai bi loai; con lai giu nguyen."""
    d = _make_data(tmp_path, ["clip.mp4", "clip (1).mp4",
                              "clip (11).mp4", "other.mp4"])
    monkeypatch.setenv("LIFESTREAM_DATA_DIR", str(d))
    clips = pick_video.list_clips()
    # "(1)" bi loai, "(11)" la ten that nen duoc giu
    assert clips == [str(d / "clip (11).mp4"), str(d / "clip.mp4"),
                     str(d / "other.mp4")]


def test_default_data_dir_is_project_data(monkeypatch):
    """Khong dat bien moi truong -> mac dinh la <goc du an>/data."""
    monkeypatch.delenv("LIFESTREAM_DATA_DIR", raising=False)
    assert pick_video.data_dir() == pick_video.DEFAULT_DATA_DIR
    assert os.path.normpath(pick_video.data_dir()) == os.path.normpath(
        os.path.join(pick_video.PROJECT_ROOT, "data"))


def test_main_prints_random_clip_from_data(tmp_path, monkeypatch, capsys):
    """Moi lan chon phai la 1 duong dan hop le trong danh sach clip."""
    d = _make_data(tmp_path, ["a.mp4", "b.mp4", "c.mp4"])
    monkeypatch.setenv("LIFESTREAM_DATA_DIR", str(d))
    clips = {str(d / n) for n in ("a.mp4", "b.mp4", "c.mp4")}
    for _ in range(30):
        assert pick_video.main() == 0
        out = capsys.readouterr().out.strip()
        assert out in clips


def test_main_reports_when_no_clip_available(tmp_path, monkeypatch, capsys):
    """data/ trong hoac khong ton tai -> thoat ma loi, khong in duong dan."""
    d = _make_data(tmp_path, [])  # thu muc data ton tai nhung trong
    monkeypatch.setenv("LIFESTREAM_DATA_DIR", str(d))
    assert pick_video.main() == 1
    res = capsys.readouterr()
    assert res.out.strip() == ""
    assert "LIFESTREAM_LOI" in res.err

    monkeypatch.setenv("LIFESTREAM_DATA_DIR", str(tmp_path / "khong-ton-tai"))
    assert pick_video.main() == 1
    res = capsys.readouterr()
    assert res.out.strip() == ""
    assert "LIFESTREAM_LOI" in res.err


def test_list_mode_prints_numbered_lines(tmp_path, monkeypatch, capsys):
    """--list in moi video mot dong 'i|duong_dan', so thu tu tinh tu 1."""
    d = _make_data(tmp_path, ["b.mp4", "a.mp4"])
    monkeypatch.setenv("LIFESTREAM_DATA_DIR", str(d))
    assert pick_video.main(["--list"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["1|%s" % (d / "a.mp4"), "2|%s" % (d / "b.mp4")]


def test_path_mode_uses_pick_index_env(tmp_path, monkeypatch, capsys):
    """--path doc LIFESTREAM_PICK_INDEX (1-based) va in dung video."""
    d = _make_data(tmp_path, ["b.mp4", "a.mp4", "c.mp4"])
    monkeypatch.setenv("LIFESTREAM_DATA_DIR", str(d))
    for idx, name in ((1, "a.mp4"), (3, "c.mp4")):
        monkeypatch.setenv("LIFESTREAM_PICK_INDEX", str(idx))
        assert pick_video.main(["--path"]) == 0
        assert capsys.readouterr().out.strip() == str(d / name)


def test_path_mode_rejects_invalid_index(tmp_path, monkeypatch, capsys):
    """--path voi index sai (ngoai pham vi / khong phai so) -> bao loi."""
    d = _make_data(tmp_path, ["a.mp4"])
    monkeypatch.setenv("LIFESTREAM_DATA_DIR", str(d))
    for bad in ("0", "2", "abc", ""):
        monkeypatch.setenv("LIFESTREAM_PICK_INDEX", bad)
        assert pick_video.main(["--path"]) == 1
        res = capsys.readouterr()
        assert res.out.strip() == ""
        assert "LIFESTREAM_LOI" in res.err
