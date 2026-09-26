#!/usr/bin/env python
"""Cong cu chon video cho menu run.bat.

Che do mac dinh : in 1 duong dan .mp4 NGAU NHIEN tu data/ ra stdout.
--list          : in danh sach so thu tu, moi dong "i|duong_dan" (i tinh tu 1).
--path          : in duong dan video thu LIFESTREAM_PICK_INDEX (1-based).

Thong bao loi in ra stderr va thoat voi ma 1 khi khong co video nao.
Dat bien moi truong LIFESTREAM_DATA_DIR de tro sang thu muc khac (dung khi test).
"""
import argparse
import glob
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def data_dir():
    """Thu muc data hien hanh - doc env moi lan goi (de test ghi de duoc)."""
    return os.environ.get("LIFESTREAM_DATA_DIR") or DEFAULT_DATA_DIR


def list_clips():
    """Tat ca .mp4 trong data/, loai bo ban sao "abc (1).mp4" cua trinh duyet."""
    clips = [p for p in glob.glob(os.path.join(data_dir(), "*.mp4"))
             if not os.path.basename(p).rsplit(".", 1)[0].strip().endswith("(1)")]
    return sorted(clips)


def _fail(msg):
    print("[LIFESTREAM_LOI] %s" % msg, file=sys.stderr)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Chon video tu data/ cho menu run.bat")
    ap.add_argument("--list", action="store_true",
                    help="in danh sach video: moi dong 'i|duong_dan'")
    ap.add_argument("--path", action="store_true",
                    help="in video thu LIFESTREAM_PICK_INDEX (1-based)")
    args = ap.parse_args(argv)

    if not os.path.isdir(data_dir()):
        return _fail("Khong tim thay thu muc data/")
    clips = list_clips()
    if not clips:
        return _fail("Thu muc data/ khong co video .mp4 nao")

    if args.list:
        for i, p in enumerate(clips, 1):
            print("%d|%s" % (i, p))
        return 0

    if args.path:
        raw = os.environ.get("LIFESTREAM_PICK_INDEX", "").strip()
        try:
            n = int(raw)
        except ValueError:
            n = None
        if n is None or not 1 <= n <= len(clips):
            return _fail("So thu tu khong hop le: '%s' (chon 1-%d)"
                         % (raw, len(clips)))
        print(clips[n - 1])
        return 0

    print(random.choice(clips))
    return 0


if __name__ == "__main__":
    sys.exit(main())
