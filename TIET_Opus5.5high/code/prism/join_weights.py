"""Rejoin weight files that are stored in the repo as <file>.partNNN chunks (GitHub's 100 MB per-file limit).

Runs automatically from run.sh before inference; does nothing once the joined files exist.
Each joined file is checked against weights/MD5SUMS.txt.
"""
import glob
import hashlib
import os
import sys

from . import config as C


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    W = C.WEIGHTS_DIR
    sums = {}
    mp = os.path.join(W, "MD5SUMS.txt")
    if os.path.exists(mp):
        for line in open(mp):
            md5, name = line.split(maxsplit=1)
            sums[os.path.normpath(name.strip().lstrip("*"))] = md5
    for first in sorted(glob.glob(os.path.join(W, "**", "*.part000"), recursive=True)):
        dst = first[: -len(".part000")]
        if os.path.exists(dst):
            continue
        parts = sorted(glob.glob(glob.escape(dst) + ".part[0-9][0-9][0-9]"))
        tmp = dst + ".joining"
        with open(tmp, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    while True:
                        b = f.read(1 << 24)
                        if not b:
                            break
                        out.write(b)
        rel = os.path.normpath(os.path.relpath(dst, W))
        if rel in sums and _md5(tmp) != sums[rel]:
            os.remove(tmp)
            sys.exit(f"weights: checksum mismatch after joining {rel}")
        os.replace(tmp, dst)
        print(f"weights: joined {rel} from {len(parts)} parts", file=sys.stderr)


if __name__ == "__main__":
    main()
