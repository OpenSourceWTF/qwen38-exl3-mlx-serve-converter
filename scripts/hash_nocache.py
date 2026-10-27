#!/usr/bin/env python3
"""sha256 of a large file with uncached (F_NOCACHE) 64 MB sequential reads, so hashing does not grow the page cache.
  python3 -I hash_nocache.py <file> <expected sha256>   (exit 0 on match)"""
import fcntl, hashlib, os, sys
p, want = sys.argv[1], sys.argv[2]
h = hashlib.sha256(); fd = os.open(p, os.O_RDONLY); fcntl.fcntl(fd, 48, 1)   # F_NOCACHE
try:
    while True:
        b = os.read(fd, 64 << 20)
        if not b: break
        h.update(b)
finally: os.close(fd)
d = h.hexdigest(); print(f"{os.path.basename(p)} {os.path.getsize(p)} sha256 {d} {'OK' if d == want else 'MISMATCH expected ' + want}")
sys.exit(0 if d == want else 1)
