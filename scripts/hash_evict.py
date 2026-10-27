#!/usr/bin/env python3
"""sha256 of a large file through 256 MiB read-only mmap windows; each window is invalidated (msync MS_SYNC|MS_INVALIDATE, as
reports/find_cached.py does) right after it is hashed, so the page cache this read creates never exceeds one window. Reports
the file's resident pages before and after (mincore).
  python3 -I hash_evict.py <file> <expected sha256>   (exit 0 on match)"""
import ctypes, hashlib, os, sys
libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p; libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]; libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
PAGE = os.sysconf("SC_PAGE_SIZE"); WIN = 256 << 20

def resident(fd, size):
    tot = 0
    for off in range(0, size, 1 << 30):
        n = min(1 << 30, size - off); addr = libc.mmap(None, n, 1, 1, fd, off)
        if addr == ctypes.c_void_p(-1).value: return -1
        try:
            vec = (ctypes.c_ubyte * ((n + PAGE - 1) // PAGE))(); libc.mincore(addr, n, vec); tot += sum(v & 1 for v in vec) * PAGE
        finally: libc.munmap(addr, n)
    return tot

p, want = sys.argv[1], sys.argv[2]
size = os.path.getsize(p); fd = os.open(p, os.O_RDONLY); h = hashlib.sha256()
try:
    before = resident(fd, size)
    for off in range(0, size, WIN):
        n = min(WIN, size - off); addr = libc.mmap(None, n, 1, 1, fd, off)
        if addr == ctypes.c_void_p(-1).value: sys.exit(f"mmap failed at {off}: errno {ctypes.get_errno()}")
        try:
            h.update(ctypes.string_at(addr, n))
            libc.msync(addr, n, 0x10 | 0x2)
        finally: libc.munmap(addr, n)
    after = resident(fd, size)
finally: os.close(fd)
d = h.hexdigest()
print(f"{os.path.basename(p)} {size} sha256 {d} {'OK' if d == want else 'MISMATCH expected ' + want}; resident before {before / 1e9:.1f} GB, after {after / 1e9:.1f} GB")
sys.exit(0 if d == want else 1)
