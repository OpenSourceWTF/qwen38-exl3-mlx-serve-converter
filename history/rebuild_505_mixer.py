#!/usr/bin/env python3
"""Rebuild the 5.05 full pack's MTP hyper-connection mixer with turboderp's own, in a sibling pack folder.

The 5.05 full pack (built 2026-10-22) carries ddalcu's 8-bit affine MTP mixer from the
hybrid pack: its download fetched only the files turboderp's shard index names, and his 5.05bpw_h6_ng6 branch ships his own
mixer as the unindexed mtp_hyper_connection_mixer_patch.safetensors (commit 7cef615f, 2026-08-31: three F16 tensors, hc_norm
[10240], input_mix_weight_down [320, 10240], input_mix_weight_up [10240, 320]). The 2.05 / 3.05 / 4.05 packs carry his mixer
from his indexed shards, converted by restack_exl3_q4_full.py --mtp-mixer turboderp: f16 -> bf16 (round to nearest even), the
delta hc_norm as stored, dense. This does the same for 5.05:
  - OUT/trunk-exl3-carry.safetensors = SRC's carry with the mixer replaced: SRC's PLE small tensors byte for byte + his three
    mixer tensors (f16 -> bf16), names sorted, the converter's header layout and carry metadata;
  - OUT/model.safetensors.index.json = SRC's without the four .scales / .biases entries of the 8-bit mixer;
  - OUT/config.json = SRC's with "mtp_mixer_from": "turboderp" after "vision_from" (as in the 4.05 pack);
  - OUT/CONVERSION.md: this record (internal, like the other packs' CONVERSION.md; not uploaded);
  - every other file of SRC hard-linked into OUT (the same inodes: no copy, one page cache);
  - checks: the patch's sha256, names, dtypes and shapes; the written carry read back; SRC's PLE tensors unchanged; and, report
    only, whether the converted mixer equals REF's (the 4.05 pack's, his mixer from his 4.05 branch).
Usage: python -I rebuild_505_mixer.py --src SRC --patch PATCH --ref REF --out OUT   (OUT must not exist)
"""
import argparse
import ctypes
import hashlib
import json
import os
import struct
import sys

import numpy as np

PATCH_SHA = "9b29ae89413854802076ad392809abe4d91a1e61a6bc08193dad8271a7510953"
PATCH_COMMIT = "7cef615f7fd3681295b68848018876dfabc336c7"
MIX = "language_model.mtp.hyper_connection_mixer."
SHAPES = {"hc_norm.weight": [10240], "input_mix_weight_down.weight": [320, 10240], "input_mix_weight_up.weight": [10240, 320]}
CARRY, INDEX, CONFIG = "trunk-exl3-carry.safetensors", "model.safetensors.index.json", "config.json"


def f16_to_bf16(raw):  # restack_exl3_q4_full.py's, verbatim
    f = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    u = f.view(np.uint32)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    rounded[np.isnan(f)] = 0x7FC0
    return rounded.tobytes()


libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p; libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]; libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
def evict(path):  # restack_exl3_q4_full.py's, verbatim
    st = os.stat(path); fd = os.open(path, os.O_RDONLY); CH = 1 << 30
    try:
        for off in range(0, st.st_size, CH):
            n = min(CH, st.st_size - off); addr = libc.mmap(None, n, 1, 1, fd, off)
            if addr == ctypes.c_void_p(-1).value: return
            try: libc.msync(addr, n, 0x10 | 0x2)
            finally: libc.munmap(addr, n)
    finally: os.close(fd)


def write_shard(path, meta, items):  # restack_exl3_q4_full.py's, verbatim
    """items: [(name, dtype, shape, nbytes, producer)] where producer() returns the bytes; streamed, one tensor in RAM."""
    h = {"__metadata__": meta}; off = 0
    for n, dt, shp, nb, _ in items: h[n] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + nb]}; off += nb
    hb = json.dumps(h, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for n, _, _, nb, prod in items:
            raw = prod()
            if len(raw) != nb: sys.exit(f"{n}: produced {len(raw)} bytes, header says {nb}")
            f.write(raw)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path); evict(path)
    return off


def read_st(path):
    """(metadata, {name: (dtype, shape, bytes)}) of a small safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        base, out = 8 + n, {}
        for k, v in h.items():
            if k == "__metadata__": continue
            a, b = v["data_offsets"]; f.seek(base + a); out[k] = (v["dtype"], v["shape"], f.read(b - a))
    return h.get("__metadata__"), out


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--src", "--patch", "--ref", "--out"): ap.add_argument(k, required=True)
    a = ap.parse_args()
    s = sha(a.patch)
    if s != PATCH_SHA: sys.exit(f"patch sha256 {s} != {PATCH_SHA}")
    _, pt = read_st(a.patch)
    want = {"mtp.hyper_connection_mixer." + k: k for k in SHAPES}
    if set(pt) != set(want): sys.exit(f"patch tensors {sorted(pt)} != {sorted(want)}")
    for n, k in want.items():
        if pt[n][0] != "F16" or pt[n][1] != SHAPES[k]: sys.exit(f"{n}: {pt[n][0]} {pt[n][1]}, expected F16 {SHAPES[k]}")
    print(f"patch ok: sha256 {s[:16]}, {len(pt)} tensors F16, commit {PATCH_COMMIT[:12]}")

    smeta, src = read_st(os.path.join(a.src, CARRY))
    keep = {n: v for n, v in src.items() if not n.startswith(MIX)}
    print(f"SRC carry: metadata {smeta}; replaced: {sorted(n for n in src if n.startswith(MIX))}; kept: {sorted(keep)}")
    new = dict(keep)
    for n, k in want.items():
        new[MIX + k] = ("BF16", SHAPES[k], f16_to_bf16(pt[n][2]))
    _, ref = read_st(os.path.join(a.ref, CARRY))
    for n in sorted(new):
        r = ref.get(n)
        print(f"vs REF (4.05 carry) {n}: {'identical' if r and r[0] == new[n][0] and r[1] == new[n][1] and r[2] == new[n][2] else 'DIFFERENT' if r else 'absent'}")

    if os.path.exists(a.out): sys.exit(f"{a.out} exists")
    os.makedirs(a.out)
    links = 0
    for fn in sorted(os.listdir(a.src)):   # every file written below is left out: a write through a hard link would change SRC
        p = os.path.join(a.src, fn)
        if fn in (CARRY, INDEX, CONFIG, "CONVERSION.md") or not os.path.isfile(p): continue
        os.link(p, os.path.join(a.out, fn)); links += 1

    meta = {"format": "pt", "note": "MTP hyper_connection_mixer: turboderp's own (5.05bpw_h6_ng6 ships it as the unindexed "
            "mtp_hyper_connection_mixer_patch.safetensors, commit 7cef615f; f16 -> bf16); PLE small tensors from ngram_embedding.safetensors",
            "source": "turboderp/Qwen3.8-Flash-Next-exl3@5.05bpw_h6_ng6"}
    items = [(n, new[n][0], new[n][1], len(new[n][2]), (lambda r=new[n][2]: r)) for n in sorted(new)]
    nbytes = write_shard(os.path.join(a.out, CARRY), meta, items)
    _, chk = read_st(os.path.join(a.out, CARRY))
    if sorted(chk) != sorted(new) or any(chk[n] != (new[n][0], new[n][1], new[n][2]) for n in new): sys.exit("carry read-back mismatch")
    if any(chk[n] != keep[n] for n in keep): sys.exit("PLE tensors changed")

    idx = json.load(open(os.path.join(a.src, INDEX)))
    wm = idx["weight_map"]
    gone = sorted(n for n in wm if n.startswith(MIX) and n.endswith((".scales", ".biases")))
    for n in gone: del wm[n]
    for n in new:
        if wm.get(n) != CARRY: sys.exit(f"index: {n} -> {wm.get(n)}, expected {CARRY}")
    with open(os.path.join(a.out, INDEX), "w") as f: json.dump(idx, f, indent=1)

    cfg = json.load(open(os.path.join(a.src, CONFIG)))
    if "mtp_mixer_from" in cfg: sys.exit(f"SRC config already has mtp_mixer_from={cfg['mtp_mixer_from']}")
    out = {}
    for k, v in cfg.items():
        out[k] = v
        if k == "vision_from": out["mtp_mixer_from"] = "turboderp"
    out.setdefault("mtp_mixer_from", "turboderp")
    with open(os.path.join(a.out, CONFIG), "w") as f: json.dump(out, f, indent=2)

    cs = sha(os.path.join(a.out, CARRY))
    with open(os.path.join(a.out, "CONVERSION.md"), "w") as f:
        f.write("# 5.05 full pack, MTP mixer rebuilt with turboderp's own (internal record, not uploaded)\n\n"
                f"Made by rebuild_505_mixer.py from `{a.src}`, 2026-10-27.\n\n"
                f"- `{CARRY}`: turboderp's `mtp_hyper_connection_mixer_patch.safetensors` (branch 5.05bpw_h6_ng6, commit "
                f"{PATCH_COMMIT}, sha256 {PATCH_SHA}), f16 -> bf16 as restack_exl3_q4_full.py converts it, the delta hc_norm as stored; "
                f"the PLE small tensors byte for byte from the source pack's carry. {nbytes} data bytes, sha256 {cs}.\n"
                f"- `{INDEX}`: the source's without {', '.join(gone)}.\n"
                f"- `{CONFIG}`: the source's plus `\"mtp_mixer_from\": \"turboderp\"`.\n"
                f"- every other file: a hard link to the source pack's ({links} files).\n"
                "- before: ddalcu's 8-bit affine MTP mixer from the hybrid pack (the download fetched only indexed files).\n")
    for fn in (CARRY, INDEX, CONFIG): evict(os.path.join(a.out, fn))
    print(f"wrote {a.out}: {CARRY} {nbytes} data bytes sha256 {cs[:16]}; index -{len(gone)} entries; config +mtp_mixer_from; {links} hard links")
    print("MIXER_REBUILD_OK")


if __name__ == "__main__":
    main()
