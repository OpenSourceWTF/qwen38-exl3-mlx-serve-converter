#!/usr/bin/env python3
"""Verify one raw turboderp shard against the converted pack before the raw shard is deleted (streaming conversion).

One sequential pass over the raw file (F_NOCACHE reads: no page-cache growth) feeds sha256 (compared with the HF LFS oid)
and, tensor by tensor in file order, compares the bytes with where the conversion put them:
  routed experts  <pfx>.layers.L.mlp.experts.E.P.{trellis,suh,svh} -> <host pfx>.layers.L.mlp.switch_mlp.P.<leaf>[E] (bytes)
  EXL3 trunk      <mod>.{trellis,suh,svh} -> host(<mod>).{weight,scales,biases} (bytes)
  *.mul1          value == 0x83DCD12D (not carried)
  plain           host/ple name; F16 -> expected bf16 (round-to-nearest-even, as the converter), else bytes; conv1d bytes
  model.visual.*  not carried (vision comes from the hybrid pack): hashed, counted, skipped
Exit 0 only if the hash matches and every carried tensor compares equal. Read-only on both sides; evicts what it read.

  python3 -I verify_raw_shard.py --src <dir> --out <pack> --shard <file> --sha256 <oid>
"""
import argparse, ctypes, fcntl, hashlib, json, os, re, struct, sys
import numpy as np

MUL1 = 0x83DCD12D
EXP = re.compile(r"^(model\.language_model|mtp)\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(trellis|suh|svh|mul1)$")
F_NOCACHE = 48

def host_name(name):
    if name.startswith("model.language_model."): return "language_model.model." + name[len("model.language_model."):]
    if name.startswith("mtp."): return "language_model.mtp." + name[len("mtp."):]
    if name.startswith("lm_head."): return "language_model.lm_head." + name[len("lm_head."):]
    if name == "lm_head": return "language_model.lm_head"
    return name

def ple_rename(name):
    m = re.match(r"^(.*\.ple\.ple_embedding)\.ngram_embedding\.(head_offsets|head_vocab_sizes|layer_multipliers|head_bias)$", name)
    if not m: return name
    base, leaf = m.groups()
    return f"{base}." + {"head_offsets": "ngram_heads_offsets", "head_vocab_sizes": "ngram_heads_vocab_sizes",
                          "layer_multipliers": "layer_multipliers", "head_bias": "head_bias"}[leaf]

def f16_to_bf16(raw):
    f = np.frombuffer(raw, dtype=np.float16).astype(np.float32); u = f.view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16); r[np.isnan(f)] = 0x7FC0
    return r.tobytes()

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hb = f.read(n)
        return json.loads(hb), 8 + n, struct.pack("<Q", n) + hb

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p; libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]; libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
def evict(path):
    st = os.stat(path); fd = os.open(path, os.O_RDONLY); CH = 1 << 30
    try:
        for off in range(0, st.st_size, CH):
            n = min(CH, st.st_size - off); addr = libc.mmap(None, n, 1, 1, fd, off)
            if addr == ctypes.c_void_p(-1).value: return
            try: libc.msync(addr, n, 0x10 | 0x2)
            finally: libc.munmap(addr, n)
    finally: os.close(fd)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", required=True); ap.add_argument("--sha256", required=True)
    a = ap.parse_args()
    outmap = {}
    for fn in os.listdir(a.out):
        if not fn.endswith(".safetensors") or not (fn.startswith("experts-") or fn.startswith("trunk-exl3-")): continue
        h, base, _ = read_header(os.path.join(a.out, fn))
        for n, t in h.items():
            if n != "__metadata__": outmap[n] = (fn, base + t["data_offsets"][0], t["data_offsets"][1] - t["data_offsets"][0], t["shape"], t["dtype"])
    path = os.path.join(a.src, a.shard)
    hdr, base, hbytes = read_header(path)
    size = os.path.getsize(path)
    sha = hashlib.sha256(); sha.update(hbytes)
    tens = sorted(((t["data_offsets"][0], t["data_offsets"][1], n, t) for n, t in hdr.items() if n != "__metadata__"))
    ofh = {}
    def out_bytes(fn, off, nb):
        if fn not in ofh: ofh[fn] = open(os.path.join(a.out, fn), "rb")
        f = ofh[fn]; f.seek(off); return f.read(nb)
    stats = {"expert": 0, "trunk_exl3": 0, "plain": 0, "mul1": 0, "visual": 0, "gap_bytes": 0}
    bad = []
    pos = 0
    fd = os.open(path, os.O_RDONLY); fcntl.fcntl(fd, F_NOCACHE, 1)
    try:
        os.lseek(fd, base, 0)
        buf = [b"", 0]   # sequential buffered reader: 64 MB uncached reads, tensors sliced from the buffer
        def rd(nb):
            while len(buf[0]) - buf[1] < nb:
                more = os.read(fd, max(64 << 20, nb))
                if not more: raise IOError("short read")
                buf[0] = buf[0][buf[1]:] + more; buf[1] = 0
            r = buf[0][buf[1]:buf[1] + nb]; buf[1] += nb
            return r
        CH = 64 << 20
        for a0, a1, n, t in tens:
            if a0 < pos: bad.append(f"{n}: overlapping offsets"); continue
            if a0 > pos: g = rd(a0 - pos); sha.update(g); stats["gap_bytes"] += len(g)
            nbt = a1 - a0
            if nbt > CH:   # big tensors (embed_tokens, lm_head, visual): identity mappings only; hash + compare in 64 MB pieces
                mod, _, leaf = n.rpartition(".")
                if n.startswith("model.visual."): dst = None
                elif leaf in ("trellis", "suh", "svh") and not EXP.match(n): dst = host_name(mod) + "." + {"trellis": "weight", "suh": "scales", "svh": "biases"}[leaf]
                elif t["dtype"] != "F16" and not EXP.match(n): dst = ple_rename(host_name(n))
                else: bad.append(f"{n}: unexpected big tensor kind ({t['dtype']})"); dst = None
                ok = dst is None or (dst in outmap and outmap[dst][2] == nbt)
                if dst is not None and not ok: bad.append(f"{n}: {dst} missing or size differs")
                done = 0
                while done < nbt:
                    piece = rd(min(CH, nbt - done)); sha.update(piece)
                    if dst is not None and ok and out_bytes(outmap[dst][0], outmap[dst][1] + done, len(piece)) != piece:
                        bad.append(f"{n}: bytes differ in {dst} at +{done}"); ok = False
                    done += len(piece)
                pos = a1
                stats["visual" if n.startswith("model.visual.") else ("trunk_exl3" if leaf in ("trellis", "suh", "svh") else "plain")] += 1
                continue
            raw = rd(nbt); sha.update(raw); pos = a1
            if n.startswith("model.visual."): stats["visual"] += 1; continue
            if n.endswith(".mul1"):
                stats["mul1"] += 1
                if t["dtype"] != "I32" or struct.unpack("<I", raw)[0] != MUL1: bad.append(f"{n}: mul1 {raw.hex()}")
                continue
            m = EXP.match(n)
            if m:
                pfx, L, e, P, leaf = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
                dst = ("language_model.model" if pfx == "model.language_model" else "language_model.mtp") + f".layers.{L}.mlp.switch_mlp.{P}.{leaf}"
                if dst not in outmap: bad.append(f"{n}: {dst} not in the pack"); continue
                fn, off, nb, shape, _ = outmap[dst]; per = nb // shape[0]
                if per != len(raw): bad.append(f"{n}: slice {per} bytes vs {len(raw)}"); continue
                if out_bytes(fn, off + e * per, per) != raw: bad.append(f"{n}: bytes differ in {dst}[{e}]")
                stats["expert"] += 1; continue
            mod, _, leaf = n.rpartition(".")
            if leaf in ("trellis", "suh", "svh"):
                dst = host_name(mod) + "." + {"trellis": "weight", "suh": "scales", "svh": "biases"}[leaf]; want = raw; stats["trunk_exl3"] += 1
            else:
                dst = ple_rename(host_name(n)); stats["plain"] += 1
                want = f16_to_bf16(raw) if t["dtype"] == "F16" and not n.endswith(".head_bias") else raw
            if dst not in outmap: bad.append(f"{n}: {dst} not in the pack"); continue
            fn, off, nb, shape, dt = outmap[dst]
            if nb != len(want) or out_bytes(fn, off, nb) != want: bad.append(f"{n}: bytes differ in {dst} ({dt} {shape})")
        tail = size - base - pos
        while tail > 0: g = rd(min(tail, CH)); sha.update(g); stats["gap_bytes"] += len(g); tail -= len(g)
    finally:
        os.close(fd)
        for f in ofh.values(): f.close()
    digest = sha.hexdigest()
    for fn in ofh: evict(os.path.join(a.out, fn))
    ok_sha = digest == a.sha256
    print(f"{a.shard}: sha256 {digest} {'OK' if ok_sha else 'MISMATCH (expected ' + a.sha256 + ')'}; {len(tens)} tensors {stats}; mismatches {len(bad)}")
    for b in bad[:20]: print("  BAD", b)
    if ok_sha and not bad: print("RAW_SHARD_VERIFIED", a.shard); sys.exit(0)
    sys.exit(1)

if __name__ == "__main__": main()
