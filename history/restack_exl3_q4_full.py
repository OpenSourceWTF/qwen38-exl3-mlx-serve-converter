#!/usr/bin/env python3
"""Build the FULL-EXL3 Qwen3.8-Flash-Next 4.05bpw_h6_ng6 pack for mlx-serve. Derived from restack_exl3_full.py (the 5.05
build, pack v5); the tensor mapping rules are unchanged (see that script): EXL3 trunk modules -> `<host name>.weight` U16
[in/16, out/16, n] / `.scales` (suh F16) / `.biases` (svh F16), mul1 asserted == 0x83DCD12D and dropped; plain f16 -> bf16
(except the n-gram head_bias, kept f16); conv1d [C, 1, K] -> [C, K, 1]; PLE small tensors from the n-gram table header,
renamed (head_bias as `...ple.ple_embedding.head_bias`); `qwen4_norm_convention` marker (delta default); vision from the
hybrid pack; tokenizer/templates linked from the hybrid pack (the same files the 5.05 full pack links).
Changes vs restack_exl3_full.py:
  * --src is turboderp's full branch download (model-0000N-of-0000M.safetensors + index): the routed experts in it are
    skipped here (restack_exl3_q4_experts.py writes experts-*.safetensors into --out first; their headers are indexed);
  * tensors are read one at a time with seek/read (the 5.05 script read whole fetched-trunk shards; 4.05's are 8 GB);
  * output shards are named and grouped like the reference pack (--ref, the 5.05 full pack): each tensor goes to the shard
    that holds the same name in the reference index, so the shard sets line up one to one;
  * the n-gram table is hard-linked from --ngram (byte-identical LFS object; default: the reference pack's);
  * the MTP hyper_connection_mixer: turboderp's 4.05 branch SHIPS it (5.05 did not), so --mtp-mixer turboderp (default)
    takes his f16 tensors (-> bf16, delta norm as stored, dense like the trunk's final mixer) into trunk-exl3-carry;
    --mtp-mixer hybrid restores the 5.05 substitution (the hybrid's 8-bit affine mixer, hc_norm -1);
  * config: from the hybrid config as before, plus expert_quant.k = --expert-k, trunk_quant.source = --source-tag,
    mtp_mixer_from; --check prints the plan + K histogram per module class and writes nothing.

  python3 -I restack_exl3_q4_full.py --src <4.05 dir> --hybrid <5.05 hybrid pack> --ref <5.05 full pack> --out <pack> [--check]
"""
import argparse, ctypes, json, os, re, struct, sys
import numpy as np

MUL1 = 0x83DCD12D
EXP_RE = re.compile(r"\.mlp\.experts\.\d+\.")

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

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

FOLD_SUFFIXES = ("hc_norm.weight", "q_norm.weight", "k_norm.weight", "q_layernorm.weight", "k_layernorm.weight", "ple.norm_key.weight",
                 "ple.norm_query.weight", "ple.norm_conv.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")
def bf16_minus_one(raw):
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    f = u.view(np.float32) - 1.0
    v = f.view(np.uint32); return ((v + 0x7FFF + ((v >> 16) & 1)) >> 16).astype(np.uint16).tobytes()

def f16_to_bf16(raw):
    f = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    u = f.view(np.uint32)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    rounded[np.isnan(f)] = 0x7FC0
    return rounded.tobytes()

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

def write_shard(path, meta, items):
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

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True); ap.add_argument("--hybrid", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--ngram", default="")
    ap.add_argument("--expert-k", type=int, default=4)
    ap.add_argument("--source-tag", default="turboderp/Qwen3.8-Flash-Next-exl3@4.05bpw_h6_ng6")
    ap.add_argument("--mtp-mixer", default="turboderp", choices=["turboderp", "hybrid"])
    ap.add_argument("--norm-convention", default="delta", choices=["delta", "folded"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    src_idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    hy_idx = json.load(open(os.path.join(a.hybrid, "model.safetensors.index.json")))["weight_map"]
    ref_idx = json.load(open(os.path.join(a.ref, "model.safetensors.index.json")))["weight_map"]
    hy_cfg = json.load(open(os.path.join(a.hybrid, "config.json")))
    tr_idx = {n: sh for n, sh in src_idx.items() if not EXP_RE.search(n) and not n.startswith("model.visual.")}
    mixer_pfx = "mtp.hyper_connection_mixer."
    if a.mtp_mixer == "hybrid": tr_idx = {n: sh for n, sh in tr_idx.items() if not n.startswith(mixer_pfx)}
    hdrs = {}
    def meta(n):
        fn = tr_idx[n]
        if fn not in hdrs: hdrs[fn] = read_header(os.path.join(a.src, fn))
        h, base = hdrs[fn]; return fn, h[n], base
    modules = {}
    for n in tr_idx:
        base, _, leaf = n.rpartition("."); modules.setdefault(base, {})[leaf] = n
    exl3_mods = {b: t for b, t in modules.items() if "trellis" in t}
    for b, t in exl3_mods.items():
        if set(t) != {"trellis", "suh", "svh", "mul1"}: sys.exit(f"{b}: EXL3 module with leaves {sorted(t)}")
    plain = [n for n in tr_idx if n.rpartition(".")[0] not in exl3_mods]
    ks = {}
    for b, t in exl3_mods.items():
        _, mt, _ = meta(t["trellis"]); c = re.sub(r"\.\d+(?=\.|$)", ".N", host_name(b))
        ks.setdefault((c, mt["shape"][-1] // 16), 0); ks[(c, mt["shape"][-1] // 16)] += 1
    print(f"EXL3 trunk modules: {len(exl3_mods)}; plain tensors: {len(plain)}; mtp mixer: {a.mtp_mixer}")
    for (c, k), n in sorted(ks.items()): print(f"  K{k} x{n:3d} {c}")
    # the output plan: (out name) -> (dtype, shape, nbytes, producer)
    plan = {}
    def reader(fn, off, nb, conv=None):
        def prod():
            with open(os.path.join(a.src, fn), "rb") as f: f.seek(off); raw = f.read(nb)
            return conv(raw) if conv else raw
        return prod
    for b, t in exl3_mods.items():
        fn, mt, base = meta(t["mul1"])
        with open(os.path.join(a.src, fn), "rb") as f:
            f.seek(base + mt["data_offsets"][0]); v = struct.unpack("<I", f.read(4))[0]
        if mt["dtype"] != "I32" or v != MUL1: sys.exit(f"{b}.mul1 {mt['dtype']} {v:#x}")
        for leaf, new_leaf in (("trellis", "weight"), ("suh", "scales"), ("svh", "biases")):
            fn, mt, base = meta(t[leaf]); nb = mt["data_offsets"][1] - mt["data_offsets"][0]
            dt = "U16" if leaf == "trellis" else mt["dtype"]
            if leaf == "trellis" and mt["dtype"] != "I16": sys.exit(f"{t[leaf]}: {mt['dtype']}")
            if leaf != "trellis" and mt["dtype"] != "F16": sys.exit(f"{t[leaf]}: {mt['dtype']}")
            plan[host_name(b) + "." + new_leaf] = (dt, list(mt["shape"]), nb, reader(fn, base + mt["data_offsets"][0], nb))
    for n in plain:
        fn, mt, base = meta(n); nb = mt["data_offsets"][1] - mt["data_offsets"][0]
        new = ple_rename(host_name(n)); dt = mt["dtype"]; conv = None
        if dt == "F16" and not n.endswith(".head_bias"): conv = f16_to_bf16; dt = "BF16"
        shape = list(mt["shape"])
        if n.endswith("conv1d.weight") and len(shape) == 3 and shape[1] == 1: shape = [shape[0], shape[2], 1]
        plan[new] = (dt, shape, nb, reader(fn, base + mt["data_offsets"][0], nb, conv))
    # carry: the hybrid's MTP mixer (only with --mtp-mixer hybrid) + the PLE small tensors from the n-gram table header
    carry = {}
    if a.mtp_mixer == "hybrid":
        for n in sorted(x for x in hy_idx if x.startswith("language_model.mtp.hyper_connection_mixer.")):
            hh, hb = read_header(os.path.join(a.hybrid, hy_idx[n])); t = hh[n]; nb = t["data_offsets"][1] - t["data_offsets"][0]
            def prod(p=os.path.join(a.hybrid, hy_idx[n]), o=hb + t["data_offsets"][0], nb=nb, fold=(a.norm_convention == "delta" and t["dtype"] == "BF16" and n.endswith(FOLD_SUFFIXES))):
                with open(p, "rb") as f: f.seek(o); raw = f.read(nb)
                return bf16_minus_one(raw) if fold else raw
            carry[n] = (t["dtype"], t["shape"], nb, prod)
    ngram_src = a.ngram or os.path.join(a.ref, "ngram_embedding.safetensors")
    if os.path.getsize(ngram_src) != 39040193720: sys.exit(f"{ngram_src}: size {os.path.getsize(ngram_src)}, expected 39040193720")
    nh, nbase = read_header(ngram_src)
    for n, t in nh.items():
        if n == "__metadata__" or n.endswith(".trellis"): continue
        nb = t["data_offsets"][1] - t["data_offsets"][0]
        def prod(o=nbase + t["data_offsets"][0], nb=nb):
            with open(ngram_src, "rb") as f: f.seek(o); return f.read(nb)
        carry[ple_rename(host_name(n))] = (t["dtype"], t["shape"], nb, prod)
    # group by the reference pack's shard names; the carry shard takes the carry set + turboderp's MTP mixer
    shards = {}
    for n in plan:
        sh = "trunk-exl3-carry.safetensors" if n.startswith("language_model.mtp.hyper_connection_mixer.") else ref_idx.get(n)
        if sh is None or not sh.startswith("trunk-exl3-"): sys.exit(f"{n}: no trunk-exl3 shard in the reference index ({sh})")
        shards.setdefault(sh, {})[n] = plan[n]
    for n in carry:
        if n in plan: sys.exit(f"{n}: in both plan and carry")
        shards.setdefault("trunk-exl3-carry.safetensors", {})[n] = carry[n]
    print("output trunk shards:", {s: len(v) for s, v in sorted(shards.items())})
    exp_files = sorted(f for f in set(ref_idx.values()) if f.startswith("experts-"))
    vision = sorted(f for f in set(hy_idx.values()) if "vision" in f)
    print(f"expert shards expected in --out: {len(exp_files)}; vision shards linked from the hybrid: {vision}; ngram: {ngram_src}")
    if a.check:
        missing = [f for f in exp_files if not os.path.exists(os.path.join(a.out, f))]
        print("expert shards missing in --out:", missing if missing else "none"); print("CHECK_DONE"); return
    missing = [f for f in exp_files if not os.path.exists(os.path.join(a.out, f))]
    if missing: sys.exit(f"run restack_exl3_q4_experts.py first; missing {missing}")
    os.makedirs(a.out, exist_ok=True)
    new_map = {}
    def link(src, dst_name):
        dst = os.path.join(a.out, dst_name)
        if os.path.exists(dst): os.remove(dst)
        os.link(src, dst)
    for f in exp_files:
        h, _ = read_header(os.path.join(a.out, f))
        for n in h:
            if n != "__metadata__": new_map[n] = f
    for f in vision:
        link(os.path.join(a.hybrid, f), f)
        for n, sh in hy_idx.items():
            if sh == f: new_map[n] = f
    link(ngram_src, "ngram_embedding.safetensors")
    for sh in sorted(shards):
        items = [(n,) + shards[sh][n] for n in sorted(shards[sh])]
        if sh == "trunk-exl3-carry.safetensors":
            m = {"format": "pt", "note": ("MTP hyper_connection_mixer: turboderp's own (4.05 ships it; f16 -> bf16)" if a.mtp_mixer == "turboderp"
                 else "MTP hyper_connection_mixer from the hybrid pack (8-bit affine)") + "; PLE small tensors from ngram_embedding.safetensors",
                 "source": a.source_tag}
        else:
            m = {"format": "pt", "source": a.source_tag, "trunk_quant": "exl3"}
        nbytes = write_shard(os.path.join(a.out, sh), m, items)
        for n in shards[sh]: new_map[n] = sh
        print(f"wrote {sh}: {len(items)} tensors, {nbytes / 1e9:.3f} GB", flush=True)
    for fn in sorted(set(tr_idx.values())): evict(os.path.join(a.src, fn))
    cfg = dict(hy_cfg)
    cfg["expert_quant"] = {"format": "exl3", "k": a.expert_k, "codebook": "mul1"}
    cfg["trunk_quant"] = {"format": "exl3", "codebook": "mul1", "window": 16, "source": a.source_tag}
    cfg["ngram_table"] = {"file": "ngram_embedding.safetensors", "format": "exl3_ngram_trellis", "k": 6, "row_dim": 160, "codebook": "mul1"}
    cfg["vision_from"] = "hybrid"
    cfg["mtp_mixer_from"] = a.mtp_mixer
    cfg["quantization_note"] = "quantization block kept nominal for the loader; no trunk tensor is affine (see trunk_quant)"
    cfg["qwen4_norm_convention"] = a.norm_convention
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for fn in os.listdir(a.hybrid):
        if fn.endswith((".json", ".jinja", ".txt", "LICENSE", "README.md")) and fn not in ("config.json", "model.safetensors.index.json") and not fn.startswith(("experts-", "trunk-")):
            src = os.path.join(a.hybrid, fn)
            if os.path.isfile(src): link(src, fn)
    json.dump({"metadata": {"total_size": 0}, "weight_map": new_map}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    print(f"DONE: {len(new_map)} tensors indexed in {a.out}")

if __name__ == "__main__": main()
