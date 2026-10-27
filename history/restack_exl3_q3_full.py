#!/usr/bin/env python3
"""Build the FULL-EXL3 Qwen3.8-Flash-Next 3.05bpw_h5_ng5 pack for mlx-serve. Copy of restack_exl3_q4_full.py (the 4.05 build)
with the same tensor mapping rules (from restack_exl3_full.py, 5.05 pack v5): EXL3 trunk modules -> `<host name>.weight` U16
[in/16, out/16, n] / `.scales` (suh F16) / `.biases` (svh F16), mul1 asserted == 0x83DCD12D and dropped; plain f16 -> bf16
(except the n-gram head_bias, kept f16); conv1d [C, 1, K] -> [C, K, 1]; PLE small tensors from the n-gram table header,
renamed (head_bias as `...ple.ple_embedding.head_bias`); `qwen4_norm_convention` marker (delta default); vision from the
hybrid pack; tokenizer/templates linked from the hybrid pack; output shards grouped by the reference (5.05) pack's names.
Changes vs restack_exl3_q4_full.py:
  * --extra-src: safetensors files in --src that the index does not list are read too. 3.05 ships its MTP
    hyper_connection_mixer as `mtp_hyper_connection_mixer_patch.safetensors` (not indexed); its tensors go to the carry shard;
  * --phase trunk | finalize | all. The streaming conversion runs `trunk` first (all trunk-exl3-* shards; the carry shard with
    the MTP mixer only; needs the raw shards, not the n-gram table and not the experts), then the experts per raw shard (the
    raw shards are deleted after verification), then `finalize` (needs the experts in --out and the table at --ngram): the
    carry shard is rewritten with the PLE small tensors from the table header added, the table is hard-linked, vision and
    small files linked, config.json and the index written (from every output shard header);
  * the table is 3.05's own (NG5): config `ngram_table.k` and `row_dim` are read from the table's `__metadata__` (K, row_dim)
    instead of being fixed at 6 / 160;
  * defaults: --expert-k 3, --source-tag ...@3.05bpw_h5_ng5, --ngram <src>/ngram_embedding.safetensors.

  python3 -I restack_exl3_q3_full.py --src <3.05 dir> --hybrid <5.05 hybrid pack> --ref <5.05 full pack> --out <pack>
                                     --phase trunk|finalize|all [--check]
"""
import argparse, ctypes, json, os, re, struct, sys
import numpy as np

MUL1 = 0x83DCD12D
EXP_RE = re.compile(r"\.mlp\.experts\.\d+\.")
CARRY = "trunk-exl3-carry.safetensors"
MIXER = "language_model.mtp.hyper_connection_mixer."

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

libc.clonefile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]; libc.clonefile.restype = ctypes.c_int

def table_layout(path):
    """turboderp's exl3_ngram_trellis table holds either one `...ngram_embedding.trellis` I16 [rows, cols] (5.05, 4.05) or N
    row shards `...ngram_embedding.shard_i.trellis` (3.05: 128 x [2500012, 51]). The host (qwen4_exp.zig parseTrellis) reads
    ONE `.trellis` tensor. Returns (K, row_dim, new_header, description): new_header None = usable as is (hard link); else the
    replacement header bytes, the ORIGINAL length (space-padded, as safetensors allows), naming the contiguous shards as one
    tensor (every data offset unchanged)."""
    nh, nbase = read_header(path); nm = nh.get("__metadata__", {})
    if nm.get("format") != "exl3_ngram_trellis" or nm.get("codebook") != "mul1": sys.exit(f"table metadata {nm}")
    K, rdim, rows = int(nm["K"]), int(nm["row_dim"]), int(nm["rows"]); cols = 1 + rdim * K // 16
    if os.path.getsize(path) != nbase + max(v["data_offsets"][1] for k, v in nh.items() if k != "__metadata__"): sys.exit("table: file size != header extent")
    trk = [k for k in nh if k.endswith(".trellis")]
    if len(trk) == 1 and trk[0].endswith(".ngram_embedding.trellis"):
        t = nh[trk[0]]
        if t["dtype"] != "I16" or t["shape"] != [rows, cols]: sys.exit(f"table {trk[0]}: {t}")
        return K, rdim, None, f"single tensor [{rows}, {cols}]"
    sh = []
    for k in trk:
        m = re.search(r"^(.*\.ngram_embedding)\.shard_(\d+)\.trellis$", k)
        if not m: sys.exit(f"table: unexpected trellis tensor {k}")
        sh.append((int(m.group(2)), m.group(1), k))
    sh.sort()
    if [i for i, _, _ in sh] != list(range(len(sh))) or len({p for _, p, _ in sh}) != 1: sys.exit("table: shard names are not 0..N-1 under one prefix")
    start = pos = nh[sh[0][2]]["data_offsets"][0]; total = 0
    for i, _, k in sh:
        t = nh[k]
        if t["dtype"] != "I16" or len(t["shape"]) != 2 or t["shape"][1] != cols: sys.exit(f"table {k}: {t['dtype']} {t['shape']}")
        if t["data_offsets"][0] != pos or t["data_offsets"][1] - t["data_offsets"][0] != t["shape"][0] * cols * 2: sys.exit(f"table {k}: not contiguous in shard order")
        pos = t["data_offsets"][1]; total += t["shape"][0]
    if total != rows: sys.exit(f"table: shard rows sum {total} != rows {rows}")
    meta = dict(nm); meta["layout_note"] = (f"header rewritten for mlx-serve: turboderp's {len(sh)} contiguous row shards (shard_rows "
                                            f"{nm.get('shard_rows')}) named as one tensor; every data byte and offset unchanged")
    new = {"__metadata__": meta}
    for k, v in nh.items():
        if k != "__metadata__" and not k.endswith(".trellis"): new[k] = v
    new[sh[0][1] + ".trellis"] = {"dtype": "I16", "shape": [rows, cols], "data_offsets": [start, pos]}
    hb = json.dumps(new, separators=(",", ":")).encode()
    if len(hb) > nbase - 8: sys.exit("table: rewritten header longer than the original")
    return K, rdim, hb + b" " * (nbase - 8 - len(hb)), f"{len(sh)} row shards [{nh[sh[0][2]]['shape'][0]}, {cols}] -> one tensor [{rows}, {cols}] at data offsets [{start}, {pos}]"

def install_table(src, dst, new_hb):
    """Hard link (new_hb None), or an APFS clone of src (clonefile: shares every data block, no copy) whose header bytes alone
    are rewritten in place (same length); then the clone's header and sampled data ranges are checked against src."""
    if os.path.exists(dst): os.remove(dst)
    if new_hb is None: os.link(src, dst); return "hard link"
    if libc.clonefile(src.encode(), dst.encode(), 0) != 0: sys.exit(f"clonefile {src} -> {dst} failed: errno {ctypes.get_errno()}")
    with open(dst, "r+b") as f:
        if struct.unpack("<Q", f.read(8))[0] != len(new_hb): sys.exit("table: header length mismatch")
        f.seek(8); f.write(new_hb); f.flush(); os.fsync(f.fileno())
    h, base = read_header(dst); hs, bs = read_header(src)
    trk = [k for k in h if k.endswith(".trellis")]
    if base != bs or os.path.getsize(dst) != os.path.getsize(src) or len(trk) != 1: sys.exit("table clone: header/size check failed")
    size = os.path.getsize(src); MB = 1 << 20; offs = [base, size - MB] + [base + (size - base - MB) * i // 13 for i in range(1, 13)]
    with open(src, "rb") as a, open(dst, "rb") as b:
        for o in offs:
            a.seek(o); b.seek(o)
            if a.read(MB) != b.read(MB): sys.exit(f"table clone: data differs at {o}")
    evict(src); evict(dst)
    return f"APFS clone + rewritten header ({trk[0]} {h[trk[0]]['shape']}); 14 x 1 MiB data samples equal to the download"

def write_shard(path, meta, items):
    """items: [(name, dtype, shape, nbytes, producer)]; streamed, one tensor in RAM; .partial + rename; page cache evicted."""
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

def file_reader(path, off, nb, conv=None):
    def prod():
        with open(path, "rb") as f: f.seek(off); raw = f.read(nb)
        return conv(raw) if conv else raw
    return prod

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True); ap.add_argument("--hybrid", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--ngram", default="")
    ap.add_argument("--extra-src", default="mtp_hyper_connection_mixer_patch.safetensors", help="comma list of unindexed safetensors in --src")
    ap.add_argument("--phase", default="all", choices=["trunk", "finalize", "all"])
    ap.add_argument("--expert-k", type=int, default=3)
    ap.add_argument("--source-tag", default="turboderp/Qwen3.8-Flash-Next-exl3@3.05bpw_h5_ng5")
    ap.add_argument("--norm-convention", default="delta", choices=["delta", "folded"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ngram_src = a.ngram or os.path.join(a.src, "ngram_embedding.safetensors")
    hy_idx = json.load(open(os.path.join(a.hybrid, "model.safetensors.index.json")))["weight_map"]
    ref_idx = json.load(open(os.path.join(a.ref, "model.safetensors.index.json")))["weight_map"]
    hy_cfg = json.load(open(os.path.join(a.hybrid, "config.json")))
    do_trunk = a.phase in ("trunk", "all"); do_final = a.phase in ("finalize", "all")
    if do_trunk:
        src_idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
        for x in filter(None, a.extra_src.split(",")):
            p = os.path.join(a.src, x)
            if not os.path.exists(p): sys.exit(f"--extra-src {x}: missing")
            for n in read_header(p)[0]:
                if n == "__metadata__": continue
                if n in src_idx: sys.exit(f"{n}: in the index and in {x}")
                src_idx[n] = x
        tr_idx = {n: sh for n, sh in src_idx.items() if not EXP_RE.search(n) and not n.startswith("model.visual.")}
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
        print(f"EXL3 trunk modules: {len(exl3_mods)}; plain tensors: {len(plain)}; extra sources: {a.extra_src}")
        for (c, k), n in sorted(ks.items()): print(f"  K{k} x{n:3d} {c}")
        plan = {}
        for b, t in exl3_mods.items():
            fn, mt, base = meta(t["mul1"])
            with open(os.path.join(a.src, fn), "rb") as f:
                f.seek(base + mt["data_offsets"][0]); v = struct.unpack("<I", f.read(4))[0]
            if mt["dtype"] != "I32" or v != MUL1: sys.exit(f"{b}.mul1 {mt['dtype']} {v:#x}")
            for leaf, new_leaf in (("trellis", "weight"), ("suh", "scales"), ("svh", "biases")):
                fn, mt, base = meta(t[leaf]); nb = mt["data_offsets"][1] - mt["data_offsets"][0]
                if leaf == "trellis" and mt["dtype"] != "I16": sys.exit(f"{t[leaf]}: {mt['dtype']}")
                if leaf != "trellis" and mt["dtype"] != "F16": sys.exit(f"{t[leaf]}: {mt['dtype']}")
                plan[host_name(b) + "." + new_leaf] = ("U16" if leaf == "trellis" else "F16", list(mt["shape"]), nb,
                                                       file_reader(os.path.join(a.src, fn), base + mt["data_offsets"][0], nb))
        for n in plain:
            fn, mt, base = meta(n); nb = mt["data_offsets"][1] - mt["data_offsets"][0]
            new = ple_rename(host_name(n)); dt = mt["dtype"]; conv = None
            if dt == "F16" and not n.endswith(".head_bias"): conv = f16_to_bf16; dt = "BF16"
            shape = list(mt["shape"])
            if n.endswith("conv1d.weight") and len(shape) == 3 and shape[1] == 1: shape = [shape[0], shape[2], 1]
            plan[new] = (dt, shape, nb, file_reader(os.path.join(a.src, fn), base + mt["data_offsets"][0], nb, conv))
        shards = {}
        for n in plan:
            sh = CARRY if n.startswith(MIXER) else ref_idx.get(n)
            if sh is None or not sh.startswith("trunk-exl3-"): sys.exit(f"{n}: no trunk-exl3 shard in the reference index ({sh})")
            shards.setdefault(sh, {})[n] = plan[n]
        missing_mixer = [x for x in ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight") if MIXER + x not in plan]
        if missing_mixer: sys.exit(f"turboderp's MTP mixer is incomplete in the sources: missing {missing_mixer}")
        print("trunk phase output shards:", {s: len(v) for s, v in sorted(shards.items())})
        if not a.check:
            os.makedirs(a.out, exist_ok=True)
            for sh in sorted(shards):
                items = [(n,) + shards[sh][n] for n in sorted(shards[sh])]
                m = ({"format": "pt", "note": "MTP hyper_connection_mixer: turboderp's own (3.05 ships it as mtp_hyper_connection_mixer_patch.safetensors; f16 -> bf16)",
                      "source": a.source_tag} if sh == CARRY else {"format": "pt", "source": a.source_tag, "trunk_quant": "exl3"})
                nbytes = write_shard(os.path.join(a.out, sh), m, items)
                print(f"wrote {sh}: {len(items)} tensors, {nbytes / 1e9:.3f} GB", flush=True)
            for fn in sorted(set(tr_idx.values())): evict(os.path.join(a.src, fn))
            print("TRUNK_PHASE_DONE", flush=True)
    if do_final:
        exp_files = sorted(f for f in set(ref_idx.values()) if f.startswith("experts-"))
        trunk_files = sorted(f for f in set(ref_idx.values()) if f.startswith("trunk-exl3-"))
        vision = sorted(f for f in set(hy_idx.values()) if "vision" in f)
        missing = [f for f in exp_files + trunk_files if not os.path.exists(os.path.join(a.out, f))]
        if not os.path.exists(ngram_src): missing.append(ngram_src)
        nh, nbase = read_header(ngram_src) if os.path.exists(ngram_src) else ({}, 0)
        nm = nh.get("__metadata__", {})
        print(f"finalize: table {ngram_src} meta {nm}; missing inputs: {missing if missing else 'none'}")
        if a.check: print("CHECK_DONE"); return
        if missing: sys.exit(f"finalize: missing {missing}")
        tk, rdim, new_tbl_hdr, tbl_desc = table_layout(ngram_src)
        print(f"table layout: {tbl_desc}; K {tk}, row_dim {rdim}; install: {'APFS clone + header rewrite' if new_tbl_hdr else 'hard link'}")
        # the carry: the MTP mixer already in it (trunk phase) + the PLE small tensors from the table header
        ch, cbase = read_header(os.path.join(a.out, CARRY))
        carry = {}
        for n, t in ch.items():
            if n == "__metadata__": continue
            if not n.startswith(MIXER): continue   # a previous finalize's PLE tensors are re-read from the table below
            nb = t["data_offsets"][1] - t["data_offsets"][0]
            with open(os.path.join(a.out, CARRY), "rb") as f: f.seek(cbase + t["data_offsets"][0]); raw = f.read(nb)
            carry[n] = (t["dtype"], t["shape"], nb, (lambda r=raw: r))
        for n, t in nh.items():
            if n == "__metadata__" or n.endswith(".trellis"): continue
            nb = t["data_offsets"][1] - t["data_offsets"][0]
            carry[ple_rename(host_name(n))] = (t["dtype"], t["shape"], nb, file_reader(ngram_src, nbase + t["data_offsets"][0], nb))
        cm = dict(ch.get("__metadata__", {})); cm["note"] = cm.get("note", "") .split("; PLE")[0] + "; PLE small tensors from ngram_embedding.safetensors"
        nbytes = write_shard(os.path.join(a.out, CARRY), cm, [(n,) + carry[n] for n in sorted(carry)])
        print(f"rewrote {CARRY}: {len(carry)} tensors, {nbytes / 1e6:.1f} MB")
        def link(src, dst_name):
            dst = os.path.join(a.out, dst_name)
            if os.path.exists(dst): os.remove(dst)
            os.link(src, dst)
        print("table installed:", install_table(ngram_src, os.path.join(a.out, "ngram_embedding.safetensors"), new_tbl_hdr), flush=True)
        new_map = {}
        for f in vision:
            link(os.path.join(a.hybrid, f), f)
        for f in exp_files + trunk_files + vision:
            for n in read_header(os.path.join(a.out, f))[0]:
                if n != "__metadata__": new_map[n] = f
        cfg = dict(hy_cfg)
        cfg["expert_quant"] = {"format": "exl3", "k": a.expert_k, "codebook": "mul1"}
        cfg["trunk_quant"] = {"format": "exl3", "codebook": "mul1", "window": 16, "source": a.source_tag}
        cfg["ngram_table"] = {"file": "ngram_embedding.safetensors", "format": "exl3_ngram_trellis", "k": tk, "row_dim": rdim, "codebook": "mul1"}
        if new_tbl_hdr: cfg["ngram_table_note"] = "ngram_embedding.safetensors = APFS clone of turboderp's file with the header rewritten: " + tbl_desc
        cfg["vision_from"] = "hybrid"
        cfg["mtp_mixer_from"] = "turboderp"
        cfg["quantization_note"] = "quantization block kept nominal for the loader; no trunk tensor is affine (see trunk_quant)"
        cfg["qwen4_norm_convention"] = a.norm_convention
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
        for fn in os.listdir(a.hybrid):
            if fn.endswith((".json", ".jinja", ".txt", "LICENSE", "README.md")) and fn not in ("config.json", "model.safetensors.index.json") and not fn.startswith(("experts-", "trunk-")):
                src = os.path.join(a.hybrid, fn)
                if os.path.isfile(src): link(src, fn)
        json.dump({"metadata": {"total_size": 0}, "weight_map": new_map}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
        print(f"DONE: {len(new_map)} tensors indexed in {a.out}; ngram_table k {tk} row_dim {rdim}")

if __name__ == "__main__": main()
