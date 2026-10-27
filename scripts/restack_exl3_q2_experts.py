#!/usr/bin/env python3
"""Expert restack for turboderp's Qwen3.8-Flash-Next EXL3 2.05bpw_h4_ng4 (routed experts K2, MUL1). Copy of
restack_exl3_q3_experts.py with one change: --k / --mtp-k default 0 = detect K from expert 0's gate_proj trellis (last dim / 16)
of each layer; every expert of the layer must match it, and all trunk layers must share one K (exit otherwise), so the config's
single expert_quant.k holds. The q3 notes follow.
Expert restack for turboderp's Qwen3.8-Flash-Next EXL3 3.05bpw_h5_ng5 (routed experts K3, MUL1) into mlx-serve's per-layer
switch_mlp stacks. Copy of restack_exl3_q4_experts.py (the 4.05 K4 restack) with: K defaults 3/3; --raw-shard FN restacks only
the layers whose experts live in that raw shard (streaming conversion: the driver deletes each raw shard after it is restacked
and verified); --skip-existing skips finished output shards (resume). The q4 notes follow.
Derived from restack_exl3_qwen38.py (the 5.05 K5 restack), with these changes:
  * source/output/K parameterized (--src, --out, --k, --mtp-k); destination prefixes are fixed to the served pack's
    (`language_model.model`, `language_model.mtp`) instead of being read from an affine trunk pack, and NO trunk shards are
    written (the full-pack script restack_exl3_q4_full.py writes the trunk);
  * streaming: the output header is computed from the source headers first, then every expert's bytes are copied with
    seek/read straight into the output file (no per-layer stack in RAM; RSS stays ~tens of MB), fsync'd, and the page cache
    of the written shard and of the source shards it read is evicted (msync MS_INVALIDATE, as reports/find_cached.py does);
  * a memory gate between layers (free+inactive >= --mem-floor-gib, swap used <= --swap-max-mib) stops the run if it trips;
  * strict checks per expert: trellis I16 [in/16, out/16, 16*K], suh F16 [in], svh F16 [out], mul1 I32 == 0x83DCD12D.
Bytes are not changed: trellis int16 -> U16 is a reinterpretation, suh/svh are F16 as stored. Tensor order inside a shard
matches the 5.05 shards (safetensors' (dtype, name) order: F16 suh/svh first, then U16 trellis). Shard stamp
{format: exl3, k: K, codebook: mul1, window: 16}.

  python3 -I restack_exl3_q2_experts.py --src <turboderp 2.05 dir> --out <pack dir> [--k 0=auto] [--mtp-k 0=auto] [--raw-shard FN]
                                        [--skip-existing] [--layers N] [--check]
"""
import argparse, ctypes, json, os, re, struct, subprocess, sys, time

MUL1 = 0x83DCD12D
libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p; libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]; libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def evict(path):
    """Drop the file's pages from the unified buffer cache (read-only mapping + MS_INVALIDATE|MS_SYNC), like find_cached.py."""
    st = os.stat(path); fd = os.open(path, os.O_RDONLY); CH = 1 << 30
    try:
        for off in range(0, st.st_size, CH):
            n = min(CH, st.st_size - off); addr = libc.mmap(None, n, 1, 1, fd, off)
            if addr == ctypes.c_void_p(-1).value: return
            try: libc.msync(addr, n, 0x10 | 0x2)
            finally: libc.munmap(addr, n)
    finally: os.close(fd)


def mem_state():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    get = lambda k: int(re.search(k + r":\s+(\d+)", out).group(1))
    avail = (get("Pages free") + get("Pages inactive") + get("Pages speculative")) * page / 2**30
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = float(re.search(r"used = ([0-9.]+)M", sw).group(1))
    return avail, used


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=0); ap.add_argument("--mtp-k", type=int, default=0)
    ap.add_argument("--raw-shard", default="", help="only the layers with expert tensors in this source shard file")
    ap.add_argument("--skip-existing", action="store_true", help="skip output shards that already exist (complete: written via .partial + rename)")
    ap.add_argument("--layers", type=int, default=0, help="only the first N trunk layers (smoke)")
    ap.add_argument("--only", default="", help="comma list of shard stems to (re)write, e.g. 00,05,mtp-00")
    ap.add_argument("--check", action="store_true", help="validate every expert header and print the plan; write nothing")
    ap.add_argument("--mem-floor-gib", type=float, default=12.0); ap.add_argument("--swap-max-mib", type=float, default=1500.0)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    cfg = json.load(open(os.path.join(a.src, "config.json"))); tc = cfg.get("text_config", cfg)
    n_layers, E = tc["num_hidden_layers"], tc["num_experts"]
    H, I = tc["hidden_size"], tc["moe_intermediate_size"]
    if a.layers: n_layers = min(n_layers, a.layers)
    hdrs = {}
    def meta(name):
        fn = idx[name]
        if fn not in hdrs: hdrs[fn] = read_header(os.path.join(a.src, fn))
        h, base = hdrs[fn]; return fn, h[name], base
    projs = (("gate_proj", H, I), ("up_proj", H, I), ("down_proj", I, H))
    jobs = [("model.language_model", "language_model.model", L, f"experts-{L:02d}.safetensors", a.k) for L in range(n_layers)]
    if not a.layers: jobs.append(("mtp", "language_model.mtp", 0, "experts-mtp-00.safetensors", a.mtp_k))
    if a.only:
        want = {f"experts-{s}.safetensors" for s in a.only.split(",")}; jobs = [j for j in jobs if j[3] in want]
    if a.raw_shard:
        in_raw = set()
        for name, fn in idx.items():
            if fn != a.raw_shard: continue
            m = re.match(r"^(model\.language_model|mtp)\.layers\.(\d+)\.mlp\.experts\.", name)
            if m: in_raw.add((m.group(1), int(m.group(2))))
        jobs = [j for j in jobs if (j[0], j[2]) in in_raw]
        print(f"--raw-shard {a.raw_shard}: {len(jobs)} expert shards: {[j[3] for j in jobs]}")
    if a.skip_existing:
        done = [j[3] for j in jobs if os.path.exists(os.path.join(a.out, j[3]))]
        if done: print(f"--skip-existing: {len(done)} already written: {done}")
        jobs = [j for j in jobs if j[3] not in done]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    trunk_ks = set()
    for src_pfx, dst_pfx, L, shard, K in jobs:
        if K == 0:   # auto: expert 0's gate_proj trellis decides; every expert of the layer is checked against it below
            K = meta(f"{src_pfx}.layers.{L}.mlp.experts.0.gate_proj.trellis")[1]["shape"][-1] // 16
        if src_pfx != "mtp":
            trunk_ks.add(K)
            if len(trunk_ks) > 1: sys.exit(f"layer {L}: expert K {K} differs from the other trunk layers' {sorted(trunk_ks)}")
        n = 16 * K
        plan = {}  # dst name -> (dtype, shape, [(file, abs offset, nbytes)] in expert order)
        mulfile = {}
        for P, fin, fout in projs:
            parts = {"trellis": [], "suh": [], "svh": []}
            for e in range(E):
                b = f"{src_pfx}.layers.{L}.mlp.experts.{e}.{P}"
                fn, mt, base = meta(b + ".trellis")
                if mt["dtype"] != "I16" or mt["shape"] != [fin // 16, fout // 16, n]:
                    sys.exit(f"{b}.trellis: {mt['dtype']} {mt['shape']}, expected I16 [{fin // 16}, {fout // 16}, {n}] (K{K})")
                parts["trellis"].append((fn, base + mt["data_offsets"][0], mt["data_offsets"][1] - mt["data_offsets"][0]))
                for leaf, dim in (("suh", fin), ("svh", fout)):
                    fn2, m2, base2 = meta(b + "." + leaf)
                    if m2["dtype"] != "F16" or m2["shape"] != [dim]: sys.exit(f"{b}.{leaf}: {m2['dtype']} {m2['shape']}, expected F16 [{dim}]")
                    parts[leaf].append((fn2, base2 + m2["data_offsets"][0], m2["data_offsets"][1] - m2["data_offsets"][0]))
                fn3, m3, base3 = meta(b + ".mul1")
                if m3["dtype"] != "I32" or m3["data_offsets"][1] - m3["data_offsets"][0] != 4: sys.exit(f"{b}.mul1: {m3}")
                mulfile.setdefault(fn3, []).append((b, base3 + m3["data_offsets"][0]))
            d = f"{dst_pfx}.layers.{L}.mlp.switch_mlp.{P}"
            plan[d + ".trellis"] = ("U16", [E, fin // 16, fout // 16, n], parts["trellis"])
            plan[d + ".suh"] = ("F16", [E, fin], parts["suh"])
            plan[d + ".svh"] = ("F16", [E, fout], parts["svh"])
        # mul1 values (tiny reads)
        nm = 0
        for fn, items in mulfile.items():
            with open(os.path.join(a.src, fn), "rb") as f:
                for b, off in items:
                    f.seek(off); v = struct.unpack("<I", f.read(4))[0]; nm += 1
                    if v != MUL1: sys.exit(f"{b}.mul1 = {v:#x}, not 0x83DCD12D")
        order = sorted(plan, key=lambda k: (plan[k][0], k))
        header = {"__metadata__": {"format": "exl3", "k": str(K), "codebook": "mul1", "window": "16"}}; off = 0
        for k in order:
            dt, shp, parts = plan[k]; nb = sum(p[2] for p in parts)
            want = 2 * 1
            for s in shp: want *= s
            if nb != want: sys.exit(f"{k}: {nb} bytes, shape {shp} needs {want}")
            header[k] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + nb]}; off += nb
        srcs = sorted({p[0] for v in plan.values() for p in v[2]})
        print(f"{shard}: K{K} n{n}, {len(order)} tensors, {off / 1e9:.3f} GB, {E} experts x 3 proj, mul1 ok x{nm}, from {srcs}", flush=True)
        if a.check: continue
        avail, used = mem_state()
        print(f"  mem free+inactive {avail:.1f} GiB, swap used {used:.0f} MiB", flush=True)
        if avail < a.mem_floor_gib or used > a.swap_max_mib:
            sys.exit(f"STOP: memory gate (free+inactive {avail:.1f} GiB < {a.mem_floor_gib} or swap {used:.0f} MiB > {a.swap_max_mib})")
        hb = json.dumps(header, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
        tmp = os.path.join(a.out, shard + ".partial")
        fh = {fn: open(os.path.join(a.src, fn), "rb") for fn in srcs}
        try:
            with open(tmp, "wb") as fo:
                fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
                for k in order:
                    for fn, so, nb in plan[k][2]:
                        f = fh[fn]; f.seek(so); buf = f.read(nb)
                        if len(buf) != nb: sys.exit(f"short read {fn} @{so}")
                        fo.write(buf)
                fo.flush(); os.fsync(fo.fileno())
        finally:
            for f in fh.values(): f.close()
        if os.path.getsize(tmp) != 8 + len(hb) + off: sys.exit(f"{tmp}: size mismatch")
        os.replace(tmp, os.path.join(a.out, shard))
        evict(os.path.join(a.out, shard))
        for fn in srcs: evict(os.path.join(a.src, fn))
        print(f"  wrote {shard} ({time.time() - t0:.0f} s elapsed)", flush=True)
    print("EXPERTS_DONE" if not a.check else "CHECK_DONE")


if __name__ == "__main__":
    main()
