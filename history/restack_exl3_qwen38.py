#!/usr/bin/env python3
"""Build an mlx-serve pack for Qwen3.8-Flash-Next whose routed experts are turboderp's EXL3 trellis (K5, MUL1) and whose
every other tensor comes from an existing mlx-serve affine pack (the trunk, router, shared expert, MTP, embeddings, head).

  python3 restack_exl3_qwen38.py --exl3 <turboderp branch dir> --trunk <mlx-serve affine pack dir> --out <pack dir>
                                 [--ngram <ngram_table.bin to hard-link> --ngram-bits B --ngram-group G] [--k 5]

Per layer L (trunk 0..47 and the MTP layer), each projection P in gate_proj/up_proj/down_proj: the E per-expert
`trellis` int16 [in/16, out/16, n] are stacked to U16 [E, in/16, out/16, n], `suh` f16 to [E, in], `svh` f16 to
[E, out], named `<prefix>.layers.L.mlp.switch_mlp.P.{trellis,suh,svh}` (the pack-format contract lib/sushi/docs/
pack-format.md; turboderp stores no separate scale: suh/svh are used as stored). Every expert's `mul1` scalar must be
the MUL1 constant 0x83DCD12D. Each experts shard carries the stamp {format: exl3, k, codebook: mul1, window: 16}.
The trunk pack's shards are rewritten without their affine `switch_mlp` tensors; config.json gains
`expert_quant = {format: exl3, k: K, codebook: mul1}` and keeps the trunk's `quantization`. Memory: one source shard
plus one layer's experts at a time (< 10 GB); run under nice.
"""
import argparse, json, os, re, shutil, struct, sys
import numpy as np
from safetensors.numpy import save_file

MUL1_MULT = 0x83DCD12D
DT = {"I16": np.int16, "U16": np.uint16, "F16": np.float16, "BF16": np.uint16, "F32": np.float32, "I32": np.int32, "I64": np.int64, "U32": np.uint32, "U8": np.uint8, "BOOL": np.bool_}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


class ShardReader:
    """Raw tensor reads by (file, name) through a memmap per shard; dtype-agnostic (BF16 stays raw u16)."""
    def __init__(self, root, index):
        self.root, self.index = root, index
        self.maps = {}

    def tensor(self, name):
        fn = self.index[name]
        if fn not in self.maps:
            hdr, base = read_header(os.path.join(self.root, fn))
            self.maps[fn] = (hdr, base, np.memmap(os.path.join(self.root, fn), dtype=np.uint8, mode="r"))
        hdr, base, mm = self.maps[fn]
        meta = hdr[name]
        a, b = meta["data_offsets"]
        raw = mm[base + a: base + b]
        dt = DT[meta["dtype"]]
        return np.frombuffer(raw, dtype=dt).reshape(meta["shape"]), meta["dtype"]

    def release(self, keep=()):
        for fn in list(self.maps):
            if fn not in keep:
                del self.maps[fn]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exl3", required=True)
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=5)
    ap.add_argument("--ngram", help="ngram_table.bin to hard-link into the pack (default: the trunk pack's)")
    ap.add_argument("--ngram-bits", type=int)
    ap.add_argument("--ngram-group", type=int)
    ap.add_argument("--layers", type=int, default=0, help="restack only the first N trunk layers (a smoke run)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ex_idx = json.load(open(os.path.join(a.exl3, "model.safetensors.index.json")))["weight_map"]
    tr_idx = json.load(open(os.path.join(a.trunk, "model.safetensors.index.json")))["weight_map"]
    tr_cfg = json.load(open(os.path.join(a.trunk, "config.json")))
    tc = tr_cfg.get("text_config", tr_cfg)
    n_layers, n_experts = tc["num_hidden_layers"], tc["num_experts"]
    hidden, inter = tc["hidden_size"], tc["moe_intermediate_size"]
    if a.layers:
        n_layers = min(n_layers, a.layers)

    # the trunk pack's own prefixes, from its router and MTP router names
    def prefix_of(pattern):
        for k in tr_idx:
            m = re.match(r"^(.*)\." + pattern + r"$", k)
            if m:
                return m.group(1)
        return None
    trunk_prefix = None
    for k in tr_idx:
        m = re.match(r"^(.*)\.layers\.0\.mlp\.gate\.weight$", k)
        if m and "mtp" not in m.group(1):
            trunk_prefix = m.group(1)
    mtp_prefix = None
    for k in tr_idx:
        m = re.match(r"^(.*mtp.*)\.layers\.0\.mlp\.gate\.weight$", k)
        if m:
            mtp_prefix = m.group(1)
    if trunk_prefix is None:
        sys.exit("trunk pack: no layers.0.mlp.gate.weight")
    # turboderp's names
    ex_prefix = None
    for k in ex_idx:
        m = re.match(r"^(.*)\.layers\.0\.mlp\.experts\.0\.gate_proj\.trellis$", k)
        if m and "mtp" not in m.group(1):
            ex_prefix = m.group(1)
    ex_mtp_prefix = None
    for k in ex_idx:
        m = re.match(r"^(.*)\.layers\.0\.mlp\.experts\.0\.gate_proj\.trellis$", k)
        if m and "mtp" in m.group(1):
            ex_mtp_prefix = m.group(1)
    print(f"trunk prefix {trunk_prefix!r}, mtp prefix {mtp_prefix!r}; exl3 prefix {ex_prefix!r}, exl3 mtp prefix {ex_mtp_prefix!r}")
    print(f"{n_layers} layers x {n_experts} experts, hidden {hidden}, inter {inter}, k {a.k}")

    rd = ShardReader(a.exl3, ex_idx)
    out_index = {}
    stamp = {"format": "exl3", "k": str(int(a.k)) if float(a.k).is_integer() else str(a.k), "codebook": "mul1", "window": "16"}
    projs = (("gate_proj", hidden, inter), ("up_proj", hidden, inter), ("down_proj", inter, hidden))
    n_expected = None

    def restack(layer_src_prefix, layer_dst_prefix, L, shard_name):
        nonlocal n_expected
        tensors = {}
        for P, fin, fout in projs:
            tr, su, sv = [], [], []
            for e in range(n_experts):
                base = f"{layer_src_prefix}.layers.{L}.mlp.experts.{e}.{P}"
                t, tdt = rd.tensor(base + ".trellis")
                s, _ = rd.tensor(base + ".suh")
                v, _ = rd.tensor(base + ".svh")
                m, _ = rd.tensor(base + ".mul1")
                if int(m.reshape(-1)[0]) & 0xFFFFFFFF != MUL1_MULT:
                    sys.exit(f"{base}: mul1 {int(m.reshape(-1)[0]) & 0xFFFFFFFF:#x} is not the MUL1 constant")
                if t.shape[0] != fin // 16 or t.shape[1] != fout // 16 or s.shape != (fin,) or v.shape != (fout,):
                    sys.exit(f"{base}: geometry {t.shape} {s.shape} {v.shape}, expected [{fin // 16}, {fout // 16}, n] [{fin}] [{fout}]")
                n = t.shape[2]
                if n_expected is None:
                    n_expected = n
                    if n != int(a.k * 16):
                        print(f"note: trellis n = {n} (K {n / 16}); --k {a.k} is the ceiling the pack bills")
                elif n != n_expected:
                    sys.exit(f"{base}: rate n {n} differs from {n_expected} (rate groups are not written by this script)")
                tr.append(t.view(np.uint16))
                su.append(s.astype(np.float16, copy=False))
                sv.append(v.astype(np.float16, copy=False))
            dst = f"{layer_dst_prefix}.layers.{L}.mlp.switch_mlp.{P}"
            tensors[dst + ".trellis"] = np.stack(tr)
            tensors[dst + ".suh"] = np.stack(su)
            tensors[dst + ".svh"] = np.stack(sv)
        save_file(tensors, os.path.join(a.out, shard_name), metadata=stamp)
        for k in tensors:
            out_index[k] = shard_name
        nbytes = sum(t.nbytes for t in tensors.values())
        print(f"  {shard_name}: {len(tensors)} tensors, {nbytes / 1e9:.2f} GB, n = {n_expected}", flush=True)
        rd.release()

    for L in range(n_layers):
        restack(ex_prefix, trunk_prefix, L, f"experts-{L:02d}.safetensors")
    if mtp_prefix and ex_mtp_prefix and not a.layers:
        restack(ex_mtp_prefix, mtp_prefix, 0, "experts-mtp-00.safetensors")
    elif not a.layers:
        print("WARNING: no MTP router in the trunk pack or no MTP experts in the EXL3 pack; MTP experts not written")

    # the trunk pack's shards without their affine routed experts (raw bytes through, every dtype kept)
    drop_re = re.compile(r"\.layers\.\d+\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$")
    by_shard = {}
    for k, fn in tr_idx.items():
        by_shard.setdefault(fn, []).append(k)
    raw = ShardReader(a.trunk, tr_idx)
    for fn in sorted(by_shard):
        keep = [k for k in by_shard[fn] if not drop_re.search(k)]
        if a.layers:
            def early(k):
                m = re.search(r"\.layers\.(\d+)\.", k)
                return m is None or "mtp" in k or int(m.group(1)) < n_layers
            keep = [k for k in keep if early(k)]
        if not keep:
            continue
        hdr, _ = read_header(os.path.join(a.trunk, fn))
        tensors = {k: raw.tensor(k)[0] for k in keep}
        meta = dict(hdr.get("__metadata__", {}) or {})
        out_name = "trunk-" + fn
        write_shard(os.path.join(a.out, out_name), tensors, {k: hdr[k]["dtype"] for k in keep}, meta)
        for k in keep:
            out_index[k] = out_name
        print(f"  {out_name}: {len(keep)} tensors kept, {len(by_shard[fn]) - len(keep)} affine expert tensors dropped", flush=True)
        del tensors
        raw.release()

    # config, index, the small files, the n-gram table
    cfg = dict(tr_cfg)
    cfg["expert_quant"] = {"format": "exl3", "k": int(a.k) if float(a.k).is_integer() else a.k, "codebook": "mul1"}
    if a.ngram:
        cfg["ngram_table"] = {"file": "ngram_table.bin", "bits": a.ngram_bits, "group_size": a.ngram_group} if a.ngram_bits else cfg.get("ngram_table")
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    json.dump({"metadata": {"format": "pt"}, "weight_map": out_index}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"))
    for fn in os.listdir(a.trunk):
        p = os.path.join(a.trunk, fn)
        if os.path.isfile(p) and not fn.endswith(".safetensors") and fn not in ("config.json", "model.safetensors.index.json", "ngram_table.bin") and not fn.startswith("."):
            shutil.copy2(p, os.path.join(a.out, fn))
    ng = a.ngram or os.path.join(a.trunk, "ngram_table.bin")
    if os.path.exists(ng) and not os.path.exists(os.path.join(a.out, "ngram_table.bin")):
        os.link(ng, os.path.join(a.out, "ngram_table.bin"))
    print("done:", a.out)


def write_shard(path, tensors, dtypes, metadata):
    """A safetensors writer that keeps BF16 (raw u16) as BF16."""
    header = {}
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    off = 0
    order = sorted(tensors)
    for k in order:
        t = np.ascontiguousarray(tensors[k])
        header[k] = {"dtype": dtypes[k], "shape": list(t.shape), "data_offsets": [off, off + t.nbytes]}
        off += t.nbytes
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for k in order:
            f.write(np.ascontiguousarray(tensors[k]).tobytes())


if __name__ == "__main__":
    main()
