#!/usr/bin/env python3
"""Repack an exllamav3 EXL3 quantization of Ricursive Ripple-65B-Preview for mlx-serve's turboexl3_ricursive plugin.

Copies bytes; quantizes nothing. The tensor mapping is the Qwen packs' (restack_exl3_q2_full.py):
  * each EXL3 module `<m>.{trellis,suh,svh,mul1}` becomes `<m>.weight` (the trellis, I16 relabelled U16: the same bytes,
    [in/16, out/16, n]), `<m>.scales` (suh, F16 [in]) and `<m>.biases` (svh, F16 [out]); mul1 is asserted 0x83DCD12D
    and dropped;
  * every other tensor (the norms, value_res_lambda, the embedding: BF16 in exllamav3's output) is copied as it is; any
    other dtype stops the script;
  * the names stay the checkpoint's (`model.layers.N.self_attn.q_proj`, `lm_head`, ...).
Each output shard holds the tensors of the source shard of the same name, in the same order. config.json is the source's
without `quantization_config`, plus `trunk_quant` (the plugin's claim marker). The tokenizer and template files are hard
links. After writing, every output tensor is read back and compared with its source bytes.

  python3 -I ripple_exl3_to_mlx_serve.py --src <exllamav3 output dir> --out <pack dir> [--source-tag <tag>] [--check]

--check skips the writing and only compares an existing --out with --src.
"""
import argparse, hashlib, json, os, struct, sys

MUL1 = 0x83DCD12D
EXL3_LEAVES = {"trellis", "suh", "svh", "mul1"}
NEW_LEAF = {"trellis": ("weight", "U16"), "suh": ("scales", "F16"), "svh": ("biases", "F16")}
LINKED = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
          "generation_config.json", "LICENSE", "README.md")
CHUNK = 64 << 20


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def shards(src):
    idx = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx):
        names = sorted(set(json.load(open(idx))["weight_map"].values()))
    else:
        names = sorted(n for n in os.listdir(src) if n.endswith(".safetensors"))
    if not names:
        sys.exit(f"no safetensors in {src}")
    return names


def plan_shard(path):
    """[(new name, dtype, shape, src offset, nbytes, conv)] for one source shard, in its order, with mul1 checked."""
    h, base = read_header(path)
    items, mods = [], {}
    for name, t in h.items():
        if name == "__metadata__":
            continue
        stem, _, leaf = name.rpartition(".")
        if leaf in EXL3_LEAVES:
            mods.setdefault(stem, {})[leaf] = t
    for stem, leaves in mods.items():
        if set(leaves) != EXL3_LEAVES:
            sys.exit(f"{stem}: EXL3 module with leaves {sorted(leaves)}")
        m = leaves["mul1"]
        if m["dtype"] != "I32" or m["data_offsets"][1] - m["data_offsets"][0] != 4:
            sys.exit(f"{stem}.mul1: {m['dtype']} {m['shape']}")
        with open(path, "rb") as f:
            f.seek(base + m["data_offsets"][0]); v = struct.unpack("<I", f.read(4))[0]
        if v != MUL1:
            sys.exit(f"{stem}.mul1 = {v:#x}, not {MUL1:#x}")
        tr, su, sv = leaves["trellis"], leaves["suh"], leaves["svh"]
        if tr["dtype"] not in ("I16", "U16") or len(tr["shape"]) != 3:
            sys.exit(f"{stem}.trellis: {tr['dtype']} {tr['shape']}")
        if su["dtype"] != "F16" or sv["dtype"] != "F16" or su["shape"] != [tr["shape"][0] * 16] or sv["shape"] != [tr["shape"][1] * 16]:
            sys.exit(f"{stem}: suh {su['dtype']} {su['shape']} svh {sv['dtype']} {sv['shape']} for trellis {tr['shape']}")
    for name, t in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0] if kv[0] != "__metadata__" else -1):
        if name == "__metadata__":
            continue
        a, b = t["data_offsets"]
        stem, _, leaf = name.rpartition(".")
        if stem in mods:
            if leaf == "mul1":
                continue
            new_leaf, dt = NEW_LEAF[leaf]
            items.append((f"{stem}.{new_leaf}", dt, t["shape"], base + a, b - a, None))
        elif t["dtype"] == "BF16":
            items.append((name, t["dtype"], t["shape"], base + a, b - a, None))
        else:
            sys.exit(f"{name}: dtype {t['dtype']}")
    return items, h.get("__metadata__") or {}


def copy_range(src, off, nb, out, conv):
    with open(src, "rb") as f:
        f.seek(off)
        if conv:
            out.write(conv(f.read(nb))); return
        left = nb
        while left:
            buf = f.read(min(CHUNK, left))
            if not buf:
                sys.exit(f"{src}: short read at {off}")
            out.write(buf); left -= len(buf)


def write_shard(src_path, dst_path, items, meta):
    h = {"__metadata__": {"format": "pt", **{k: v for k, v in meta.items() if k != "format"}}}
    off = 0
    for n, dt, shp, _, nb, _ in items:
        h[n] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + nb]}; off += nb
    hb = json.dumps(h, separators=(",", ":")).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = dst_path + ".partial"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for _, _, _, so, nb, conv in items:
            copy_range(src_path, so, nb, f, conv)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, dst_path)


def digest(path, off, nb, conv=None):
    hs = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        if conv:
            hs.update(conv(f.read(nb))); return hs.hexdigest()
        left = nb
        while left:
            buf = f.read(min(CHUNK, left)); hs.update(buf); left -= len(buf)
    return hs.hexdigest()


def check_shard(src_path, dst_path, items):
    h, base = read_header(dst_path)
    names = [k for k in h if k != "__metadata__"]
    if sorted(names) != sorted(i[0] for i in items):
        sys.exit(f"{dst_path}: tensor names differ from the plan")
    for n, dt, shp, so, nb, conv in items:
        t = h[n]; a, b = t["data_offsets"]
        if t["dtype"] != dt or t["shape"] != shp or b - a != nb:
            sys.exit(f"{dst_path}:{n}: {t} vs {dt} {shp} {nb}")
        if digest(src_path, so, nb, conv) != digest(dst_path, base + a, nb):
            sys.exit(f"{dst_path}:{n}: bytes differ from the source")
    return len(items)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--source-tag", default="Ripple-65B-Preview-exl3-3.0bpw-h6 (exllamav3 convert.py -b 3.0 -hb 6)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    weight_map, total, checked = {}, 0, 0
    for sh in shards(a.src):
        sp, dp = os.path.join(a.src, sh), os.path.join(a.out, sh)
        items, meta = plan_shard(sp)
        if not a.check:
            write_shard(sp, dp, items, meta)
        checked += check_shard(sp, dp, items)
        for n, _, _, _, nb, _ in items:
            weight_map[n] = sh; total += nb
        print(f"{sh}: {len(items)} tensors, {sum(i[4] for i in items) / 2**30:.2f} GiB, bytes equal to the source", flush=True)
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    qc = cfg.pop("quantization_config", None)
    if cfg.get("model_type") != "ricursive":
        sys.exit(f"model_type {cfg.get('model_type')}")
    cfg["trunk_quant"] = {"format": "exl3", "codebook": "mul1", "window": 16, "source": a.source_tag}
    if qc is not None:
        cfg["trunk_quant_note"] = "exllamav3 quantization_config: " + json.dumps(qc, separators=(",", ":"))
    if not a.check:
        with open(os.path.join(a.out, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2); f.write("\n")
        with open(os.path.join(a.out, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=2); f.write("\n")
        for n in LINKED:
            s, d = os.path.join(a.src, n), os.path.join(a.out, n)
            if os.path.exists(s) and not os.path.exists(d):
                os.link(s, d)
    print(f"done: {checked} tensors checked, {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
