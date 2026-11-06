#!/usr/bin/env python3
"""Repack turboderp's Ling-3.1-flash EXL3 branch (turboderp/Ling-3.1-flash-exl3) into the streaming pack that
turboexl3_ling's copy of mlx-serve's expert streamer reads. It copies bytes and does not quantize. CPU only.

What it writes, in --out:
- experts.bin: the routed experts of trunk layers 4..71 (bank layer b = model layer - 4), 512 experts a layer, as
  bank-v2 records: per record the nine segments gate code, rout, rin, up code, rout, rin, down code, rout, rin (code =
  exllamav3 trellis I16 [in/16, out/16, 16K], rout = svh F16 [out], rin = suh F16 [in]). The plan's layout:
  - classic (the default; mlx-serve's expert/bank.zig layerSegments, bankv2.py): the nine segments back to back in that
    order from offset 0, the record zero-padded to 4096;
  - a4k (turboexl3_ds41's expert/a4k.zig, assemble_sc281_a4k.py): a zero lead of (-(the gate/up scales)) mod 4096, the
    gate/up span [gate.rout gate.rin up.rout up.rin gate.code up.code], the down span [down.code down.rout down.rin],
    zero padding to 4096.
  Every expert's three mul1 words are checked equal to 0x83DCD12D and dropped. K is a number per layer (2.5 or 2).
- expert-manifest-v2.json (mtplx-expert-manifest-v2; `sidecar.layout` "a4k" only in an a4k bank) and
  expert-manifest.json (the v1 runtime manifest, absolute segment offsets), in the key structure of the DeepSeek-V4.1
  streaming packs (bankv2.py, build_v1_manifest.py, assemble_sc281_a4k.py).
- model-000NN-of-000MM-nonexpert.safetensors: each source shard's tensors other than the routed experts and the MTP
  layer, byte for byte (names, dtypes, shapes unchanged); mtp-nonbank.safetensors: every tensor of the MTP layer
  (model.layers.<num_hidden_layers>.*), its experts included; model.safetensors.index.json naming all of them.
- The small files (config.json unchanged, tokenizer, chat template, README, the modeling code) hard-linked.
- receipts/<shard>.json per shard, receipts/state.json, LING-REPACK-RECEIPT.json at the end.

Checks: a shard is taken only when its size is the branch's LFS size and the download lane's <shard>.sha256ok holds
the branch's LFS sha256 (else the shard is hashed here); each record's header geometry (dtypes, shapes, K constant per
layer and equal to the plan's, mul1); after the shard's records are written and fsynced, every record is read back and
each part compared byte for byte with the source span read again, the lead and the tail checked zero, and the padded
sha256 checked; each non-expert file is read back against the source tensors.

Usage (on the box):
  ling_exl3_to_streaming.py plan  --src DIR --rev 2.60bpw --plan PLAN.json [--layout classic|a4k]   (HF -> plan)
  ling_exl3_to_streaming.py run   --src DIR --out DIR --plan PLAN.json [--only SHARD] [--watch SECONDS] [--threads N]
  ling_exl3_to_streaming.py finalize --src DIR --out DIR --plan PLAN.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import fcntl
import hashlib
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.request

REPO = 'turboderp/Ling-3.1-flash-exl3'
FORMAT_V2 = 'mtplx-expert-manifest-v2'
FORMAT_V1 = 'mtplx-expert-manifest-v1'
MODEL_KEY = 'ling-3.1-flash-expert-exl3'
ALIGN = 4096
MUL1_MULT = 0x83DCD12D
F_NOCACHE = getattr(fcntl, 'F_NOCACHE', 48)
F_FULLFSYNC = getattr(fcntl, 'F_FULLFSYNC', 51)
CHUNK = 16 << 20
PROJ = ('gate_proj', 'up_proj', 'down_proj')
COMPONENTS = [f'{p}.{c}' for p in PROJ for c in ('code', 'rout', 'rin')]
LEAF = {'code': 'trellis', 'rout': 'svh', 'rin': 'suh'}
# a4k on-disk order (component indices), expert/a4k.zig disk_order
DISK = [1, 2, 4, 5, 0, 3, 6, 7, 8]
EXPERT_RE = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(suh|svh|trellis|mul1|mcg)$')
LAYER_RE = re.compile(r'^model\.layers\.(\d+)\.')
SMALL_FILES = ('config.json', 'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
               'README.md', 'configuration_bailing_moe_v3.py', 'modeling_bailing_moe_v3.py', '.gitattributes')


def log(msg):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), msg, flush=True)


def nocache(fd):
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    except OSError:
        pass


def full_fsync(fd):
    try:
        fcntl.fcntl(fd, F_FULLFSYNC)
    except OSError:
        os.fsync(fd)


def atomic_json(path, obj, compact=False):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        if compact:
            json.dump(obj, f, separators=(',', ':'))
        else:
            json.dump(obj, f, indent=1)
        f.flush()
        full_fsync(f.fileno())
    os.replace(tmp, path)


def pread_all(fd, n, off):
    out = bytearray()
    while len(out) < n:
        b = os.pread(fd, min(CHUNK, n - len(out)), off + len(out))
        if not b:
            raise IOError(f'short read at {off + len(out)}')
        out += b
    return bytes(out)


def sha256_file(path):
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    nocache(fd)
    try:
        off = 0
        while True:
            b = os.pread(fd, CHUNK, off)
            if not b:
                break
            h.update(b)
            off += len(b)
    finally:
        os.close(fd)
    return h.hexdigest()


def read_header(fd):
    n = struct.unpack('<Q', os.pread(fd, 8, 0))[0]
    h = json.loads(os.pread(fd, n, 8))
    meta = h.pop('__metadata__', None)
    return h, 8 + n, meta


# ------------------------------------------------------------------------------------------------ geometry

def code_halfwords(K):
    n = round(16 * K)
    if abs(16 * K - n) > 1e-9 or n % 2:
        raise ValueError(f'K={K} is not an even number of halfwords a tile')
    return n


def classic_segments(K, hidden, inter):
    """bankv2.layer_segments at a K that may be fractional (n = 16 K halfwords a tile): the nine segments in component
    order, back to back -> (segments, logical_bytes)."""
    n = code_halfwords(K)
    segs, off = [], 0
    for p in PROJ:
        i, o = (inter, hidden) if p == 'down_proj' else (hidden, inter)
        for comp, dtype, shape, nbytes in (('code', 'I16', [i // 16, o // 16, n], (i // 16) * (o // 16) * n * 2),
                                           ('rout', 'F16', [o], o * 2), ('rin', 'F16', [i], i * 2)):
            segs.append({'component': f'{p}.{comp}', 'dtype': dtype, 'shape': shape, 'offset': off, 'length': nbytes})
            off += nbytes
    return segs, off


def record_layout(K, hidden, inter, layout):
    """The record geometry in `layout`: classic = bankv2.layer_segments (lead 0, extent = the logical bytes)."""
    if layout == 'a4k':
        return a4k_layer(K, hidden, inter)
    if layout != 'classic':
        raise ValueError(f'layout {layout!r} (classic or a4k)')
    segs, logical = classic_segments(K, hidden, inter)
    return {'segments': segs, 'lead': 0, 'extent': logical, 'record_bytes': (logical + ALIGN - 1) // ALIGN * ALIGN,
            'logical_bytes': logical}


def a4k_layer(K, hidden, inter):
    """The a4k geometry (expert/a4k.zig `geometry`): segments in component order at their a4k offsets, lead, extent,
    record_bytes, logical_bytes."""
    segs, logical = classic_segments(K, hidden, inter)
    lens = [s['length'] for s in segs]
    for c in (0, 3, 6):
        if lens[c] == 0 or lens[c] % ALIGN:
            raise ValueError(f'K={K}: {COMPONENTS[c]} is {lens[c]} B, not a multiple of {ALIGN}: no a4k geometry')
    lead = (ALIGN - sum(lens[c] for c in DISK[:4]) % ALIGN) % ALIGN
    pos = lead
    off = [0] * 9
    for c in DISK:
        off[c] = pos
        pos += lens[c]
    a = []
    for c, s in enumerate(segs):
        t = dict(s)
        t['offset'] = off[c]
        a.append(t)
    return {'segments': a, 'lead': lead, 'extent': pos, 'record_bytes': (pos + ALIGN - 1) // ALIGN * ALIGN,
            'logical_bytes': logical}


def layer_table(k_by_layer, hidden, inter, n_experts, layout):
    out, base = [], 0
    for b, K in enumerate(k_by_layer):
        g = record_layout(K, hidden, inter, layout)
        out.append({'layer': b, 'K': K, 'record_bytes': g['record_bytes'], 'logical_bytes': g['logical_bytes'],
                    'base_offset': base, 'segments': g['segments'], '_lead': g['lead'], '_extent': g['extent']})
        base += n_experts * g['record_bytes']
    return out, base


def f16s(buf, seg):
    """A segment's F16 values as Python floats (the stats the DeepSeek converters keep: max |rin|, rout's zeros and range)."""
    n = seg['length'] // 2
    return struct.unpack_from(f'<{n}e', buf, seg['offset'])


def public(t):
    return {k: v for k, v in t.items() if not k.startswith('_')}


# ------------------------------------------------------------------------------------------------ plan

def http_json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def http_range(url, a, b):
    req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b}'})
    with urllib.request.urlopen(req) as r:
        return r.read()


def make_plan(args):
    cfg = json.load(open(os.path.join(args.src, 'config.json')))
    rev = http_json(f'https://huggingface.co/api/models/{REPO}/revision/{args.rev}')['sha']
    tree = http_json(f'https://huggingface.co/api/models/{REPO}/tree/{rev}?recursive=true')
    shards = {}
    for x in tree:
        if x['type'] == 'file' and x['path'].endswith('.safetensors'):
            shards[x['path']] = {'size': x['size'], 'sha256': x['lfs']['oid']}
    n_layers = cfg['num_hidden_layers']
    first = cfg['first_k_dense_replace']
    k_layer = {}
    homes = {}
    for name in sorted(shards):
        url = f'https://huggingface.co/{REPO}/resolve/{rev}/{name}'
        n = struct.unpack('<Q', http_range(url, 0, 7))[0]
        hdr = json.loads(http_range(url, 8, 7 + n))
        hdr.pop('__metadata__', None)
        for k, v in hdr.items():
            m = EXPERT_RE.match(k)
            if m and m.group(4) == 'trellis' and int(m.group(1)) < n_layers:
                L = int(m.group(1))
                k_layer.setdefault(L, set()).add(v['shape'][-1] / 16)
                homes.setdefault(L, set()).add(name)
    k_by_layer = []
    for L in range(first, n_layers):
        ks = k_layer.get(L)
        if not ks or len(ks) != 1:
            raise SystemExit(f'layer {L}: rates {ks} (want exactly one)')
        K = ks.pop()
        k_by_layer.append(int(K) if K == int(K) else K)
    plan = {'repo': REPO, 'branch': args.rev, 'revision': rev, 'exllamav3': cfg['quantization_config'].get('version', ''),
            'codebook': cfg['quantization_config'].get('codebook'), 'codebook_multiplier': MUL1_MULT,
            'model_key': MODEL_KEY, 'hidden': cfg['hidden_size'], 'inter': cfg['moe_intermediate_size'],
            'n_experts': cfg['num_experts'], 'first_moe_layer': first, 'num_hidden_layers': n_layers,
            'mtp_layer': n_layers, 'layout': args.layout, 'k_by_layer': k_by_layer, 'shards': shards,
            'layer_shards': {str(L): sorted(v) for L, v in sorted(homes.items())}}
    if plan['codebook'] != 'mul1':
        raise SystemExit(f'codebook {plan["codebook"]} (want mul1)')
    atomic_json(args.plan, plan)
    table, total = layer_table(k_by_layer, plan['hidden'], plan['inter'], plan['n_experts'], plan['layout'])
    for K in sorted(set(k_by_layer)):
        t = next(x for x in table if x['K'] == K)
        log(f'PLAN {plan["layout"]} K{K}: record {t["record_bytes"]} B (logical {t["logical_bytes"]}, lead {t["_lead"]}, extent {t["_extent"]})')
    log(f'PLAN {args.plan}: revision {rev}, {len(shards)} shards, {len(k_by_layer)} bank layers, experts.bin {total} B')


# ------------------------------------------------------------------------------------------------ conversion

class Converter:
    def __init__(self, args):
        self.args = args
        self.plan = json.load(open(args.plan))
        p = self.plan
        self.src, self.out = os.path.abspath(args.src), os.path.abspath(args.out)
        self.first, self.mtp = p['first_moe_layer'], p['mtp_layer']
        self.layout = p.get('layout', 'classic')
        self.table, self.total = layer_table(p['k_by_layer'], p['hidden'], p['inter'], p['n_experts'], self.layout)
        os.makedirs(os.path.join(self.out, 'receipts'), exist_ok=True)
        self.state_path = os.path.join(self.out, 'receipts', 'state.json')
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {
            'repo': p['repo'], 'revision': p['revision'], 'layout': self.layout, 'shards': {}}
        if self.state.get('revision') != p['revision']:
            raise SystemExit(f'state.json belongs to revision {self.state.get("revision")}, the plan is {p["revision"]}')
        if self.state.get('layout', 'a4k') != self.layout:
            raise SystemExit(f'state.json is a {self.state.get("layout", "a4k")} bank, the plan is {self.layout}')

    def bank_layer(self, L):
        return L - self.first

    def is_bank(self, L):
        return self.first <= L < self.mtp

    def verified(self, name):
        """The shard is complete and its sha256 is the branch's: the download lane's marker, else hashed here."""
        meta = self.plan['shards'][name]
        path = os.path.join(self.src, name)
        if not os.path.exists(path) or os.path.getsize(path) != meta['size']:
            return None
        mk = path + '.sha256ok'
        if not os.path.exists(mk):
            return None
        words = open(mk).read().split()
        return 'marker' if words and words[0] == meta['sha256'] else None

    def eligible(self):
        out = []
        for name in sorted(self.plan['shards']):
            st = self.state['shards'].get(name)
            if st is not None:
                continue
            if self.verified(name):
                out.append(name)
        return out

    def convert_shard(self, name):
        p = self.plan
        meta = p['shards'][name]
        path = os.path.join(self.src, name)
        t0 = time.time()
        rc = {'shard': name, 'repo': p['repo'], 'revision': p['revision'], 'started': time.strftime('%F %T'),
              'converter': 'ling_exl3_to_streaming.py', 'checks': {}, 'records': [], 'pass': False}
        how = self.verified(name)
        if how is None:
            digest = sha256_file(path)
            how = 'hashed' if digest == meta['sha256'] else None
        rc['checks']['sha256'] = how is not None
        rc['sha256_from'] = how
        if not how:
            rc['error'] = 'the shard is not the branch file (size or sha256)'
            return self._finish(name, rc)
        fd = os.open(path, os.O_RDONLY)
        nocache(fd)
        try:
            hdr, data0, _ = read_header(fd)
            groups, nonexpert = {}, []
            for tname, v in hdr.items():
                m = EXPERT_RE.match(tname)
                lm = LAYER_RE.match(tname)
                if m and self.is_bank(int(m.group(1))):
                    groups.setdefault((int(m.group(1)), int(m.group(2))), {})[(m.group(3), m.group(4))] = v
                elif lm and int(lm.group(1)) == self.mtp:
                    pass  # the MTP layer: mtp-nonbank.safetensors at finalize
                else:
                    nonexpert.append(tname)
            problems = self._verify_header(fd, data0, groups)
            rc['checks']['header'] = not problems
            rc['records_in_shard'] = len(groups)
            if problems:
                rc['error'] = problems[:20]
                return self._finish(name, rc)
            ofd = os.open(os.path.join(self.out, 'experts.bin'), os.O_RDWR | os.O_CREAT, 0o644)
            nocache(ofd)
            try:
                if os.fstat(ofd).st_size < self.total:
                    os.ftruncate(ofd, self.total)
                t1 = time.time()
                keys = sorted(groups)
                with cf.ThreadPoolExecutor(max_workers=self.args.threads) as ex:
                    rc['records'] = list(ex.map(lambda k: self._write_record(fd, data0, groups[k], k[0], k[1], ofd), keys))
                full_fsync(ofd)
                rc['timings'] = {'write_s': round(time.time() - t1, 2)}
                t2 = time.time()
                with cf.ThreadPoolExecutor(max_workers=self.args.threads) as ex:
                    list(ex.map(lambda r: self._check_record(fd, data0, groups[(r['model_layer'], r['expert'])], r, ofd),
                                rc['records']))
                rc['timings']['readback_s'] = round(time.time() - t2, 2)
            finally:
                os.close(ofd)
            t3 = time.time()
            rc['nonexpert'] = self._write_tensors(fd, data0, hdr, nonexpert,
                                                  os.path.join(self.out, name.replace('.safetensors', '-nonexpert.safetensors')))
            rc['timings']['nonexpert_s'] = round(time.time() - t3, 2)
        finally:
            os.close(fd)
        rc['checks']['records'] = all(r['pass'] for r in rc['records'])
        rc['checks']['nonexpert'] = rc['nonexpert'] is None or rc['nonexpert']['readback']
        rc['pass'] = all(rc['checks'].values())
        rc['timings']['total_s'] = round(time.time() - t0, 2)
        return self._finish(name, rc)

    def _verify_header(self, fd, data0, groups):
        probs = []
        for (L, e), g in sorted(groups.items()):
            t = self.table[self.bank_layer(L)]
            segs = {s['component']: s for s in t['segments']}
            if any(leaf == 'mcg' for _, leaf in g):
                probs.append(f'({L},{e}) has an mcg tensor')
            for proj in PROJ:
                for comp, dt in (('code', 'I16'), ('rout', 'F16'), ('rin', 'F16')):
                    v = g.get((proj, LEAF[comp]))
                    s = segs[f'{proj}.{comp}']
                    if v is None or v['dtype'] != dt or list(v['shape']) != s['shape'] or \
                            v['data_offsets'][1] - v['data_offsets'][0] != s['length']:
                        probs.append(f'({L},{e}) {proj}.{LEAF[comp]} {None if v is None else (v["dtype"], v["shape"])} '
                                     f'!= {dt} {s["shape"]} (plan K={t["K"]})')
                mv = g.get((proj, 'mul1'))
                if mv is None or mv['dtype'] != 'I32' or mv['data_offsets'][1] - mv['data_offsets'][0] != 4:
                    probs.append(f'({L},{e}) {proj}.mul1 missing or not one I32')
                else:
                    val = struct.unpack('<I', os.pread(fd, 4, data0 + mv['data_offsets'][0]))[0]
                    if val != MUL1_MULT:
                        probs.append(f'({L},{e}) {proj}.mul1 = {val:#x}')
            if len(g) != 12:
                probs.append(f'({L},{e}) has {len(g)} tensors, not 12')
        return probs

    def _span(self, g):
        """The record's 12 source tensors as one span [a, b) (they are contiguous in turboderp's shards), or None."""
        offs = sorted(v['data_offsets'] for v in g.values())
        if all(offs[i][1] == offs[i + 1][0] for i in range(len(offs) - 1)):
            return offs[0][0], offs[-1][1]
        return None

    def _pieces(self, fd, data0, g):
        """Each component's source bytes: one read of the record's span when its tensors are contiguous, else one a part."""
        sp = self._span(g)
        mv, a0 = (memoryview(pread_all(fd, sp[1] - sp[0], data0 + sp[0])), sp[0]) if sp is not None else (None, 0)
        out = {}
        for c in COMPONENTS:
            proj, comp = c.split('.')
            a, b = g[(proj, LEAF[comp])]['data_offsets']
            out[c] = mv[a - a0:b - a0] if mv is not None else memoryview(pread_all(fd, b - a, data0 + a))
        return out

    def _write_record(self, fd, data0, g, L, e, ofd):
        b = self.bank_layer(L)
        t = self.table[b]
        pieces = self._pieces(fd, data0, g)
        rec = bytearray(t['record_bytes'])
        src_sha, logical = {}, hashlib.sha256()
        for s in t['segments']:
            piece = pieces[s['component']]
            assert len(piece) == s['length']
            rec[s['offset']:s['offset'] + s['length']] = piece
            src_sha[s['component']] = hashlib.sha256(piece).hexdigest()
            logical.update(piece)
        off = t['base_offset'] + e * t['record_bytes']
        if os.pwrite(ofd, rec, off) != len(rec):
            raise IOError(f'short write ({L},{e})')
        seg = {s['component']: s for s in t['segments']}
        stats = {'max_abs_rin': [], 'rout_zeros': [], 'rout_min': [], 'rout_max': []}
        for proj in PROJ:
            ri = f16s(rec, seg[f'{proj}.rin'])
            ro = f16s(rec, seg[f'{proj}.rout'])
            stats['max_abs_rin'].append(max(abs(v) for v in ri))
            stats['rout_zeros'].append(sum(1 for v in ro if v == 0))
            stats['rout_min'].append(min(ro))
            stats['rout_max'].append(max(ro))
        return {'layer': b, 'model_layer': L, 'expert': e, 'K': t['K'], 'sidecar_offset': off,
                'record_bytes': t['record_bytes'], 'logical_bytes': t['logical_bytes'],
                'sha256': hashlib.sha256(rec).hexdigest(), 'sha256_logical': logical.hexdigest(),
                'segment_source_sha256': src_sha, 'stats': stats, 'pass': False}

    def _check_record(self, fd, data0, g, r, ofd):
        """The written record read back: every part byte-equal to the source span read again, the lead and the tail
        zero, the padded sha256 the written one."""
        t = self.table[r['layer']]
        rec = pread_all(ofd, t['record_bytes'], r['sidecar_offset'])
        pieces = self._pieces(fd, data0, g)
        ok_bytes = all(bytes(rec[s['offset']:s['offset'] + s['length']]) == bytes(pieces[s['component']])
                       for s in t['segments'])
        ok_pad = not any(rec[:t['_lead']]) and not any(rec[t['_extent']:])
        ok_sha = hashlib.sha256(rec).hexdigest() == r['sha256']
        r.update({'bytes_identical': ok_bytes, 'pad_zero': ok_pad, 'sha256_readback': ok_sha})
        r['pass'] = ok_bytes and ok_pad and ok_sha

    def _write_tensors(self, fd, data0, hdr, names, path):
        """`names` of the source shard into a safetensors file at `path`, byte for byte, then read back."""
        if not names:
            return None
        names = sorted(names, key=lambda n: hdr[n]['data_offsets'][0])
        new, off = {}, 0
        for n in names:
            a, b = hdr[n]['data_offsets']
            new[n] = {'dtype': hdr[n]['dtype'], 'shape': hdr[n]['shape'], 'data_offsets': [off, off + b - a]}
            off += b - a
        hb = json.dumps(new, separators=(',', ':')).encode()
        hb += b' ' * ((8 - len(hb) % 8) % 8)
        tmp = path + '.tmp'
        src_sha = {}
        with open(tmp, 'wb') as f:
            nocache(f.fileno())
            f.write(struct.pack('<Q', len(hb)) + hb)
            for n in names:
                a, b = hdr[n]['data_offsets']
                h = hashlib.sha256()
                pos = a
                while pos < b:
                    buf = os.pread(fd, min(CHUNK, b - pos), data0 + pos)
                    if not buf:
                        raise IOError('short read')
                    f.write(buf)
                    h.update(buf)
                    pos += len(buf)
                src_sha[n] = h.hexdigest()
            f.flush()
            full_fsync(f.fileno())
        # read back
        rfd = os.open(tmp, os.O_RDONLY)
        nocache(rfd)
        try:
            rh, r0, _ = read_header(rfd)
            ok = set(rh) == set(new)
            for n in names:
                if not ok:
                    break
                a, b = rh[n]['data_offsets']
                ok = rh[n]['dtype'] == new[n]['dtype'] and rh[n]['shape'] == new[n]['shape'] and \
                    hashlib.sha256(pread_all(rfd, b - a, r0 + a)).hexdigest() == src_sha[n]
            ok = ok and os.fstat(rfd).st_size == r0 + off
        finally:
            os.close(rfd)
        os.replace(tmp, path)
        return {'file': os.path.basename(path), 'tensors': len(names), 'data_bytes': off, 'readback': ok,
                'names': names}

    def _finish(self, name, rc):
        rc['finished'] = time.strftime('%F %T')
        rpath = os.path.join(self.out, 'receipts', name.replace('.safetensors', '.json'))
        atomic_json(rpath, rc)
        self.state['shards'][name] = {'pass': bool(rc['pass']), 'receipt': os.path.relpath(rpath, self.out),
                                      'finished': rc['finished']}
        atomic_json(self.state_path, self.state)
        n_bad = sum(not r.get('pass') for r in rc['records'])
        log(f'{name}: pass={rc["pass"]} records={len(rc["records"])} bad={n_bad} sha256={rc.get("sha256_from")} '
            f'nonexpert={(rc.get("nonexpert") or {}).get("tensors")} timings={rc.get("timings")} err={rc.get("error")}')
        return rc

    # -- the manifests, the MTP shard, the index, the links
    def finalize(self):
        p = self.plan
        recs, nonexpert_files, mtp_src = {}, [], []
        for name in sorted(p['shards']):
            st = self.state['shards'].get(name)
            if not (st and st.get('pass')):
                raise SystemExit(f'finalize: {name} has no passing receipt')
            rc = json.load(open(os.path.join(self.out, st['receipt'])))
            for r in rc['records']:
                recs[(r['layer'], r['expert'])] = r
            if rc.get('nonexpert'):
                nonexpert_files.append(rc['nonexpert'])
        need = len(self.table) * p['n_experts']
        if len(recs) != need:
            raise SystemExit(f'finalize: {len(recs)} of {need} records have passing receipts')
        side = os.path.join(self.out, 'experts.bin')
        if os.path.getsize(side) != self.total:
            raise SystemExit(f'finalize: experts.bin is {os.path.getsize(side)} B, the table says {self.total}')
        # the MTP layer's tensors, from every source shard holding them
        mtp_names = []
        weight_map = {}
        total_size = 0
        for name in sorted(p['shards']):
            fd = os.open(os.path.join(self.src, name), os.O_RDONLY)
            try:
                hdr, _, _ = read_header(fd)
            finally:
                os.close(fd)
            ns = [n for n in hdr if LAYER_RE.match(n) and int(LAYER_RE.match(n).group(1)) == self.mtp]
            if ns:
                mtp_src.append((name, ns))
                mtp_names += ns
        mtp_info = self._write_mtp(mtp_src)
        for nf in nonexpert_files:
            for n in nf['names']:
                weight_map[n] = nf['file']
            total_size += nf['data_bytes']
        for n in mtp_names:
            weight_map[n] = 'mtp-nonbank.safetensors'
        total_size += mtp_info['data_bytes']
        index = {'metadata': {'total_size': total_size}, 'weight_map': dict(sorted(weight_map.items()))}
        # every source tensor that is not a bank expert must be named, once
        src_index = json.load(open(os.path.join(self.src, 'model.safetensors.index.json')))['weight_map']
        want = {n for n in src_index if not (EXPERT_RE.match(n) and self.is_bank(int(EXPERT_RE.match(n).group(1))))}
        if want != set(weight_map):
            raise SystemExit(f'finalize: index differs from the source index: missing {sorted(want - set(weight_map))[:5]}, '
                             f'extra {sorted(set(weight_map) - want)[:5]}')
        records = [recs[k] for k in sorted(recs)]
        v2 = {
            'format': FORMAT_V2,
            'model_key': p['model_key'],
            'quantization': {
                'mode': 'exl3', 'codebook': 'mul1', 'codebook_multiplier': MUL1_MULT,
                'tile': {'size': 16, 'layout': 'exl3-tensor-core', 'bitstream': 'msb-first-swap16'},
                'hadamard': {'size': 128, 'order': 'sylvester-natural', 'scale': '1/sqrt(128)'},
                'scales': {'rin': 'exl3.suh', 'rout': 'exl3.svh'},
                'source': {'repo': p['repo'], 'revision': p['revision'],
                           'exllamav3': f'{p["exllamav3"]} (the branch quantization_config version)'},
            },
            'dims': {'hidden': p['hidden'], 'inter': p['inter'], 'n_experts': p['n_experts'], 'n_layers': len(self.table)},
            'sidecar': {'file': 'experts.bin', 'alignment': ALIGN, 'size': self.total},
            'layers': [public(t) for t in self.table],
            'records': [{k: r[k] for k in ('layer', 'expert', 'sidecar_offset', 'record_bytes', 'logical_bytes', 'sha256',
                                           'stats')} for r in records],
            'artifact': {'record_count': len(records), 'routed_expert_bytes': self.total},
            'parity': {'receipts': 'receipts', 'all_pass': all(r['pass'] for r in records)},
        }
        if self.layout == 'a4k':
            v2['sidecar']['layout'] = 'a4k'
            v2['relayout'] = {'layout': 'a4k', 'from': f'{p["repo"]}@{p["branch"]} ({p["revision"]}), written a4k directly',
                              'tool': 'ling_exl3_to_streaming.py', 'disk_order': [COMPONENTS[c] for c in DISK],
                              'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        v1recs = []
        for r in records:
            t = self.table[r['layer']]
            off = r['sidecar_offset']
            v1recs.append({'layer': r['layer'], 'expert': r['expert'], 'logical_bytes': t['logical_bytes'],
                           'segments': [{'component': s['component'], 'dtype': s['dtype'], 'shape': s['shape'],
                                         'offset': off + s['offset'], 'length': s['length'], 'shard': 'experts.bin'}
                                        for s in t['segments']],
                           'sha256': r['sha256_logical'], 'sidecar_offset': off, 'sidecar_length': t['logical_bytes']})
        v1 = {'format': FORMAT_V1, 'model_key': p['model_key'],
              'quantization': {'mode': 'exl3-mul1', 'codebook': 'mul1', 'codebook_multiplier': MUL1_MULT,
                               'source': v2['quantization']['source']},
              'sidecar': {'file': 'experts.bin', 'alignment': ALIGN, 'size': self.total}, 'records': v1recs}
        probs = check_v2(v2, os.path.getsize(side))
        if probs:
            raise SystemExit('finalize: v2 check: ' + '; '.join(probs[:10]))
        atomic_json(os.path.join(self.out, 'model.safetensors.index.json'), index)
        atomic_json(os.path.join(self.out, 'expert-manifest-v2.json'), v2, compact=True)
        atomic_json(os.path.join(self.out, 'expert-manifest.json'), v1, compact=True)
        linked = []
        for f in SMALL_FILES:
            sp, dp = os.path.join(self.src, f), os.path.join(self.out, f)
            if os.path.exists(sp) and not os.path.exists(dp):
                os.link(sp, dp)
                linked.append(f)
        receipt = {'what': f'{p["repo"]}@{p["branch"]} ({p["revision"]}) as a streaming EXL3 pack for turboexl3_ling: '
                           f'the routed experts of model layers {self.first}..{self.mtp - 1} in experts.bin (a4k, bank '
                           f'layer = model layer - {self.first}), the rest byte for byte in the -nonexpert shards and '
                           f'mtp-nonbank.safetensors (layer {self.mtp}, the MTP layer)',
                   'source': {'repo': p['repo'], 'branch': p['branch'], 'revision': p['revision']},
                   'first_moe_layer': self.first, 'k_by_layer': p['k_by_layer'],
                   'layers': [{'layer': t['layer'], 'K': t['K'], 'record_bytes': t['record_bytes'], 'lead': t['_lead'],
                               'extent': t['_extent'], 'logical': t['logical_bytes']} for t in self.table],
                   'layout': self.layout,
                   'disk_order': [COMPONENTS[c] for c in (DISK if self.layout == 'a4k' else range(9))],
                   'experts_bin': self.total,
                   'records_verified': sum(r['pass'] for r in records), 'nonexpert_files': len(nonexpert_files),
                   'mtp': {k: v for k, v in mtp_info.items() if k != 'names'}, 'linked_files': linked,
                   'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        atomic_json(os.path.join(self.out, 'LING-REPACK-RECEIPT.json'), receipt)
        log(f'FINALIZED {self.out}: {len(records)} records, experts.bin {self.total} B, {len(nonexpert_files)} '
            f'non-expert files, mtp {mtp_info["tensors"]} tensors, {len(linked)} links')

    def _write_mtp(self, mtp_src):
        """mtp-nonbank.safetensors: every MTP-layer tensor of the source shards, byte for byte, read back."""
        path = os.path.join(self.out, 'mtp-nonbank.safetensors')
        items = []
        for name, ns in mtp_src:
            fd = os.open(os.path.join(self.src, name), os.O_RDONLY)
            hdr, d0, _ = read_header(fd)
            os.close(fd)
            for n in sorted(ns, key=lambda n: hdr[n]['data_offsets'][0]):
                items.append((name, d0, n, hdr[n]))
        new, off = {}, 0
        for _, _, n, v in items:
            a, b = v['data_offsets']
            new[n] = {'dtype': v['dtype'], 'shape': v['shape'], 'data_offsets': [off, off + b - a]}
            off += b - a
        hb = json.dumps(new, separators=(',', ':')).encode()
        hb += b' ' * ((8 - len(hb) % 8) % 8)
        tmp = path + '.tmp'
        sha = {}
        fds = {name: os.open(os.path.join(self.src, name), os.O_RDONLY) for name, _ in mtp_src}
        try:
            with open(tmp, 'wb') as f:
                nocache(f.fileno())
                f.write(struct.pack('<Q', len(hb)) + hb)
                for name, d0, n, v in items:
                    a, b = v['data_offsets']
                    buf = pread_all(fds[name], b - a, d0 + a)
                    f.write(buf)
                    sha[n] = hashlib.sha256(buf).hexdigest()
                f.flush()
                full_fsync(f.fileno())
        finally:
            for fd in fds.values():
                os.close(fd)
        rfd = os.open(tmp, os.O_RDONLY)
        try:
            rh, r0, _ = read_header(rfd)
            ok = set(rh) == set(new) and all(
                hashlib.sha256(pread_all(rfd, rh[n]['data_offsets'][1] - rh[n]['data_offsets'][0],
                                         r0 + rh[n]['data_offsets'][0])).hexdigest() == sha[n] for n in new)
        finally:
            os.close(rfd)
        if not ok:
            raise SystemExit('finalize: mtp-nonbank.safetensors read-back differs')
        os.replace(tmp, path)
        return {'file': 'mtp-nonbank.safetensors', 'tensors': len(new), 'data_bytes': off, 'readback': ok}


def check_v2(m, sidecar_size):
    """bankv2.check_manifest's format checks at the manifest's layout (classic unless sidecar.layout says a4k) and a
    fractional K."""
    p = []
    d = m['dims']
    base = 0
    if len(m['layers']) != d['n_layers']:
        p.append('layer table length != dims.n_layers')
    for b, t in enumerate(m['layers']):
        layout = m['sidecar'].get('layout', 'classic')
        g = record_layout(t['K'], d['hidden'], d['inter'], layout)
        if t['layer'] != b or t['segments'] != g['segments'] or t['record_bytes'] != g['record_bytes'] or \
                t['logical_bytes'] != g['logical_bytes'] or t['base_offset'] != base:
            p.append(f'layer {b}: not the {layout} geometry at K={t["K"]}')
        base += d['n_experts'] * t['record_bytes']
    if m['sidecar']['size'] != base or sidecar_size != base:
        p.append(f'sidecar size {m["sidecar"]["size"]} / file {sidecar_size} != {base}')
    seen = set()
    for r in m['records']:
        t = m['layers'][r['layer']]
        if r['sidecar_offset'] != t['base_offset'] + r['expert'] * t['record_bytes'] or r['sidecar_offset'] % ALIGN \
                or r['record_bytes'] != t['record_bytes'] or len(r['sha256']) != 64:
            p.append(f'record ({r["layer"]},{r["expert"]}) offset/size/sha')
        seen.add((r['layer'], r['expert']))
    if len(seen) != len(m['layers']) * d['n_experts'] or len(m['records']) != len(seen):
        p.append('records do not cover every (layer, expert) once')
    if not m['parity']['all_pass']:
        p.append('parity.all_pass is not true')
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('plan', 'run', 'finalize'))
    ap.add_argument('--src', required=True)
    ap.add_argument('--out')
    ap.add_argument('--plan', required=True)
    ap.add_argument('--rev')
    ap.add_argument('--layout', default='classic', choices=('classic', 'a4k'))
    ap.add_argument('--only')
    ap.add_argument('--watch', type=int, default=0, help='poll interval seconds; 0 = one pass')
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    a.src = os.path.expanduser(a.src)
    if a.cmd == 'plan':
        return make_plan(a)
    a.out = os.path.expanduser(a.out)
    c = Converter(a)
    if a.cmd == 'finalize':
        return c.finalize()
    while True:
        todo = [a.only] if a.only else c.eligible()
        for name in todo:
            log(f'converting {name}')
            c.convert_shard(name)
        if a.only:
            return
        done = sum(1 for s in c.state['shards'].values() if s.get('pass'))
        failed = [n for n, s in c.state['shards'].items() if not s.get('pass')]
        if failed:
            log(f'STOP: failed shards {failed}')
            raise SystemExit(3)
        if done == len(c.plan['shards']):
            log(f'ALL_SHARDS_PASS {done}/{len(c.plan["shards"])}')
            return
        if not a.watch:
            log(f'{done}/{len(c.plan["shards"])} shards pass')
            return
        time.sleep(a.watch)


if __name__ == '__main__':
    main()
