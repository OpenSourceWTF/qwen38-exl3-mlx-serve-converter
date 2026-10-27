# Qwen3.8-Flash-Next EXL3 to mlx-serve converter

These scripts convert turboderp's EXL3 quantizations of Qwen3.8-Flash-Next
([turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)) into the pack layout
that [mlx-serve](https://github.com/ddalcu/mlx-serve) loads with the [turboexl3](https://github.com/davidtai/turboexl3)
quant plugin. They built these four packs:

- [OpensourceWTF/Qwen3.8-Flash-Next-EXL3-2.05-mlx-serve](https://huggingface.co/OpensourceWTF/Qwen3.8-Flash-Next-EXL3-2.05-mlx-serve)
- [OpensourceWTF/Qwen3.8-Flash-Next-EXL3-3.05-mlx-serve](https://huggingface.co/OpensourceWTF/Qwen3.8-Flash-Next-EXL3-3.05-mlx-serve)
- [OpensourceWTF/Qwen3.8-Flash-Next-EXL3-4.05-mlx-serve](https://huggingface.co/OpensourceWTF/Qwen3.8-Flash-Next-EXL3-4.05-mlx-serve)
- [OpensourceWTF/Qwen3.8-Flash-Next-EXL3-5.05-mlx-serve](https://huggingface.co/OpensourceWTF/Qwen3.8-Flash-Next-EXL3-5.05-mlx-serve)

The scripts copy bytes. They do not quantize. Only three things change form: unquantized f16 tensors become bf16 (round
to nearest even), the shape label of the `conv1d` weights changes, and the header of a sharded n-gram table is rewritten.
They run on the CPU. They need no GPU and load no model.

## Files

| file | what it does |
|---|---|
| `scripts/restack_exl3_q2_experts.py` | Restacks the routed experts into one shard per layer. Detects K per layer. |
| `scripts/restack_exl3_q2_full.py` | Writes the trunk shards, the carry shard, the n-gram table, `config.json` and the index. |
| `scripts/verify_raw_shard.py` | Optional. Checks one of turboderp's files against its HF sha256 and against the converted pack, byte for byte. |
| `scripts/hash_evict.py` | Optional. Computes the sha256 of a large file without filling the OS file cache. |
| `scripts/hash_nocache.py` | Optional. The same with uncached reads (used for the 3.05 table). |
| `history/` | The exact scripts that built the 3.05, 4.05 and 5.05 packs (see [Provenance](#provenance)), including `rebuild_505_mixer.py`, the last step of the 5.05 pack (see [The 5.05 MTP mixer](#the-505-mtp-mixer)). |

The names keep the `q2` suffix because these copies built the 2.05 pack. They are the most general versions.

## What the conversion does

- **Routed experts.** For each layer L (the 48 trunk layers and the MTP layer) and each projection P (gate, up, down),
  the 512 per-expert tensors `<prefix>.layers.L.mlp.experts.E.P.{trellis,suh,svh}` become three stacked tensors:
  `language_model.model.layers.L.mlp.switch_mlp.P.trellis` U16 [512, in/16, out/16, 16K], `.suh` F16 [512, in] and
  `.svh` F16 [512, out] (`language_model.mtp.` for the MTP layer). Every expert's `mul1` value must be 0x83DCD12D. Each
  layer goes to its own file (`experts-00.safetensors` to `experts-47.safetensors`, `experts-mtp-00.safetensors`), with
  the stamp `{format: exl3, k: K, codebook: mul1, window: 16}`.
- **Trunk.** Each EXL3 linear `<module>.{trellis,suh,svh,mul1}` becomes `<host name>.weight` (the trellis, U16),
  `.scales` (suh, F16) and `.biases` (svh, F16). The `mul1` value is checked and dropped. The names change from
  `model.language_model.` to `language_model.model.`, from `mtp.` to `language_model.mtp.` and from `lm_head` to
  `language_model.lm_head`. Unquantized f16 tensors become bf16, except the n-gram `head_bias`, which stays f16. The
  `conv1d` weights are relabelled from [C, 1, K] to [C, K, 1]; the bytes are the same. The vision tower
  (`model.visual.*`) is skipped. The tensors go to `trunk-exl3-00.safetensors` to `trunk-exl3-10.safetensors`, grouped
  as in the reference index.
- **Carry shard.** `trunk-exl3-carry.safetensors` holds the MTP hyper-connection mixer and four small tensors from the
  header of the n-gram file: `head_bias`, the head offsets, the head vocabulary sizes and the layer multipliers (renamed
  for mlx-serve).
- **n-gram table.** A table stored as one tensor (the 4.05 and 5.05 branches) is hard-linked. A table stored as 128 row
  shards (the 2.05 and 3.05 branches) is cloned with APFS `clonefile`, which copies no data. Then only its header is
  rewritten, to name the same bytes as one tensor. The new header has the same length, so every data offset stays the
  same.
- **config.json.** The base config, plus `expert_quant`, `trunk_quant`, `ngram_table` (its K and row width come from
  the table's metadata), `qwen4_norm_convention: "delta"` and some notes.
- **Other files.** The vision shard and the small files (tokenizer, chat template, `generation_config.json`, `LICENSE`)
  are hard-linked from the base folder.

## Inputs

1. **One of turboderp's branches**, downloaded:

   ```sh
   hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 2.05bpw_h4_ng4 --local-dir src
   ```

2. **A base folder** with the vision shard and the small files of one of our packs. Use the 4.05 or the 5.05 pack. The
   `config.json` of the 2.05 and 3.05 packs has an `ngram_table_note` about their rewritten table, and the script keeps
   that note when it hard-links a one-tensor table, so a 4.05 or 5.05 conversion from those bases gets a wrong note. Do
   not download the base pack's `README.md`, because the script links every `README.md` that it finds in the base folder.

   ```sh
   hf download OpensourceWTF/Qwen3.8-Flash-Next-EXL3-4.05-mlx-serve \
     config.json model.safetensors.index.json generation_config.json tokenizer.json tokenizer_config.json \
     chat_template.jinja vocab.json merges.txt LICENSE trunk-model-vision.safetensors --local-dir base
   ```

   Pass this folder as both `--hybrid` (the vision shard, the base config and the small files) and `--ref` (the shard
   layout, read from its index). Our own builds passed an earlier, unpublished pack of ours as `--hybrid` and our first
   5.05 pack (the one before its mixer was rebuilt) as `--ref`. Those hold the same vision shard and small files, but we
   did not run the scripts with a downloaded base folder.

`src`, the base folder and the output folder must be on the same APFS volume, because the scripts make hard links and
clones.

## Output

A folder that mlx-serve loads: 49 expert shards, `trunk-exl3-00.safetensors` to `trunk-exl3-10.safetensors`,
`trunk-exl3-carry.safetensors`, `ngram_embedding.safetensors`, `trunk-model-vision.safetensors`, `config.json`,
`model.safetensors.index.json`, `generation_config.json`, the tokenizer files, `chat_template.jinja` and `LICENSE`:
2,520 tensors in 62 safetensors files.

## Run

```sh
B=2.05bpw_h4_ng4; K=2     # see the table below
PY=python3                # a Python 3 that has numpy

$PY -I scripts/restack_exl3_q2_experts.py --src src --out pack --check     # dry run: checks every expert header
$PY -I scripts/restack_exl3_q2_experts.py --src src --out pack

$PY -I scripts/restack_exl3_q2_full.py --expert-k $K --source-tag turboderp/Qwen3.8-Flash-Next-exl3@$B \
    --src src --hybrid base --ref base --out pack --phase all --check      # dry run
$PY -I scripts/restack_exl3_q2_full.py --expert-k $K --source-tag turboderp/Qwen3.8-Flash-Next-exl3@$B \
    --src src --hybrid base --ref base --out pack --phase all
```

Run the expert restack first: the last phase of `restack_exl3_q2_full.py` stops if an expert shard is missing.

| branch | `--expert-k` | `--extra-src` | n-gram table | MTP mixer comes from |
|---|---|---|---|---|
| `2.05bpw_h4_ng4` | 2 | default | 128 row shards: clone and new header | `mtp_hyper_connection_mixer_patch.safetensors` |
| `3.05bpw_h5_ng5` | 3 | default | 128 row shards: clone and new header | `mtp_hyper_connection_mixer_patch.safetensors` |
| `4.05bpw_h6_ng6` | 4 | `""` | one tensor: hard link | turboderp's model shards |
| `5.05bpw_h6_ng6` | 5 | default | one tensor: hard link | `mtp_hyper_connection_mixer_patch.safetensors` |

The 4.05 branch has its mixer in the model shards and no patch file, so pass `--extra-src ""` there.

On the 5.05 branch these scripts make a pack with turboderp's MTP mixer, as in our published 5.05 pack. That pack was
made another way (see [The 5.05 MTP mixer](#the-505-mtp-mixer)): its carry shard holds the same tensors, but the note in
the shard's header differs, so the file's sha256 differs too. We did not compare the other files.

Only the 2.05 pack was built with these two scripts. The 3.05, 4.05 and 5.05 packs were built with earlier copies (see
[Provenance](#provenance)), so running these two scripts on those branches is untested.

### Less disk

The 2.05 build converted one of turboderp's files at a time and deleted each one after it was checked:

1. `restack_exl3_q2_full.py ... --phase trunk` while every model shard is present.
2. For each model shard F: `restack_exl3_q2_experts.py --src src --out pack --raw-shard F --skip-existing`, then
   `verify_raw_shard.py --src src --out pack --shard F --sha256 <its LFS sha256>`, then delete F.
3. Check the mixer patch the same way.
4. Download `ngram_embedding.safetensors` last and check it: `hash_evict.py src/ngram_embedding.safetensors <sha256>`.
5. `restack_exl3_q2_full.py ... --phase finalize`.

The LFS sha256 of each file is in `https://huggingface.co/api/models/turboderp/Qwen3.8-Flash-Next-exl3/tree/<branch>`.

## The 5.05 MTP mixer

Our first 5.05 pack held ddalcu's 8-bit affine copy of the MTP hyper-connection mixer. Its download fetched only the files
that turboderp's shard index names, and his `5.05bpw_h6_ng6` branch ships his own mixer as the unindexed file
`mtp_hyper_connection_mixer_patch.safetensors` (13,128,048 bytes, commit `7cef615f`). `history/rebuild_505_mixer.py` made
the published 5.05 pack from that first pack, in a new folder:

- `trunk-exl3-carry.safetensors`: the first pack's 4 small n-gram tensors, byte for byte, and turboderp's 3 mixer tensors
  (`hc_norm` [10240], `input_mix_weight_down` [320, 10240], `input_mix_weight_up` [10240, 320]). The mixer tensors change
  from f16 to bf16 (round to nearest even), as `history/restack_exl3_q4_full.py` converts them; `hc_norm` keeps its
  zero-centered form.
- `model.safetensors.index.json`: the first pack's index without the 4 `.scales` and `.biases` entries of the 8-bit mixer
  (2,524 tensors become 2,520).
- `config.json`: the first pack's config with `"mtp_mixer_from": "turboderp"` added after `vision_from`.
- Every other file: a hard link to the first pack's file. The script writes no file through a link.
- `CONVERSION.md`: a record of the run (we did not publish ours).

The script checks the sha256 of the patch (`9b29ae89413854802076ad392809abe4d91a1e61a6bc08193dad8271a7510953`) and its
tensor names, dtypes and shapes. It reads the new carry shard back, and it stops if the n-gram tensors changed. It also
compares each tensor of the new carry shard with the carry shard of a reference pack (`--ref`) and prints the result. In
our run all 7 tensors were byte-identical to our 4.05 pack's, whose mixer came from turboderp's 4.05 branch: the 2.05,
3.05, 4.05 and 5.05 packs carry the same mixer bytes. The new carry shard has 13,134,304 bytes, sha256
`47023869eab0858d41c8e166952fe9effe718a4bf91ede39fe3e572ac027ceee`.

Run it. The inputs other than `--src` download from Hugging Face:

```sh
PY=python3                # a Python 3 that has numpy
hf download turboderp/Qwen3.8-Flash-Next-exl3 mtp_hyper_connection_mixer_patch.safetensors \
  --revision 7cef615f7fd3681295b68848018876dfabc336c7 --local-dir patch
hf download OpensourceWTF/Qwen3.8-Flash-Next-EXL3-4.05-mlx-serve trunk-exl3-carry.safetensors --local-dir ref

$PY -I history/rebuild_505_mixer.py --src pack505 --patch patch/mtp_hyper_connection_mixer_patch.safetensors \
    --ref ref --out pack505-tdmixer       # prints MIXER_REBUILD_OK at the end
```

`--src` is our first 5.05 pack (with the 8-bit mixer). We did not publish it; [Provenance](#provenance) says how it was
built. `--ref` needs only `trunk-exl3-carry.safetensors`. `--out` must not exist, and it must be on the same APFS volume as
`--src`, because the script makes hard links.

To check the published 5.05 carry shard with downloaded files only, give the script the published pack's carry shard,
index and config as `--src`:

```sh
hf download OpensourceWTF/Qwen3.8-Flash-Next-EXL3-5.05-mlx-serve trunk-exl3-carry.safetensors \
  model.safetensors.index.json config.json --local-dir pub505
$PY -I history/rebuild_505_mixer.py --src pub505 --patch patch/mtp_hyper_connection_mixer_patch.safetensors \
    --ref ref --out check505
shasum -a 256 check505/trunk-exl3-carry.safetensors pub505/trunk-exl3-carry.safetensors
```

Here the script stops at the end with `SRC config already has mtp_mixer_from=turboderp`, because the published config
already has that key. It writes the carry shard and the index before that check, so `check505/` holds them, and the two
sha256 values must be the same (`47023869eab0858d…`). We did not run this check.

The run used a revision of `history/rebuild_505_mixer.py` that differs from this copy only in two text strings: a quote in
its docstring, and the same quote in the first sentence of the `CONVERSION.md` record that it writes. The conversion and
its checks are the same. sha256 of the revision that ran: `bfebd80537bcba9d368cf98024aa57682c57194cc36c9b8941b09491060ec462`; of this copy: `f64324d87d50fbe6a1a81d4c6d89e7a25d91c8f0b7d8f264bf29820cf0a80416`.

## Requirements

- macOS on an APFS volume. The scripts call `clonefile`, `vm_stat`, `sysctl vm.swapusage` and `F_NOCACHE`. They do not
  run on Linux as written.
- Python 3 with numpy. We used Python 3.12.13 and numpy 2.5.2. `history/restack_exl3_qwen38.py` also needs
  `safetensors`.
- The Hugging Face CLI (`hf`) for the downloads. We used huggingface_hub 1.24.0.
- Disk: the branch download plus the new expert and trunk shards. The table is linked or cloned, not copied. For the
  5.05 branch that is about 123 GB plus 84 GB.
- Memory: a few GB. The expert restack streams one expert at a time, and it stops if free plus inactive memory drops
  below `--mem-floor-gib` (default 12) or swap use passes `--swap-max-mib` (default 1500). The trunk phase holds one
  tensor at a time; the largest, the token embedding, is 1.3 GB.

## Provenance

These are the exact files that built the published packs (sha256):

| pack | scripts |
|---|---|
| 2.05 | `scripts/restack_exl3_q2_experts.py` (`7ab0cf5602c39b3f893574aa98d38fad99596bf9caba2d865ddf5a828edbdcfd`), `scripts/restack_exl3_q2_full.py` (`5454899b4e2a2f8a4899f1ad05ff5c6ab72551c79c86cf5b323b017e01ef0321`); checks: `scripts/verify_raw_shard.py` (`2ee4393c3ce40c94e2c80309449e6ff0380de25573e37a587d6a82b57a9263da`), `scripts/hash_evict.py` (`9be115c164df3f3e2f84432cbc5edea87b540b6fd7fa6e07492042c56a373548`) |
| 3.05 | `history/restack_exl3_q3_experts.py` (`074138416317b3643ba835d0b838c27b5340122bf46df15f6ee08ce9da549cbf`), `history/restack_exl3_q3_full.py` (`064bd09fcc0508f359ff0e7f1d9b444d91c9804307966abd560a9bc2ee285606`); checks: `scripts/verify_raw_shard.py`, `scripts/hash_nocache.py` (`498d4f11317411f531d6036efaf0bdf59a32fc7a22c6831cb8ab668794f91424`) |
| 4.05 | `history/restack_exl3_q4_experts.py` (`1baf2224f8a07bcfe63e3b4950a6bef0fd15d1f4711a36e645cca592d526e21c`), `history/restack_exl3_q4_full.py` (`5a2ccf5c807672a160a9424fdea15aff3fcf6adcd9eb3b0f9118bd8732c74381`) |
| 5.05 | `history/restack_exl3_qwen38.py` (`32aed8fa7c80c9dc10c8af2e53b5061bd7c662dc3c0dea4844f1522c990a8376`), `history/restack_exl3_full.py` (`720290dff6c6b880e9ee9357b85edae2a693ed0d4d432fc6df247c51315ec02a`) and `history/rebuild_505_mixer.py` (`f64324d87d50fbe6a1a81d4c6d89e7a25d91c8f0b7d8f264bf29820cf0a80416`; the run used `bfebd80537bcba9d368cf98024aa57682c57194cc36c9b8941b09491060ec462`, see [The 5.05 MTP mixer](#the-505-mtp-mixer)) |

The 5.05 pack took three steps. `restack_exl3_qwen38.py` stacked turboderp's experts into a pack whose other tensors
came from [ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit).
Then `restack_exl3_full.py` replaced that trunk with turboderp's. It kept the experts, the vision shard, the small files
and the 8-bit MTP mixer from the first pack. Its `--trunk` input was a folder of turboderp's non-expert tensors, read
from his 5.05 branch with HTTP range requests by a script that we no longer have. Last, `rebuild_505_mixer.py` replaced
the 8-bit mixer with turboderp's own (see [The 5.05 MTP mixer](#the-505-mtp-mixer)).

The docstrings mention `reports/find_cached.py`. That is an internal tool that is not in this repository. Each script has
its own copy of the cache eviction.

## License

The scripts: MIT (see `LICENSE`). The packs they make are derived from Qwen3.8-Flash-Next and stay under the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).

## Credits

- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): the model.
- [turboderp](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3): the EXL3 quantizations, made with
  [exllamav3](https://github.com/turboderp-org/exllamav3).
- [ddalcu](https://github.com/ddalcu): mlx-serve and its pack layout for this model.
