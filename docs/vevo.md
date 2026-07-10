# Vevo-Timbre

A weight-compatible pure-candle port of **Vevo-Timbre**
([Amphion](https://github.com/open-mmlab/Amphion), ICLR 2025,
[OpenReview](https://openreview.net/pdf?id=anQDiQZhDP)) — style-preserved
zero-shot voice conversion via HuBERT-large content-style tokens, a
flow-matching DiffLlama converter, and a Vocos vocoder. Ported
checkpoint: `amphion/Vevo`'s `Vq8192ToMels` line (content-style tokenizer
+ FM ≈ 380 M params, Vocos ≈ 255 M params, 24 kHz).

> **License — read this first.** Amphion's **code is MIT**, so
> `crates/vevo` ships in the **default** MIT OR Apache-2.0 build — no
> feature gate, unlike [Seed-VC](seedvc.md)'s GPL crate. The **released
> weights are CC-BY-NC-4.0** — stricter than Seed-VC's GPL for our
> purposes (GPL permits commercial use; NC forbids it outright).
> `babiniku-fetch vevo` prompts for this before downloading; weights are
> never bundled into a distributed binary, and anything you produce with
> them stays non-commercial.

## Why we care, and why Vevo2 is parked instead

Recon findings (issue [#72](https://github.com/m96-chan/babiniku.rs/issues/72)):

- **Needle-clean**, same class as Seed-VC's BigVGAN line — the Vocos
  vocoder line adds **zero** events on the issue-#42 torture protocol
  (amplitude steps: source 418 detector events → output 0).
- **On par with Seed-VC by ear** (maintainer's A/B verdict) — no
  decisive quality win, but a second needle-free, permissively-licensed
  engine with an actively-maintained upstream (unlike seed-vc, which is
  archived).
- **Vevo2** (the newer speech+singing-unified line, tracked in the same
  recon) is explicitly **out of scope**: its 12.5 Hz content-style
  tokens (vs Vevo-Timbre's 50 Hz) can't track Japanese mora/sokuon
  timing — confirmed both by the needle scanner (sokuon torture: 474
  detector events on Vevo2's output vs 0 on Vevo-Timbre's) and by ear
  ("日本語がくずれてる"). Vevo2's weights also carry the stricter
  CC-BY-NC-**ND** license. Singing voice conversion — the one thing
  Vevo2 offers that nothing else here does — stays a noted idea for a
  future offline-tools issue, not a live engine.

## Architecture (as shipped in the checkpoint)

| stage | module | shape notes |
|---|---|---|
| content-style tokenizer | **HuBERT-large** (`facebook/hubert-large-ll60k`, torchaudio's fairseq port) layer 18 of 24 → **RepCodec** fvq8192 (`VocosBackbone` encoder, L2-normalized cosine-distance lookup) | 16 kHz → 50 Hz × 1024, z-normalized (`hubert_large_l18_mean_std.npz`); codebook 8192, dim 8 |
| converter | **DiffLlama** 1024 dim × 16 layers × 16 heads (RoPE + standard MHA/SwiGLU, every LayerNorm replaced by an adaptive-RMSNorm conditioned on the **diffusion timestep only** — content tokens are added once as an input-level bias, not fed into the per-layer norms) | CFM Euler sampling, rescaled CFG (cfg 1.0, rescale 0.75), 32 steps offline / 6 live |
| mel | 128 bins @ 24 kHz, n_fft 1920 / hop 480, `center=False`, mean −4.92 / var 8.14 normalization | HiFiGAN-style, `ln(clamp(x, 1e-5))` |
| vocoder | **Vocos** (ConvNeXt backbone, dim 1024 × 30 layers + ISTFT head, `padding="same"`) | 24 kHz out |

Two porting traps found via fixture-driven bisection (both documented
in code, `crates/vevo/src/hubert.rs`):

1. torchaudio's `HUBERT_LARGE` bundle z-normalizes the raw waveform
   before the feature extractor
   (`nn.functional.layer_norm(waveforms, waveforms.shape)`). This looks
   like a no-op at first glance — every conv in the feature extractor is
   bias-free and immediately followed by a per-position LayerNorm, which
   is scale-invariant — but the *re-centering* half does **not** cancel
   (per-channel kernel sums are nonzero), so skipping it is a real bug,
   not a harmless simplification.
2. torchaudio's `Transformer` class has its **own** `layer_norm_first`
   flag, separate from each `EncoderLayer`'s. For `HUBERT_LARGE` it is
   `False` — so `Transformer.layer_norm` never fires on the
   `extract_features()`/`get_intermediate_outputs()` path Vevo uses
   (only on the unused full `forward()`), even though every individual
   `EncoderLayer` has its own `layer_norm_first=True` and looks
   pre-LN at a glance. Wiring the transformer-level LN in anyway (the
   naive reading of the printed module tree) produced a 1215-max-abs-
   diff blowup by layer 18; removing it drops the residual to plain
   CPU/GPU float noise (correlation > 0.999 against the official
   implementation).

Also note: Vocos here uses `padding="same"` overlap-add
(`(n_fft - hop) / 2` trim, `T * hop` output samples) — a **different**
convention from `crates/meanvc`'s Vocos port, which implements
`torch.istft`'s `center=True` convention (`n_fft / 2` trim,
`(T - 1) * hop` samples). Same math, different final trim — ported
fresh in `crates/vevo/src/vocos.rs` rather than generalizing the
existing copy, matching this workspace's per-engine-crate precedent.

## Status: golden parity (`cargo test -p vevo`, skip-if-absent)

| stage | vs official |
|---|---|
| mel front-end | max abs < 1e-2 |
| HuBERT-large layer 18 | correlation > 0.999 (CPU vs the GPU-generated fixture) |
| RepCodec encoder / codes | correlation > 0.999 / match ratio > 0.95 |
| DiffLlama, both CFG branches | correlation > 0.999 |
| CFM, full 32-step trajectory | correlation > 0.999 |
| Vocos | correlation > 0.999 |
| **offline end-to-end** (`VevoEngine::inference_fm`) | correlation > 0.999 |

Every stage matched on its first fixture run except HuBERT (the two
traps above); the fixture generator (`tools/gen_vevo_fixtures.py`, runs
the official Amphion implementation — Python by design) captures the
CFM's initial noise and both CFG-branch attention inputs/outputs
alongside the usual per-stage tensors, so the DiffLlama/CFM tests are
bit-reproducible, not just "close."

## Streaming (live TUI) — real-time as of the #77/#79 fixes

`VevoStream` follows the same shape as [`SeedVcStream`](seedvc.md): a
sliding window (1.5 s context + 320 ms block) is re-processed through
the whole chain every hop and SOLA-spliced against the previous
emission's held-back tail. Unlike Seed-VC there is **no
length-regulation step** — content-style codes feed `cond_emb` directly,
frame-for-frame with the mel grid — so there's no separate
"long-context-for-content, short-context-for-diffusion" split either;
one window feeds HuBERT, RepCodec, and the CFM alike.

**This port was not real-time at first, and it was audible**: field
report — live mic use produced clicking/popping, not just delayed
audio. Root cause ([#77](https://github.com/m96-chan/babiniku.rs/issues/77)):
`candle_core::Tensor::conv1d_with_algo`'s `groups > 1` path issues one
CUDA dispatch **per group** (`chunk` → per-group `conv1d` → `cat`, no
native grouped kernel). `Vocos`'s ConvNeXt blocks are fully depthwise
(`groups == dim == 1024`), so every block paid **1024 tiny dispatches**
for one 7-tap conv — ~14 ms/block × 30 blocks ≈ 420 ms of the ~560 ms
total, while every other op in a block combined cost ~0.2 ms. Fixed by
hand-writing the depthwise conv as a shift-and-accumulate over the 7
kernel taps (`crates/vevo/src/vocos.rs::DepthwiseConv1d`) — O(kernel
size) whole-tensor ops instead of O(channels) tiny dispatches. Net:
**0.56 s → 0.15 s per 320 ms block** (3.7x), golden parity unaffected.

That headroom funded the [#79](https://github.com/m96-chan/babiniku.rs/issues/79)
fix: the original 0.5 s context was fast but too short for
HuBERT/RepCodec/DiffLlama to reliably resolve content (field report:
"pitch tracks, content doesn't" — a genuine short-context quality
ceiling, distinct from #77's latency issue). A needle scan
(`vc-core::declick::NeedleGuard`) across a context sweep found a real
ceiling of its own: 0 events up to 1.5 s context, 6+ events at 2.0 s
and beyond — longer context isn't free, it reintroduces needle-class
pulses this port was otherwise clean of. Landed at **1.5 s** (3x the
original, zero new artifacts) rather than chasing more context at the
cost of reopening #42-class needle risk.

Current measured cost (RTX 5090): **~0.16 s per 320 ms block, `late
0`** through the full TUI pipeline (mic preprocessing, soft-gate
expander included) — comfortable real-time headroom (~50 % of
budget). `--vevo-steps` (default 6) still exists for quality/compute
tuning but the CFM loop was never the bottleneck. `VevoEngine::
inference_fm` (offline) remains unaffected either way — it never had a
latency budget to miss.

## Usage

```sh
# one-time: fetch + convert the official checkpoints (pure Rust, no Python)
cargo run --release -p babiniku-fetch -- vevo
# → ckpt/vevo_{hubert,hubert_stats,repcodec,fmt,vocos}.safetensors (~1.7 GB fp32);
#   downloads from torchaudio's model hub + Hugging Face + a small repo-
#   checked-in stats file, with a CC-BY-NC-4.0 confirmation prompt
#   (--yes to skip).

# build + run (default build — no feature flag, MIT OR Apache-2.0)
cargo run --release -p babiniku --features cuda --bin babiniku -- \
    --engine vevo --reference her_voice_48k.wav --monitor --denoise
```

- **Use a 48 kHz reference** — same rationale as Seed-VC: the engine
  resamples internally, but a 16 kHz file starves the timbre prompt
  above 8 kHz and caps the voice-profile EQ target
  ([#62](https://github.com/m96-chan/babiniku.rs/issues/62)).
- `--vevo-steps` (live default 6, offline default 32 via
  `VevoEngine::inference_fm`'s `steps` argument).
- Golden fixtures for development: `tools/gen_vevo_fixtures.py` (runs
  the official Amphion implementation — per CLAUDE.md it stays Python
  by design). Debug/bench examples: `stream_probe` (per-hop wall-time
  breakdown, writes `ckpt/vevo_stream_demo.wav`; `VEVO_CONTEXT_MS` /
  `VEVO_STEPS` env vars override `StreamConfig` for tuning;
  `VEVO_VOCOS_PROFILE=1` breaks the vocoder forward pass down
  block-by-block), `offline_convert` (true whole-utterance
  `inference_fm`, the path golden tests actually cover).

## Performance (RTX 5090, fp32)

| mode | measured |
|---|---|
| offline (32 steps, ~3 s clip) | sub-second per stage; no latency budget |
| streaming, 6 steps, 1.5 s context, 320 ms block | **~0.16 s/block, `late 0`** — real-time, ~50 % of budget |
| algorithmic latency | block 320 ms + SOLA/crossfade ~40 ms + output resample |

CPU is far from real-time for this engine (comparable to Seed-VC's CPU
class); `meanvc` remains the CPU baseline for engines without a GPU.

## Citation

```bibtex
@inproceedings{zhang2025vevo,
  title={Vevo: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglement},
  author={Zhang, Xueyao and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## Acknowledgements

- [open-mmlab/Amphion](https://github.com/open-mmlab/Amphion) — the
  official implementation and released weights (code MIT, weights
  CC-BY-NC-4.0).
- torchaudio's `HUBERT_LARGE` bundle — the content encoder
  (`facebook/hubert-large-ll60k`, fairseq-format weights on torch hub).
