# tools — PyTorch parity fixtures & checkpoint conversion

Support scripts for the PyTorch parity campaign
([issue #8](https://github.com/m96-chan/meanvc2.rs/issues/8)): each
`gen_*_fixture.py` script produces a small deterministic safetensors file
under `fixtures/` containing inputs, weights, and reference outputs computed
by PyTorch; the Rust golden tests in [`tests/golden.rs`](../tests/golden.rs)
rebuild the same computation in candle and compare.

Small fixtures are **committed**, so `cargo test` exercises the golden tests
without a Python environment. Regenerate after changing a script:

```sh
pip install -r tools/requirements.txt
python3 tools/gen_jvp_fixture.py
cargo test --test golden
```

## Scripts

| Script | Fixture | Validates | Status |
|---|---|---|---|
| `gen_jvp_fixture.py` | `fixtures/jvp.safetensors` | `candle_core::forward_ad::jvp` vs `torch.func.jvp` on a mini DiT-like graph | ✅ |
| `gen_mel_fixture.py` | `fixtures/mel.safetensors` | `MelSpectrogram` vs torchaudio | planned (#8) |
| `convert_vocos.py` | — | gemelo-ai Vocos checkpoint → safetensors + golden output | planned (#8) |
| `convert_ecapa.py` | — | SpeechBrain ECAPA checkpoint → safetensors + golden output | planned (#8) |
| `convert_wenet.py` | — | WeNet `encoder.*` checkpoint → safetensors + golden output | planned (#8) |
