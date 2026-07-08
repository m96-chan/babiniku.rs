# Contributing to babiniku.rs

Thanks for your interest! Contributions are welcome — here is the short
version of how they work in this repository.

## The one rule that gates everything

**Code arrives as a fork + pull request. Nothing else.**

- We do **not** accept code, patches, or files as issue attachments —
  ZIP/archive attachments from non-collaborators are auto-hidden as
  spam by a moderation workflow, unread. (A wave of malware-drop spam
  made this policy explicit; it will not be relaxed.)
- Inline screenshots/logs in bug reports are fine and appreciated.

## Before you write code

Read [`CLAUDE.md`](CLAUDE.md) — its "Development rules" apply to humans
too. The essentials:

1. **Issue first.** Work is scoped in a GitHub issue before code exists
   (rule 6). If there is no issue for what you want to do, open one and
   get a nod before investing effort.
2. **TDD.** New behavior comes with tests, written first (rule 1).
3. **Demo before push.** A green test suite is necessary, not
   sufficient — run the real thing and record *what you ran and what
   you observed* in the PR description (rule 2).
4. **Docs are part of done.** README / `docs/*.md` / the issue itself
   must reflect your change (rules 5 & 7).

## Build & test

```sh
cargo build
cargo test                       # golden tests skip when ckpt/ is absent
cargo clippy --all-targets       # must be clean
cargo fmt
```

- CPU is the baseline: nothing may *require* a GPU or the CUDA Toolkit
  (GPU support stays behind the opt-in `cuda`/`metal` features).
- Checkpoints (`ckpt/`) are optional for most development — parity
  ("golden") tests are skip-if-absent. Engine docs
  ([meanvc](docs/meanvc.md), [xvc](docs/xvc.md),
  [seedvc](docs/seedvc.md)) describe checkpoint setup where needed.

## Where things live

| Area | Crate / label |
|---|---|
| Engine-agnostic DSP & traits | `crates/vc-core` |
| MeanVC v1/v2 | `crates/meanvc` — label `meanvc` / `meanvc2` |
| X-VC | `crates/xvc` — label `xvc` |
| Seed-VC (**GPL-3.0**, feature-gated) | `crates/seedvc` — label `seedvc` |
| TUI app + audio backends | `crates/babiniku` — label `tui` |
| Tooling / CI / packaging | label `infra` |

## Tooling language policy

- Anything a **user** runs is **Rust**.
- Golden-fixture generators stay **Python by design** — they must run
  the *official* PyTorch implementations to remain an independent
  ground truth. Do not port them to Rust.

## Licensing of contributions

The project is dual-licensed **MIT OR Apache-2.0**, with one exception:
`crates/seedvc` is **GPL-3.0** (its upstream code and weights are GPL).

By submitting a pull request you agree that your contribution is
licensed under the license of the crate it touches (MIT OR Apache-2.0
everywhere except `crates/seedvc`, which is GPL-3.0), without
additional terms.

One more thing that matters in a voice-conversion project: **never
commit reference audio of real people or characters** — recordings stay
out of the repository, and converted-voice use requires the consent of
the voice's owner.
