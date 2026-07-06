//! # xvc
//!
//! The **X-VC** engine — *Zero-shot Streaming Voice Conversion in Codec
//! Space* ([arXiv:2604.12456](https://arxiv.org/abs/2604.12456)).
//!
//! X-VC is the workspace's language-agnostic engine candidate: its semantic
//! side is the GLM-4-Voice tokenizer (Whisper-encoder based, multilingual
//! incl. Japanese), removing the Mandarin lock of MeanVC.
//!
//! Implemented so far (Phase 1, stage-by-stage weight-compatible port on
//! top of [`vc_core`]; see
//! [`docs/xvc.md`](https://github.com/m96-chan/babiniku.rs/blob/main/docs/xvc.md)
//! and [issue #30](https://github.com/m96-chan/babiniku.rs/issues/30)):
//!
//! * [`codec`] — the SAC acoustic codec ([`SacCodec`]): DAC-style
//!   encoder + factorized VQ (encode) and the DAC/HiFiGAN-style decoder
//!   that synthesizes the 16 kHz waveform directly (decode);
//! * [`converter`] — the 6-block MMDiT dual-conditioning converter
//!   ([`AcousticConverter`]): one non-iterative conversion pass in codec
//!   space.
//!
//! Both load the official weights converted by
//! `tools/convert_xvc_generator.py` and are golden-verified against the
//! official PyTorch implementation (`tests/golden_codec.rs`,
//! `tests/golden_converter.rs`).

pub mod codec;
pub mod converter;

pub use codec::{SacCodec, SacCodecConfig, SacEncodeOutput};
pub use converter::{AcousticConverter, AcousticConverterConfig};
