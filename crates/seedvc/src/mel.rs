//! The HiFiGAN-style mel spectrogram of the official implementation
//! (`modules/audio.py::mel_spectrogram`): reflect-pad `(n_fft − hop)/2`,
//! periodic Hann window, STFT with `center=False`, magnitude
//! `sqrt(re² + im² + 1e-9)`, librosa Slaney mel filterbank
//! (`fmin 0 / fmax None` → Nyquist, per the shipped
//! `config_dit_mel_seed_uvit_whisper_small_wavenet.yml`), then
//! `ln(clamp(x, 1e-5))`.
//!
//! Preset: `n_fft 1024 / win 1024 / hop 256 / 80 mels @ 22 050 Hz`.

use rustfft::{num_complex::Complex64, FftPlanner};

pub const SR: usize = 22_050;
pub const N_FFT: usize = 1_024;
pub const HOP: usize = 256;
pub const N_MELS: usize = 80;
const FMIN: f64 = 0.0;
const FMAX: f64 = SR as f64 / 2.0; // config fmax "None" → Nyquist

fn hz_to_mel_slaney(f: f64) -> f64 {
    if f < 1_000.0 {
        f * 3.0 / 200.0
    } else {
        15.0 + (f / 1_000.0).ln() * (27.0 / (6.4f64).ln())
    }
}

fn mel_to_hz_slaney(m: f64) -> f64 {
    if m < 15.0 {
        m * 200.0 / 3.0
    } else {
        1_000.0 * ((m - 15.0) * (6.4f64).ln() / 27.0).exp()
    }
}

/// librosa `mel(sr, n_fft, n_mels, fmin, fmax)` — Slaney scale and
/// Slaney area normalization (librosa defaults, matching the official
/// `librosa_mel_fn` call).
fn mel_filterbank() -> Vec<Vec<f64>> {
    let n_bins = N_FFT / 2 + 1;
    let (mlo, mhi) = (hz_to_mel_slaney(FMIN), hz_to_mel_slaney(FMAX));
    let pts: Vec<f64> = (0..N_MELS + 2)
        .map(|i| mel_to_hz_slaney(mlo + (mhi - mlo) * i as f64 / (N_MELS + 1) as f64))
        .collect();
    let mut fb = vec![vec![0.0; n_bins]; N_MELS];
    for (m, row) in fb.iter_mut().enumerate() {
        let (lo, ctr, hi) = (pts[m], pts[m + 1], pts[m + 2]);
        let enorm = 2.0 / (hi - lo);
        for (k, w) in row.iter_mut().enumerate() {
            let f = k as f64 * SR as f64 / N_FFT as f64;
            let up = (f - lo) / (ctr - lo);
            let down = (hi - f) / (hi - ctr);
            *w = up.min(down).max(0.0) * enorm;
        }
    }
    fb
}

/// Streaming-free one-shot mel: `[T]` samples → `[N_MELS][frames]`
/// with `frames = (T − n_fft) / hop + 1` after the reflect pad
/// (matches `center=False` + pad `(n_fft − hop)/2` both sides).
pub struct MelExtractor {
    fb: Vec<Vec<f64>>,
    window: Vec<f64>,
}

impl Default for MelExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl MelExtractor {
    pub fn new() -> Self {
        // torch.hann_window: periodic.
        let window: Vec<f64> = (0..N_FFT)
            .map(|i| {
                let t = i as f64 / N_FFT as f64;
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * t).cos()
            })
            .collect();
        Self {
            fb: mel_filterbank(),
            window,
        }
    }

    pub fn extract(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let pad = (N_FFT - HOP) / 2;
        // Reflect pad.
        let mut y = Vec::with_capacity(samples.len() + 2 * pad);
        for i in (1..=pad).rev() {
            y.push(samples[i.min(samples.len() - 1)] as f64);
        }
        y.extend(samples.iter().map(|&s| s as f64));
        for i in 2..=pad + 1 {
            y.push(samples[samples.len().saturating_sub(i)] as f64);
        }
        let frames = if y.len() >= N_FFT {
            (y.len() - N_FFT) / HOP + 1
        } else {
            0
        };
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let n_bins = N_FFT / 2 + 1;
        let mut out = vec![vec![0f32; frames]; N_MELS];
        let mut buf = vec![Complex64::new(0.0, 0.0); N_FFT];
        for t in 0..frames {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = Complex64::new(y[t * HOP + i] * self.window[i], 0.0);
            }
            fft.process(&mut buf);
            let mag: Vec<f64> = buf[..n_bins]
                .iter()
                .map(|c| (c.norm_sqr() + 1e-9).sqrt())
                .collect();
            for (m, row) in self.fb.iter().enumerate() {
                let e: f64 = row.iter().zip(&mag).map(|(w, v)| w * v).sum();
                out[m][t] = (e.max(1e-5)).ln() as f32;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp, Tensor};
    use std::collections::HashMap;

    fn fixture() -> Option<HashMap<String, Tensor>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt/seedvc_e2e_fixture.safetensors");
        if !path.exists() {
            return None;
        }
        Some(candle_core::safetensors::load(path, &Device::Cpu).unwrap())
    }

    #[test]
    fn mel_matches_official() {
        let Some(fx) = fixture() else { return };
        let src: Vec<f32> = fx["source_22k"].i(0).unwrap().to_vec1().unwrap();
        let want = fx["mel"].i(0).unwrap().to_vec2::<f32>().unwrap();
        let got = MelExtractor::new().extract(&src);
        assert_eq!(got.len(), want.len());
        assert_eq!(got[0].len(), want[0].len(), "frame count mismatch");
        let mut dmax = 0f32;
        for (gr, wr) in got.iter().zip(&want) {
            for (g, w) in gr.iter().zip(wr) {
                dmax = dmax.max((g - w).abs());
            }
        }
        println!("mel max abs diff {dmax:.2e}");
        assert!(dmax < 1e-3, "mel mismatch: {dmax}");
    }
}
