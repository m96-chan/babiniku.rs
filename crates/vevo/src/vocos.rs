//! Vocos vocoder (`models/codec/amphion_codec/vocos.py::Vocos`):
//! ConvNeXt backbone + ISTFT head, `padding="same"` — a **different**
//! overlap-add trim convention than `crates/meanvc`'s Vocos port
//! (which implements torch's `center=True`/`torch.istft` convention).
//! `"same"` trims `(win_length - hop_length) / 2` from each end
//! instead of `n_fft / 2`, giving `T * hop` output samples instead of
//! `(T - 1) * hop` — ported fresh here rather than generalizing the
//! meanvc copy, matching this workspace's per-engine-crate precedent.

use candle_core::{Device, Module, Tensor};
use candle_nn::{conv1d, layer_norm, linear, Conv1d, Conv1dConfig, LayerNorm, LayerNormConfig, Linear, VarBuilder};
use rustfft::{num_complex::Complex32, FftPlanner};

/// Depthwise (`groups == in_channels`) 1D convolution, implemented as a
/// shift-and-accumulate over the (small) kernel taps instead of via
/// candle's generic `groups > 1` path.
///
/// **Why**: `candle_core::Tensor::conv1d_with_algo` handles `groups > 1`
/// by `chunk`-ing the input/kernel into `groups` pieces and issuing one
/// `conv1d_single_group` CUDA call per group, then `cat`-ing the
/// results back together (`candle-core/src/conv.rs`) — no native
/// grouped/depthwise kernel. For a fully depthwise conv (`groups ==
/// dim == 1024` here), that's **1024 separate tiny CUDA dispatches per
/// block**, measured at ~14 ms/block (30 blocks ⇒ ~420 ms, the entire
/// #77 Vocos bottleneck — every other op in the block combined costs
/// ~0.2 ms). This implementation instead does `kernel_size` (7)
/// broadcast multiply-accumulates over the *whole* `[B, C, T]` tensor
/// at once — O(kernel_size) dispatches instead of O(channels).
struct DepthwiseConv1d {
    /// `[channels, kernel_size]` (the PyTorch `[out, in/groups=1, k]`
    /// weight with the redundant middle dim squeezed).
    weight: Tensor,
    bias: Tensor,
    kernel_size: usize,
    padding: usize,
}

impl DepthwiseConv1d {
    fn load(channels: usize, kernel_size: usize, padding: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get((channels, 1, kernel_size), "weight")?.squeeze(1)?;
        let bias = vb.get(channels, "bias")?;
        Ok(Self {
            weight,
            bias,
            kernel_size,
            padding,
        })
    }

    /// `x`: `[batch, channels, time]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (_b, c, t) = x.dims3()?;
        let xp = x.pad_with_zeros(2, self.padding, self.padding)?;
        let mut acc: Option<Tensor> = None;
        for k in 0..self.kernel_size {
            let xk = xp.narrow(2, k, t)?; // [b, c, t]
            let wk = self.weight.narrow(1, k, 1)?.reshape((1, c, 1))?; // [1, c, 1]
            let term = xk.broadcast_mul(&wk)?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        acc.unwrap().broadcast_add(&self.bias.reshape((1, c, 1))?)
    }
}

use vc_core::Result;

#[derive(Debug, Clone)]
pub struct VocosConfig {
    pub input_channels: usize,
    pub dim: usize,
    pub intermediate_dim: usize,
    pub num_layers: usize,
    pub n_fft: usize,
    pub hop_size: usize,
    pub sample_rate: usize,
}

impl Default for VocosConfig {
    fn default() -> Self {
        Self {
            input_channels: 128,
            dim: 1024,
            intermediate_dim: 4096,
            num_layers: 30,
            n_fft: 1920,
            hop_size: 480,
            sample_rate: 24_000,
        }
    }
}

struct ConvNeXtBlock {
    dwconv: DepthwiseConv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn new(dim: usize, intermediate_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            dwconv: DepthwiseConv1d::load(dim, 7, 3, vb.pp("dwconv"))?,
            norm: layer_norm(dim, LayerNormConfig::default(), vb.pp("norm"))?,
            pwconv1: linear(dim, intermediate_dim, vb.pp("pwconv1"))?,
            pwconv2: linear(intermediate_dim, dim, vb.pp("pwconv2"))?,
            gamma: vb.get((dim,), "gamma")?,
        })
    }

    fn forward_profiled(&self, x: &Tensor, profile: bool) -> candle_core::Result<Tensor> {
        let dev = x.device();
        macro_rules! step {
            ($label:expr, $e:expr) => {{
                let t0 = std::time::Instant::now();
                let r = $e;
                if profile {
                    dev.synchronize()?;
                    eprintln!("    {}: {:.5}s", $label, t0.elapsed().as_secs_f32());
                }
                r
            }};
        }
        let residual = x;
        let x = step!("dwconv", self.dwconv.forward(x))?;
        let x = step!("transpose1", x.transpose(1, 2))?;
        let x = step!("norm", self.norm.forward(&x))?;
        let x = step!("pwconv1", self.pwconv1.forward(&x))?;
        let x = step!("gelu", x.gelu_erf())?;
        let x = step!("pwconv2", self.pwconv2.forward(&x))?;
        let x = step!("gamma_mul", x.broadcast_mul(&self.gamma))?;
        let x = step!("transpose2", x.transpose(1, 2))?;
        step!("residual_add", residual + x)
    }
}

pub struct Vocos {
    embed: Conv1d,
    norm: LayerNorm,
    blocks: Vec<ConvNeXtBlock>,
    final_norm: LayerNorm,
    out: Linear,
    window: Vec<f32>,
    ifft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    cfg: VocosConfig,
}

impl Vocos {
    pub fn new(cfg: VocosConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let embed_cfg = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let bb = vb.pp("backbone");
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(ConvNeXtBlock::new(
                cfg.dim,
                cfg.intermediate_dim,
                bb.pp(format!("convnext.{i}")),
            )?);
        }
        // torch.hann_window default (periodic=True): sin^2(pi*n/N).
        let window: Vec<f32> = (0..cfg.n_fft)
            .map(|i| {
                let x = std::f32::consts::PI * i as f32 / cfg.n_fft as f32;
                x.sin().powi(2)
            })
            .collect();
        Ok(Self {
            embed: conv1d(cfg.input_channels, cfg.dim, 7, embed_cfg, bb.pp("embed"))?,
            norm: layer_norm(cfg.dim, LayerNormConfig::default(), bb.pp("norm"))?,
            blocks,
            final_norm: layer_norm(
                cfg.dim,
                LayerNormConfig::default(),
                bb.pp("final_layer_norm"),
            )?,
            out: linear(cfg.dim, cfg.n_fft + 2, vb.pp("head.out"))?,
            window,
            ifft: FftPlanner::new().plan_fft_inverse(cfg.n_fft),
            cfg,
        })
    }

    pub fn load<P: AsRef<std::path::Path>>(
        cfg: VocosConfig,
        path: P,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, device)?
        };
        Self::new(cfg, vb).map_err(Into::into)
    }

    pub fn config(&self) -> &VocosConfig {
        &self.cfg
    }

    fn spectrogram(&self, mel: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let profile = std::env::var("VEVO_VOCOS_PROFILE").is_ok();
        let dev = mel.device();
        let t0 = std::time::Instant::now();
        let x = self.embed.forward(mel)?;
        let x = self.norm.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        if profile {
            dev.synchronize()?;
            eprintln!("  embed+norm: {:.4}s", t0.elapsed().as_secs_f32());
        }
        let mut x = x;
        let mut block_total = 0f32;
        for (i, block) in self.blocks.iter().enumerate() {
            let tb = std::time::Instant::now();
            x = block.forward_profiled(&x, profile && i == 0)?;
            if profile {
                dev.synchronize()?;
                let dt = tb.elapsed().as_secs_f32();
                block_total += dt;
                if i == 0 || i == self.blocks.len() - 1 {
                    eprintln!("  block[{i}]: {dt:.4}s");
                }
            }
        }
        if profile {
            eprintln!("  blocks total ({} blocks): {:.4}s", self.blocks.len(), block_total);
        }
        let tf = std::time::Instant::now();
        let x = self.final_norm.forward(&x.transpose(1, 2)?)?;
        if profile {
            dev.synchronize()?;
            eprintln!("  final_norm: {:.4}s", tf.elapsed().as_secs_f32());
        }
        let th = std::time::Instant::now();
        let x = self.out.forward(&x)?.transpose(1, 2)?; // [b, n_fft+2, t]
        let half = self.cfg.n_fft / 2 + 1;
        // `mag = exp(raw); mag = clip(mag, max=1e2)` — no pre-exp clamp,
        // no lower post-exp clamp (exp is always positive already).
        let mag = x.narrow(1, 0, half)?.exp()?.minimum(1e2f32)?;
        let phase = x.narrow(1, half, half)?;
        let real = mag.mul(&phase.cos()?)?;
        let imag = mag.mul(&phase.sin()?)?;
        if profile {
            dev.synchronize()?;
            eprintln!("  head+trig: {:.4}s", th.elapsed().as_secs_f32());
        }
        Ok((real, imag))
    }

    /// Inverse STFT, `padding="same"` convention: overlap-add T frames
    /// spaced by `hop`, trim `(n_fft - hop) / 2` from each end (giving
    /// `T * hop` samples), normalized by the summed window-squared
    /// envelope.
    fn istft(&self, real: &[Vec<f32>], imag: &[Vec<f32>]) -> Vec<f32> {
        let n_fft = self.cfg.n_fft;
        let hop = self.cfg.hop_size;
        let half = n_fft / 2 + 1;
        let frames = real.len();
        let padded_len = (frames - 1) * hop + n_fft;
        let mut y = vec![0f32; padded_len];
        let mut wsum = vec![0f32; padded_len];

        let mut buf = vec![Complex32::default(); n_fft];
        for (frame, (re, im)) in real.iter().zip(imag.iter()).enumerate() {
            for bin in 0..half {
                buf[bin] = Complex32::new(re[bin], im[bin]);
            }
            for bin in half..n_fft {
                let src = buf[n_fft - bin];
                buf[bin] = Complex32::new(src.re, -src.im);
            }
            self.ifft.process(&mut buf);
            let offset = frame * hop;
            for i in 0..n_fft {
                let s = buf[i].re / n_fft as f32;
                y[offset + i] += s * self.window[i];
                wsum[offset + i] += self.window[i] * self.window[i];
            }
        }
        for (s, w) in y.iter_mut().zip(&wsum) {
            if *w > 1e-11 {
                *s /= w;
            }
        }
        let pad = (n_fft - hop) / 2;
        let len = frames * hop;
        y[pad..pad + len].to_vec()
    }

    /// `mel`: `[frames, num_mels]` or `[1, frames, num_mels]`.
    pub fn synthesize(&self, mel: &Tensor) -> Result<Vec<f32>> {
        let mel = match mel.dims().len() {
            2 => mel.clone(),
            3 => mel.squeeze(0)?,
            _ => {
                return Err(vc_core::Error::Input(
                    "mel must be [frames, num_mels] or [1, frames, num_mels]".into(),
                ))
            }
        };
        let (_frames, n_mels) = mel.dims2()?;
        if n_mels != self.cfg.input_channels {
            return Err(vc_core::Error::Input(format!(
                "expected {} mel bins, got {n_mels}",
                self.cfg.input_channels
            )));
        }
        let mel = mel.transpose(0, 1)?.unsqueeze(0)?; // [1, mel, t]
        let (real, imag) = self.spectrogram(&mel)?;
        let real: Vec<Vec<f32>> = real.squeeze(0)?.transpose(0, 1)?.to_vec2()?;
        let imag: Vec<Vec<f32>> = imag.squeeze(0)?.transpose(0, 1)?.to_vec2()?;
        Ok(self.istft(&real, &imag))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn fixture() -> Option<HashMap<String, Tensor>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt/vevo_e2e_fixture.safetensors");
        if !path.exists() {
            return None;
        }
        Some(candle_core::safetensors::load(path, &Device::Cpu).unwrap())
    }

    fn ckpt() -> Option<std::path::PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt/vevo_vocos.safetensors");
        path.exists().then_some(path)
    }

    #[test]
    fn synthesize_matches_official() {
        let (Some(fx), Some(ckpt)) = (fixture(), ckpt()) else {
            return;
        };
        let dev = Device::Cpu;
        let model = Vocos::load(VocosConfig::default(), ckpt, &dev).unwrap();

        let mel = fx["fm_mel"].clone(); // [1, t, 128]
        let want: Vec<f32> = fx["wave_out"]
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1()
            .unwrap();
        let got = model.synthesize(&mel).unwrap();

        assert_eq!(got.len(), want.len(), "sample count mismatch");
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (g, w) in got.iter().zip(&want) {
            dot += (*g as f64) * (*w as f64);
            na += (*g as f64).powi(2);
            nb += (*w as f64).powi(2);
        }
        let corr = dot / (na.sqrt() * nb.sqrt());
        assert!(corr > 0.999, "correlation {corr}");
    }
}
