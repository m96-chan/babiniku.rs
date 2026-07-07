//! BigVGAN v2 vocoder — candle port of NVIDIA's
//! `bigvgan_v2_22khz_80band_256x` generator (arXiv:2206.04658), the mel →
//! waveform stage of the Seed-VC pipeline (`vocoder_fn(vc_mel).squeeze(1)`).
//!
//! Architecture (`modules/bigvgan/bigvgan.py` of the official repo):
//! `conv_pre` (k7, 80 → 1536) → 6 transposed-conv upsampling stages
//! (rates `[4,4,2,2,2,2]`, kernels `[8,8,4,4,4,4]`, channels halving each
//! stage) each followed by 3 parallel [`AmpBlock1`]s (kernels `[3,7,11]`,
//! dilations `[1,3,5]`, outputs averaged) → alias-free SnakeBeta →
//! `conv_post` (k7 → 1, no bias) → `clamp(−1, 1)` (this checkpoint sets
//! `use_tanh_at_final: false` and `use_bias_at_final: false`).
//!
//! The AMP blocks use SnakeBeta activations (`snake_logscale: true`)
//! wrapped in the alias-free [`Activation1d`]: 2× upsample with a
//! kaiser-windowed sinc low-pass (12 taps), the pointwise snake at the
//! doubled rate, then a 2× low-pass decimation back. The two filters are
//! deterministic buffers of the checkpoint (`upsample.filter`,
//! `downsample.lowpass.filter`) and are loaded rather than re-derived.
//!
//! Layout and parameter names mirror the official implementation 1:1
//! (`conv_pre`, `ups.{i}.0`, `resblocks.{i*3+j}.{convs1,convs2,activations}`,
//! `activation_post`, `conv_post`); the `weight_norm` parametrizations are
//! folded into plain conv weights by `tools/convert_seedvc.py`.

use std::path::Path;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    conv1d, conv1d_no_bias, conv_transpose1d, Conv1d, Conv1dConfig, ConvTranspose1d,
    ConvTranspose1dConfig, Module, VarBuilder,
};

use vc_core::Result;

/// Generator hyperparameters (the fields of the official `config.json`
/// that shape the network).
#[derive(Debug, Clone)]
pub struct BigVganConfig {
    /// Mel bins of the input spectrogram.
    pub num_mels: usize,
    /// Channels after `conv_pre`; halves at every upsampling stage.
    pub upsample_initial_channel: usize,
    /// Per-stage upsampling rates; their product is the hop (256).
    pub upsample_rates: Vec<usize>,
    /// Transposed-convolution kernel sizes (`padding = (k − rate) / 2`).
    pub upsample_kernel_sizes: Vec<usize>,
    /// Kernel sizes of the parallel AMP blocks at each stage.
    pub resblock_kernel_sizes: Vec<usize>,
    /// Dilations of `convs1` within each AMP block.
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
}

impl Default for BigVganConfig {
    /// `nvidia/bigvgan_v2_22khz_80band_256x`: 80 mels @ 22 050 Hz,
    /// 256× upsampling.
    fn default() -> Self {
        Self {
            num_mels: 80,
            upsample_initial_channel: 1536,
            upsample_rates: vec![4, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
        }
    }
}

impl BigVganConfig {
    /// Waveform samples per mel frame (the product of the rates); 256.
    pub fn hop_length(&self) -> usize {
        self.upsample_rates.iter().product()
    }
}

/// SnakeBeta activation, log-scale variant (`snake_logscale: true`):
/// `x + sin²(x · e^α) / (e^β + 1e-9)` with per-channel `α`, `β`.
#[derive(Debug)]
struct SnakeBeta {
    /// `e^α`, reshaped to `[1, channels, 1]` at load time.
    alpha: Tensor,
    /// `1 / (e^β + 1e-9)`, reshaped to `[1, channels, 1]` at load time.
    beta_recip: Tensor,
}

impl SnakeBeta {
    fn new(channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let alpha = vb.get(channels, "alpha")?.exp()?.reshape((1, channels, 1))?;
        let beta = vb.get(channels, "beta")?.exp()?.reshape((1, channels, 1))?;
        Ok(Self {
            alpha,
            beta_recip: (beta + 1e-9)?.recip()?,
        })
    }

    /// `x`: `[batch, channels, time]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let s = x.broadcast_mul(&self.alpha)?.sin()?.sqr()?;
        x + s.broadcast_mul(&self.beta_recip)?
    }
}

/// Ratio and kernel size of the alias-free resampling pair
/// (`Activation1d` defaults of the official implementation).
const AA_RATIO: usize = 2;
const AA_KERNEL: usize = 12;

/// Alias-free activation (`alias_free_activation/torch/act.py`):
/// 2× upsample → [`SnakeBeta`] → 2× downsample, both resamplers using a
/// 12-tap kaiser-windowed sinc low-pass at `cutoff 0.25 / half_width 0.3`.
///
/// The filter is identical across channels (a `[1, 1, 12]` buffer expanded
/// to `groups = C` upstream), so the per-channel convolutions collapse to a
/// single-channel conv over a `[B·C, 1, T]` view.
#[derive(Debug)]
struct Activation1d {
    act: SnakeBeta,
    /// `upsample.filter`, `[1, 1, 12]`.
    up_filter: Tensor,
    /// `downsample.lowpass.filter`, `[1, 1, 12]`.
    down_filter: Tensor,
}

impl Activation1d {
    fn new(channels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            act: SnakeBeta::new(channels, vb.pp("act"))?,
            up_filter: vb.get((1, 1, AA_KERNEL), "upsample.filter")?,
            down_filter: vb.get((1, 1, AA_KERNEL), "downsample.lowpass.filter")?,
        })
    }

    /// `UpSample1d`: replicate-pad by `K/ratio − 1 = 5`, transposed conv
    /// (stride 2) scaled by the ratio, then trim `pad·stride + (K − stride)/2
    /// = 15` on both sides → exactly `2 T` samples.
    fn upsample(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        let pad = AA_KERNEL / AA_RATIO - 1;
        let trim = pad * AA_RATIO + (AA_KERNEL - AA_RATIO) / 2;
        let x = x.pad_with_same(D::Minus1, pad, pad)?.reshape((b * c, 1, t + 2 * pad))?;
        let x = x.conv_transpose1d(&self.up_filter, 0, 0, AA_RATIO, 1, 1)?;
        let x = (x * AA_RATIO as f64)?;
        x.narrow(D::Minus1, trim, AA_RATIO * t)?.reshape((b, c, AA_RATIO * t))
    }

    /// `DownSample1d` / `LowPassFilter1d`: replicate-pad `(K/2 − 1, K/2)`,
    /// then a strided low-pass conv → `T / 2` samples.
    fn downsample(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        let x = x
            .pad_with_same(D::Minus1, AA_KERNEL / 2 - 1, AA_KERNEL / 2)?
            .reshape((b * c, 1, t + AA_KERNEL - 1))?;
        let x = x.conv1d(&self.down_filter, 0, AA_RATIO, 1, 1)?;
        x.reshape((b, c, t / AA_RATIO))
    }

    /// `x`: `[batch, channels, time]` → same shape.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.upsample(x)?;
        let x = self.act.forward(&x)?;
        self.downsample(&x)
    }
}

/// AMPBlock1: three `snake → dilated conv → snake → conv` residual pairs
/// (`convs1` dilated by `[1, 3, 5]`, `convs2` at dilation 1, all
/// length-preserving), every activation alias-free.
#[derive(Debug)]
struct AmpBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    /// `2 · dilations.len()` activations, consumed pairwise.
    acts: Vec<Activation1d>,
}

impl AmpBlock1 {
    fn new(
        channels: usize,
        kernel: usize,
        dilations: &[usize],
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let conv = |dilation: usize, vb: VarBuilder| {
            let cfg = Conv1dConfig {
                padding: (kernel - 1) * dilation / 2,
                dilation,
                ..Default::default()
            };
            conv1d(channels, channels, kernel, cfg, vb)
        };
        let convs1 = dilations
            .iter()
            .enumerate()
            .map(|(i, &d)| conv(d, vb.pp(format!("convs1.{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let convs2 = (0..dilations.len())
            .map(|i| conv(1, vb.pp(format!("convs2.{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let acts = (0..2 * dilations.len())
            .map(|i| Activation1d::new(channels, vb.pp(format!("activations.{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            convs1,
            convs2,
            acts,
        })
    }

    /// `x`: `[batch, channels, time]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        for (i, (c1, c2)) in self.convs1.iter().zip(&self.convs2).enumerate() {
            let xt = self.acts[2 * i].forward(&x)?;
            let xt = c1.forward(&xt)?;
            let xt = self.acts[2 * i + 1].forward(&xt)?;
            let xt = c2.forward(&xt)?;
            x = (xt + x)?;
        }
        Ok(x)
    }
}

/// The BigVGAN v2 generator.
#[derive(Debug)]
pub struct BigVgan {
    conv_pre: Conv1d,
    /// One transposed conv per stage (`ups.{i}.0`).
    ups: Vec<ConvTranspose1d>,
    /// `stages × num_kernels` AMP blocks, stage-major.
    resblocks: Vec<AmpBlock1>,
    activation_post: Activation1d,
    conv_post: Conv1d,
    num_kernels: usize,
}

impl BigVgan {
    /// Loads the generator from a converted safetensors checkpoint
    /// (`tools/convert_seedvc.py` output, weight norm folded).
    pub fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(BigVganConfig::default(), vb)
    }

    /// Builds the generator from a [`VarBuilder`] rooted at the generator
    /// state dict.
    pub fn new(cfg: BigVganConfig, vb: VarBuilder) -> Result<Self> {
        let conv_pre = conv1d(
            cfg.num_mels,
            cfg.upsample_initial_channel,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_pre"),
        )?;

        let mut ups = Vec::with_capacity(cfg.upsample_rates.len());
        let mut resblocks = Vec::new();
        for (i, (&rate, &kernel)) in cfg
            .upsample_rates
            .iter()
            .zip(&cfg.upsample_kernel_sizes)
            .enumerate()
        {
            let in_ch = cfg.upsample_initial_channel >> i;
            ups.push(conv_transpose1d(
                in_ch,
                in_ch / 2,
                kernel,
                ConvTranspose1dConfig {
                    padding: (kernel - rate) / 2,
                    stride: rate,
                    ..Default::default()
                },
                vb.pp(format!("ups.{i}.0")),
            )?);
            for (j, (&k, d)) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(&cfg.resblock_dilation_sizes)
                .enumerate()
            {
                let idx = i * cfg.resblock_kernel_sizes.len() + j;
                resblocks.push(AmpBlock1::new(in_ch / 2, k, d, vb.pp(format!("resblocks.{idx}")))?);
            }
        }

        let ch = cfg.upsample_initial_channel >> cfg.upsample_rates.len();
        let activation_post = Activation1d::new(ch, vb.pp("activation_post"))?;
        // `use_bias_at_final: false` for this checkpoint.
        let conv_post = conv1d_no_bias(
            ch,
            1,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_post"),
        )?;

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            activation_post,
            conv_post,
            num_kernels: cfg.resblock_kernel_sizes.len(),
        })
    }

    /// Mel `[batch, num_mels, frames]` → waveform `[batch, 1,
    /// frames · 256]` in `[−1, 1]` (the official `forward`; callers squeeze
    /// dim 1, as in `vocoder_fn(vc_mel).squeeze(1)`).
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_pre.forward(mel)?;
        for (i, up) in self.ups.iter().enumerate() {
            x = up.forward(&x)?;
            let mut xs = self.resblocks[i * self.num_kernels].forward(&x)?;
            for j in 1..self.num_kernels {
                xs = (xs + self.resblocks[i * self.num_kernels + j].forward(&x)?)?;
            }
            x = (xs / self.num_kernels as f64)?;
        }
        let x = self.activation_post.forward(&x)?;
        let x = self.conv_post.forward(&x)?;
        // `use_tanh_at_final: false` → hard clamp.
        Ok(x.clamp(-1.0, 1.0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;
    use std::collections::HashMap;

    fn ckpt(name: &str) -> Option<std::path::PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../ckpt/{name}"));
        path.exists().then_some(path)
    }

    fn fixture(name: &str) -> Option<HashMap<String, Tensor>> {
        Some(candle_core::safetensors::load(ckpt(name)?, &Device::Cpu).unwrap())
    }

    fn max_abs_diff(got: &Tensor, want: &Tensor) -> f32 {
        assert_eq!(got.dims(), want.dims(), "shape mismatch");
        (got - want)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap()
    }

    /// Golden vs the official implementation run in fp32 on CPU
    /// (`vc_mel` → `conv_pre` / `wave` captured from
    /// `BigVGAN.from_pretrained('nvidia/bigvgan_v2_22khz_80band_256x')`
    /// after `remove_weight_norm()`). Measured parity: `conv_pre`
    /// 3.4e-5, `wave` 6.7e-6 max abs.
    #[test]
    fn bigvgan_matches_official() {
        let Some(weights) = ckpt("seedvc_bigvgan.safetensors") else {
            return;
        };
        let Some(fx) = fixture("seedvc_bigvgan_fixture.safetensors") else {
            return;
        };
        let model = BigVgan::load(weights, &Device::Cpu).unwrap();

        // Cheap first-stage diagnostic before the full 22-second synth.
        let pre = model.conv_pre.forward(&fx["vc_mel"]).unwrap();
        let d = max_abs_diff(&pre, &fx["conv_pre"]);
        println!("conv_pre max abs diff {d:.2e}");
        assert!(d < 1e-4, "conv_pre mismatch: {d}");

        let wave = model.forward(&fx["vc_mel"]).unwrap();
        let d = max_abs_diff(&wave, &fx["wave"]);
        println!("bigvgan wave max abs diff {d:.2e}");
        assert!(d < 1e-3, "waveform mismatch: {d}");
    }

    /// End-to-end pipeline golden: `vc_mel` → waveform vs the `vc_wave`
    /// captured from the official full pipeline
    /// (`vocoder_fn(vc_mel).squeeze(1)`).
    ///
    /// That fixture was generated on CUDA where cuDNN uses TF32 convs by
    /// default; the official implementation itself, re-run in fp32 on CPU,
    /// differs from it by 1.92e-2 max abs (the deep snake/conv stack
    /// amplifies the TF32 rounding), so this check documents the looser
    /// device-noise bound while `bigvgan_matches_official` holds the tight
    /// fp32-parity bound.
    #[test]
    fn bigvgan_matches_e2e_pipeline() {
        let Some(weights) = ckpt("seedvc_bigvgan.safetensors") else {
            return;
        };
        let Some(fx) = fixture("seedvc_e2e_fixture.safetensors") else {
            return;
        };
        let model = BigVgan::load(weights, &Device::Cpu).unwrap();
        let wave = model.forward(&fx["vc_mel"]).unwrap().squeeze(1).unwrap();
        let got: Vec<f32> = wave.i(0).unwrap().to_vec1().unwrap();
        let want: Vec<f32> = fx["vc_wave"].i(0).unwrap().to_vec1().unwrap();
        assert_eq!(got.len(), want.len(), "sample count mismatch");
        let dmax = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        println!("bigvgan e2e max abs diff {dmax:.2e}");
        assert!(dmax < 2.5e-2, "waveform mismatch vs CUDA fixture: {dmax}");
    }
}
