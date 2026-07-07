//! CAM++ speaker (style) encoder — the `campplus_cn_common` D-TDNN of the
//! official pipeline (`modules/campplus/DTDNN.py::CAMPPlus(feat_dim=80,
//! embedding_size=192)` with the `funasr/campplus` checkpoint):
//! kaldi fbank (16 kHz, 80 bins, mean-subtracted over time) → FCM 2-D
//! conv front-end (`head`) → TDNN stem → 3 CAM dense-TDNN blocks
//! (12/24/16 layers, dilation 1/2/2) with transit layers → stats pooling
//! (mean ‖ unbiased std) → 192-dim dense embedding
//! (`style2 = campplus_model(feat2.unsqueeze(0))` in `inference.py`).
//!
//! The front-end is `torchaudio.compliance.kaldi.fbank(wave_16k,
//! num_mel_bins=80, dither=0, sample_frequency=16000)` with kaldi
//! defaults: 25 ms / 10 ms frames, `snip_edges=True`,
//! `remove_dc_offset=True`, `preemphasis_coefficient=0.97`, povey window
//! (symmetric Hann^0.85), zero-pad to a 512 FFT, power spectrum, kaldi
//! mel scale (`1127·ln(1 + f/700)`, `low_freq 20 / high_freq` Nyquist),
//! `log(max(e, 1.1921e-7))`.
//!
//! Weights: `ckpt/seedvc_campplus.safetensors` (re-exported by
//! `tools/convert_seedvc.py` from the standalone funasr
//! `campplus_cn_common.bin` — the main checkpoint's `style_encoder` is
//! an unrelated MelStyleEncoder that the inference path never uses).

use candle_core::{Device, Module, Tensor};
use candle_nn::{
    conv1d, conv1d_no_bias, conv2d_no_bias, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig,
    VarBuilder,
};
use rustfft::{num_complex::Complex64, FftPlanner};

use crate::Result;

pub const SR: usize = 16_000;
pub const NUM_MEL_BINS: usize = 80;
const FRAME_LENGTH: usize = 400; // 25 ms
const FRAME_SHIFT: usize = 160; // 10 ms
const PADDED_WINDOW_SIZE: usize = 512; // round_to_power_of_two
const LOW_FREQ: f64 = 20.0;
const HIGH_FREQ: f64 = SR as f64 / 2.0; // high_freq 0.0 → + Nyquist
const PREEMPHASIS_COEFFICIENT: f64 = 0.97;
const EPSILON: f64 = 1.192_092_895_507_812_5e-7; // torch.finfo(float32).eps

/// Kaldi mel scale: `1127 · ln(1 + f/700)`.
fn mel_scale(freq: f64) -> f64 {
    1_127.0 * (1.0 + freq / 700.0).ln()
}

/// `torchaudio.compliance.kaldi.get_mel_banks(80, 512, 16000, 20, 8000)`
/// — triangles on the kaldi mel scale over the first `512/2` FFT bins,
/// then a zero Nyquist column (torchaudio pads the bank with one zero).
fn mel_banks() -> Vec<Vec<f64>> {
    let num_fft_bins = PADDED_WINDOW_SIZE / 2;
    let fft_bin_width = SR as f64 / PADDED_WINDOW_SIZE as f64;
    let mel_low = mel_scale(LOW_FREQ);
    let mel_high = mel_scale(HIGH_FREQ);
    let mel_freq_delta = (mel_high - mel_low) / (NUM_MEL_BINS + 1) as f64;
    let mut banks = vec![vec![0.0; num_fft_bins + 1]; NUM_MEL_BINS];
    for (i, row) in banks.iter_mut().enumerate() {
        let left_mel = mel_low + i as f64 * mel_freq_delta;
        let center_mel = left_mel + mel_freq_delta;
        let right_mel = center_mel + mel_freq_delta;
        for (k, w) in row.iter_mut().enumerate().take(num_fft_bins) {
            let mel = mel_scale(fft_bin_width * k as f64);
            let up_slope = (mel - left_mel) / (center_mel - left_mel);
            let down_slope = (right_mel - mel) / (right_mel - center_mel);
            *w = up_slope.min(down_slope).max(0.0);
        }
    }
    banks
}

/// One-shot kaldi log-fbank: `[T]` samples @ 16 kHz → `[frames][80]`
/// (raw, before the pipeline's per-utterance mean subtraction) with
/// `frames = 1 + (T − 400) / 160` (`snip_edges=True`).
pub struct FbankExtractor {
    banks: Vec<Vec<f64>>,
    window: Vec<f64>,
}

impl Default for FbankExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FbankExtractor {
    pub fn new() -> Self {
        // Povey window: symmetric Hann (denominator N − 1) to the 0.85.
        let window: Vec<f64> = (0..FRAME_LENGTH)
            .map(|i| {
                let t = i as f64 / (FRAME_LENGTH - 1) as f64;
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * t).cos()).powf(0.85)
            })
            .collect();
        Self {
            banks: mel_banks(),
            window,
        }
    }

    pub fn extract(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        if samples.len() < FRAME_LENGTH {
            return Vec::new();
        }
        let m = 1 + (samples.len() - FRAME_LENGTH) / FRAME_SHIFT;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(PADDED_WINDOW_SIZE);
        let n_bins = PADDED_WINDOW_SIZE / 2 + 1;
        let mut out = Vec::with_capacity(m);
        let mut frame = [0f64; FRAME_LENGTH];
        let mut buf = vec![Complex64::new(0.0, 0.0); PADDED_WINDOW_SIZE];
        for t in 0..m {
            for (f, &s) in frame.iter_mut().zip(&samples[t * FRAME_SHIFT..]) {
                *f = s as f64;
            }
            // remove_dc_offset, then preemphasis with replicated x[−1] = x[0].
            let mean = frame.iter().sum::<f64>() / FRAME_LENGTH as f64;
            for f in frame.iter_mut() {
                *f -= mean;
            }
            for i in (1..FRAME_LENGTH).rev() {
                frame[i] -= PREEMPHASIS_COEFFICIENT * frame[i - 1];
            }
            frame[0] -= PREEMPHASIS_COEFFICIENT * frame[0];
            for (b, (f, w)) in buf.iter_mut().zip(frame.iter().zip(&self.window)) {
                *b = Complex64::new(f * w, 0.0);
            }
            for b in buf.iter_mut().skip(FRAME_LENGTH) {
                *b = Complex64::new(0.0, 0.0);
            }
            fft.process(&mut buf);
            let power: Vec<f64> = buf[..n_bins].iter().map(|c| c.norm_sqr()).collect();
            let row: Vec<f32> = self
                .banks
                .iter()
                .map(|bank| {
                    let e: f64 = bank.iter().zip(&power).map(|(w, p)| w * p).sum();
                    e.max(EPSILON).ln() as f32
                })
                .collect();
            out.push(row);
        }
        out
    }
}

/// Eval-mode `BatchNorm{1,2}d` folded to per-channel `x·scale + shift`
/// (`affine=False` for the final `batchnorm_` of the dense layer).
struct BatchNorm {
    scale: Tensor,
    shift: Tensor,
}

impl BatchNorm {
    fn load(channels: usize, affine: bool, vb: VarBuilder) -> Result<Self> {
        let mean = vb.get(channels, "running_mean")?;
        let var = vb.get(channels, "running_var")?;
        let denom = ((var + 1e-5)?).sqrt()?;
        let (scale, shift) = if affine {
            let scale = (vb.get(channels, "weight")? / &denom)?;
            let shift = (vb.get(channels, "bias")? - (mean * &scale)?)?;
            (scale, shift)
        } else {
            let scale = denom.recip()?;
            let shift = (mean * &scale)?.neg()?;
            (scale, shift)
        };
        Ok(Self { scale, shift })
    }

    /// Broadcast over any rank with dim 1 as channels.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut dims = vec![1; x.rank()];
        dims[1] = self.scale.dim(0)?;
        let y = x.broadcast_mul(&self.scale.reshape(dims.clone())?)?;
        Ok(y.broadcast_add(&self.shift.reshape(dims)?)?)
    }
}

/// candle `Conv2dConfig` has a single (uniform) stride, but the FCM uses
/// stride `(2, 1)` (halve frequency, keep time): run the conv at stride 1
/// and take every 2nd row — identical because
/// `conv(stride s)[i] = conv(stride 1)[s·i]`.
fn take_even_rows(x: &Tensor) -> Result<Tensor> {
    let h = x.dim(2)?;
    let idx: Vec<u32> = (0..h as u32).step_by(2).collect();
    let idx = Tensor::from_vec(idx.clone(), idx.len(), x.device())?;
    Ok(x.index_select(&idx, 2)?.contiguous()?)
}

/// `BasicResBlock` (3×3 + 3×3, 1×1-conv shortcut when `stride != 1`).
struct BasicResBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    shortcut: Option<(Conv2d, BatchNorm)>,
    stride: usize,
}

impl BasicResBlock {
    fn load(planes: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let shortcut = if stride != 1 {
            Some((
                conv2d_no_bias(planes, planes, 1, Default::default(), vb.pp("shortcut.0"))?,
                BatchNorm::load(planes, true, vb.pp("shortcut.1"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            conv1: conv2d_no_bias(planes, planes, 3, cfg, vb.pp("conv1"))?,
            bn1: BatchNorm::load(planes, true, vb.pp("bn1"))?,
            conv2: conv2d_no_bias(planes, planes, 3, cfg, vb.pp("conv2"))?,
            bn2: BatchNorm::load(planes, true, vb.pp("bn2"))?,
            shortcut,
            stride,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = self.conv1.forward(x)?;
        if self.stride != 1 {
            out = take_even_rows(&out)?;
        }
        let out = self.bn1.forward(&out)?.relu()?;
        let out = self.bn2.forward(&self.conv2.forward(&out)?)?;
        let res = match &self.shortcut {
            Some((conv, bn)) => bn.forward(&take_even_rows(&conv.forward(x)?)?)?,
            None => x.clone(),
        };
        Ok((out + res)?.relu()?)
    }
}

/// FCM front-end: `[B, 80, T]` → conv3×3 + 2 residual stages (each
/// halving the 80-bin frequency axis) + conv3×3 stride `(2, 1)` →
/// `[B, 32·10 = 320, T]`.
struct Fcm {
    conv1: Conv2d,
    bn1: BatchNorm,
    layers: Vec<BasicResBlock>,
    conv2: Conv2d,
    bn2: BatchNorm,
}

impl Fcm {
    const M_CHANNELS: usize = 32;

    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let mut layers = Vec::new();
        for l in 1..=2 {
            for (b, stride) in [2usize, 1].into_iter().enumerate() {
                layers.push(BasicResBlock::load(
                    Self::M_CHANNELS,
                    stride,
                    vb.pp(format!("layer{l}.{b}")),
                )?);
            }
        }
        Ok(Self {
            conv1: conv2d_no_bias(1, Self::M_CHANNELS, 3, cfg, vb.pp("conv1"))?,
            bn1: BatchNorm::load(Self::M_CHANNELS, true, vb.pp("bn1"))?,
            layers,
            conv2: conv2d_no_bias(Self::M_CHANNELS, Self::M_CHANNELS, 3, cfg, vb.pp("conv2"))?,
            bn2: BatchNorm::load(Self::M_CHANNELS, true, vb.pp("bn2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.unsqueeze(1)?; // [B, 1, F, T]
        let mut out = self.bn1.forward(&self.conv1.forward(&x)?)?.relu()?;
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        let out = take_even_rows(&self.conv2.forward(&out)?)?;
        let out = self.bn2.forward(&out)?.relu()?;
        let (b, c, f, t) = out.dims4()?;
        Ok(out.reshape((b, c * f, t))?)
    }
}

/// `CAMLayer::seg_pooling`: non-overlapping 100-frame average pooling
/// (`ceil_mode=True`, so the last partial segment averages its actual
/// length), each segment mean replicated back over its frames.
fn seg_pooling(x: &Tensor, seg_len: usize) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    let mut parts = Vec::new();
    let mut s = 0;
    while s < t {
        let len = seg_len.min(t - s);
        let m = x.narrow(2, s, len)?.mean_keepdim(2)?; // [B, C, 1]
        parts.push(m.expand((b, c, len))?.contiguous()?);
        s += len;
    }
    Ok(Tensor::cat(&parts, 2)?)
}

/// `CAMDenseTDNNLayer`: BN+ReLU → 1×1 bottleneck → BN+ReLU → `CAMLayer`
/// (local conv modulated by a sigmoid mask from global + segment context).
struct CamDenseTdnnLayer {
    nonlinear1: BatchNorm,
    linear1: Conv1d,
    nonlinear2: BatchNorm,
    linear_local: Conv1d,
    cam_linear1: Conv1d,
    cam_linear2: Conv1d,
}

impl CamDenseTdnnLayer {
    fn load(
        in_channels: usize,
        out_channels: usize,
        bn_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let local_cfg = Conv1dConfig {
            padding: (kernel_size - 1) / 2 * dilation,
            dilation,
            ..Default::default()
        };
        let cam = vb.pp("cam_layer");
        Ok(Self {
            nonlinear1: BatchNorm::load(in_channels, true, vb.pp("nonlinear1.batchnorm"))?,
            linear1: conv1d_no_bias(in_channels, bn_channels, 1, Default::default(), vb.pp("linear1"))?,
            nonlinear2: BatchNorm::load(bn_channels, true, vb.pp("nonlinear2.batchnorm"))?,
            linear_local: conv1d_no_bias(
                bn_channels,
                out_channels,
                kernel_size,
                local_cfg,
                cam.pp("linear_local"),
            )?,
            cam_linear1: conv1d(
                bn_channels,
                bn_channels / 2,
                1,
                Default::default(),
                cam.pp("linear1"),
            )?,
            cam_linear2: conv1d(
                bn_channels / 2,
                out_channels,
                1,
                Default::default(),
                cam.pp("linear2"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let z = self.linear1.forward(&self.nonlinear1.forward(x)?.relu()?)?;
        let z = self.nonlinear2.forward(&z)?.relu()?;
        let y = self.linear_local.forward(&z)?;
        let context = z.mean_keepdim(2)?.broadcast_add(&seg_pooling(&z, 100)?)?;
        let context = self.cam_linear1.forward(&context)?.relu()?;
        let m = candle_nn::ops::sigmoid(&self.cam_linear2.forward(&context)?)?;
        Ok((y * m)?)
    }
}

/// `TransitLayer` / `DenseLayer` share the shape BN+ReLU ∘ 1×1 conv (in
/// opposite order); only the transit variant is needed as a struct.
struct TransitLayer {
    nonlinear: BatchNorm,
    linear: Conv1d,
}

impl TransitLayer {
    fn load(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            nonlinear: BatchNorm::load(in_channels, true, vb.pp("nonlinear.batchnorm"))?,
            linear: conv1d_no_bias(
                in_channels,
                out_channels,
                1,
                Default::default(),
                vb.pp("linear"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.linear.forward(&self.nonlinear.forward(x)?.relu()?)?)
    }
}

/// `CAMPPlus(feat_dim=80, embedding_size=192)` — mean-normalized fbank
/// `[B, T, 80]` → 192-dim speaker embedding `[B, 192]`.
pub struct CampPlus {
    head: Fcm,
    tdnn_linear: Conv1d,
    tdnn_nonlinear: BatchNorm,
    blocks: Vec<Vec<CamDenseTdnnLayer>>,
    transits: Vec<TransitLayer>,
    out_nonlinear: BatchNorm,
    dense_linear: Tensor, // [192, 1024]
    dense_nonlinear: BatchNorm,
}

impl CampPlus {
    pub const EMBEDDING_SIZE: usize = 192;
    const GROWTH_RATE: usize = 32;
    const BN_SIZE: usize = 4;
    const INIT_CHANNELS: usize = 128;

    /// Expects the official `campplus_cn_common.bin` key names
    /// (`head.*`, `xvector.*`) at the `vb` root.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let head = Fcm::load(vb.pp("head"))?;
        let xv = vb.pp("xvector");
        let tdnn_cfg = Conv1dConfig {
            stride: 2,
            padding: 2,
            ..Default::default()
        };
        let mut channels = Self::INIT_CHANNELS;
        let mut blocks = Vec::new();
        let mut transits = Vec::new();
        for (i, (num_layers, kernel_size, dilation)) in
            [(12usize, 3usize, 1usize), (24, 3, 2), (16, 3, 2)].into_iter().enumerate()
        {
            let bvb = xv.pp(format!("block{}", i + 1));
            let mut layers = Vec::new();
            for l in 0..num_layers {
                layers.push(CamDenseTdnnLayer::load(
                    channels + l * Self::GROWTH_RATE,
                    Self::GROWTH_RATE,
                    Self::BN_SIZE * Self::GROWTH_RATE,
                    kernel_size,
                    dilation,
                    bvb.pp(format!("tdnnd{}", l + 1)),
                )?);
            }
            blocks.push(layers);
            channels += num_layers * Self::GROWTH_RATE;
            transits.push(TransitLayer::load(
                channels,
                channels / 2,
                xv.pp(format!("transit{}", i + 1)),
            )?);
            channels /= 2;
        }
        let dense = xv.pp("dense");
        Ok(Self {
            head,
            tdnn_linear: conv1d_no_bias(320, Self::INIT_CHANNELS, 5, tdnn_cfg, xv.pp("tdnn.linear"))?,
            tdnn_nonlinear: BatchNorm::load(
                Self::INIT_CHANNELS,
                true,
                xv.pp("tdnn.nonlinear.batchnorm"),
            )?,
            blocks,
            transits,
            out_nonlinear: BatchNorm::load(channels, true, xv.pp("out_nonlinear.batchnorm"))?,
            dense_linear: dense
                .get((Self::EMBEDDING_SIZE, 2 * channels, 1), "linear.weight")?
                .reshape((Self::EMBEDDING_SIZE, 2 * channels))?,
            dense_nonlinear: BatchNorm::load(
                Self::EMBEDDING_SIZE,
                false,
                dense.pp("nonlinear.batchnorm"),
            )?,
        })
    }

    /// `feat` `[B, T, 80]`, already mean-subtracted over time →
    /// `[B, 192]`.
    pub fn forward(&self, feat: &Tensor) -> Result<Tensor> {
        let x = feat.transpose(1, 2)?.contiguous()?; // (B,T,F) → (B,F,T)
        let x = self.head.forward(&x)?;
        let mut x = self
            .tdnn_nonlinear
            .forward(&self.tdnn_linear.forward(&x)?)?
            .relu()?;
        for (layers, transit) in self.blocks.iter().zip(&self.transits) {
            for layer in layers {
                x = Tensor::cat(&[&x, &layer.forward(&x)?], 1)?;
            }
            x = transit.forward(&x)?;
        }
        let x = self.out_nonlinear.forward(&x)?.relu()?;
        // StatsPool: mean ‖ std (unbiased) over time.
        let t = x.dim(2)?;
        let mean = x.mean(2)?; // [B, C]
        let centred = x.broadcast_sub(&mean.unsqueeze(2)?)?;
        let std = (centred.sqr()?.sum(2)? / (t as f64 - 1.0))?.sqrt()?;
        let stats = Tensor::cat(&[&mean, &std], 1)?; // [B, 2C]
        // DenseLayer: 1×1 linear (no bias) + BatchNorm1d(affine=False).
        let emb = stats.matmul(&self.dense_linear.t()?)?;
        self.dense_nonlinear.forward(&emb)
    }

    /// Convenience: raw fbank frames (`FbankExtractor::extract` output)
    /// → per-utterance mean subtraction → embedding `[1, 192]`.
    pub fn embed(&self, fbank: &[Vec<f32>], device: &Device) -> Result<Tensor> {
        let t = fbank.len();
        let flat: Vec<f32> = fbank.iter().flatten().copied().collect();
        let feat = Tensor::from_vec(flat, (t, NUM_MEL_BINS), device)?;
        let feat = feat.broadcast_sub(&feat.mean_keepdim(0)?)?;
        self.forward(&feat.unsqueeze(0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, IndexOp};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn ckpt(name: &str) -> Option<PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ckpt")
            .join(name);
        path.exists().then_some(path)
    }

    fn load(path: PathBuf) -> HashMap<String, Tensor> {
        candle_core::safetensors::load(path, &Device::Cpu).unwrap()
    }

    #[test]
    fn fbank_matches_official() {
        // `wave_16k` / `fbank` were generated with torchaudio
        // (`functional.resample(ref_22k, 22050, 16000)` then
        // `compliance.kaldi.fbank`) so the fbank golden is decoupled
        // from resampler parity.
        let Some(fx) = ckpt("seedvc_campplus_fixture.safetensors") else {
            return;
        };
        let fx = load(fx);
        let wave: Vec<f32> = fx["wave_16k"].i(0).unwrap().to_vec1().unwrap();
        let want = fx["fbank"].to_vec2::<f32>().unwrap();
        let got = FbankExtractor::new().extract(&wave);
        assert_eq!(got.len(), want.len(), "frame count mismatch");
        let mut dmax = 0f32;
        for (gr, wr) in got.iter().zip(&want) {
            for (g, w) in gr.iter().zip(wr) {
                dmax = dmax.max((g - w).abs());
            }
        }
        println!("fbank max abs diff {dmax:.2e}");
        assert!(dmax < 1e-3, "fbank mismatch: {dmax}");
    }

    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap()
    }

    // Tolerance note (#50): like the regulator fixture, the e2e
    // fixture's `style2` was generated on CUDA with TF32 matmuls/convs
    // (PyTorch default) — the official CAMPPlus itself, re-run in
    // strict fp32 on CPU on the fixture's own `feat2`, reproduces
    // `style2` only to 3.06e-3 max abs (cos 0.9999993). The `style` /
    // `style_wave` goldens in the campplus fixture are that strict-fp32
    // rerun, so the port is asserted to < 1e-3 against them and to
    // cos > 0.9999 against `style2` (its inherent TF32 noise floor).
    #[test]
    fn campplus_matches_official() {
        let (Some(weights), Some(fixture)) = (
            ckpt("seedvc_campplus.safetensors"),
            ckpt("seedvc_campplus_fixture.safetensors"),
        ) else {
            return;
        };
        let dev = Device::Cpu;
        let fx = load(fixture);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &dev).unwrap()
        };
        let model = CampPlus::load(vb).unwrap();

        // Golden 1: the exact e2e `feat2` → embedding.
        if let Some(e2e) = ckpt("seedvc_e2e_fixture.safetensors") {
            let e2e = load(e2e);
            let got = model.forward(&e2e["feat2"].unsqueeze(0).unwrap()).unwrap();
            let d = max_abs_diff(&got, &fx["style"]);
            let (d2, cos) = (max_abs_diff(&got, &e2e["style2"]), cosine(&got, &e2e["style2"]));
            println!("feat2→emb vs fp32 golden {d:.2e}; vs style2 max {d2:.2e} cos {cos:.7}");
            assert!(d < 1e-3, "embedding mismatch vs fp32 golden: {d}");
            assert!(cos > 0.9999, "cos vs style2: {cos}");
        }

        // Golden 2: wave → fbank → mean-sub → embedding (full front-end).
        let wave: Vec<f32> = fx["wave_16k"].i(0).unwrap().to_vec1().unwrap();
        let fbank = FbankExtractor::new().extract(&wave);
        let got = model.embed(&fbank, &dev).unwrap();
        let d = max_abs_diff(&got, &fx["style_wave"]);
        println!("wave→emb vs fp32 golden {d:.2e}");
        assert!(d < 1e-3, "wave embedding mismatch: {d}");
    }
}
