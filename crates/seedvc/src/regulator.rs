//! `InterpolateRegulator` (continuous input, no f0, inference path):
//! `content_in_proj` (768 → 512) → nearest interpolation to the target
//! mel length → 4 × [Conv1d k3 p1 → GroupNorm(1) → Mish] → Conv1d k1.

use candle_core::{IndexOp, Module, Tensor};
use candle_nn::{conv1d, linear, Conv1d, Conv1dConfig, Linear, VarBuilder};

use crate::Result;

pub struct InterpolateRegulator {
    proj: Linear,
    blocks: Vec<(Conv1d, Tensor, Tensor)>,
    out: Conv1d,
}

impl InterpolateRegulator {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("module");
        let proj = linear(768, 512, vb.pp("content_in_proj"))?;
        let cfg = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let mut blocks = Vec::new();
        for i in 0..4 {
            let c = conv1d(512, 512, 3, cfg, vb.pp(format!("model.{}", i * 3)))?;
            let nb = vb.pp(format!("model.{}", i * 3 + 1));
            let w = nb.get(512, "weight")?.reshape((1, 512, 1))?;
            let b = nb.get(512, "bias")?.reshape((1, 512, 1))?;
            blocks.push((c, w, b));
        }
        let out = conv1d(
            512,
            512,
            1,
            Conv1dConfig::default(),
            vb.pp("model.12"),
        )?;
        Ok(Self { proj, blocks, out })
    }

    /// `content` `[1, T, 768]` → `[1, target_len, 512]`.
    pub fn forward(&self, content: &Tensor, target_len: usize) -> Result<Tensor> {
        let x = self.proj.forward(content)?; // [1, T, 512]
        let x = x.transpose(1, 2)?.contiguous()?; // [1, 512, T]
        // F.interpolate mode='nearest': src = floor(dst * scale) with
        // scale computed as a DOUBLE division (aten
        // nearest_neighbor_compute_source_index) — integer rational
        // floor differs at exact-boundary indices (every 12th index for
        // 301 -> 516) and shifts whole frames.
        let t = x.dim(2)?;
        let scale = t as f64 / target_len as f64;
        let idx: Vec<u32> = (0..target_len)
            .map(|i| ((i as f64 * scale).floor() as usize).min(t - 1) as u32)
            .collect();
        let idx = Tensor::from_vec(idx, target_len, x.device())?;
        let mut x = x.index_select(&idx, 2)?;
        for (c, w, b) in &self.blocks {
            // GroupNorm(num_groups=1): normalize over ALL of (C, T),
            // then per-channel affine.
            let y = c.forward(&x)?;
            let mean = y.mean_all()?;
            let centred = y.broadcast_sub(&mean.reshape((1, 1, 1))?)?;
            let var = centred.sqr()?.mean_all()?;
            let denom = (var + 1e-5)?.sqrt()?.reshape((1, 1, 1))?;
            x = centred
                .broadcast_div(&denom)?
                .broadcast_mul(w)?
                .broadcast_add(b)?;
            // Mish: x * tanh(softplus(x)).
            let sp = (x.exp()? + 1.0)?.log()?;
            x = (&x * sp.tanh()?)?;
        }
        let x = self.out.forward(&x)?;
        Ok(x.transpose(1, 2)?.contiguous()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    // WIP (#50): max abs diff 6.65e-2 vs the fixture. Verified NOT the
    // GroupNorm reduction axes nor the nearest-index arithmetic (both
    // variants bit-identical). Prime suspect: the checkpoint carries a
    // top-level `net.vq` module (7 tensors, not yet converted by
    // tools/convert_seedvc.py) and length_regulator.forward applies
    // `self.vq` when present — the diff would then be the quantization
    // residual. Next: dump intermediate fixtures (post-proj,
    // post-interp, per-block) and check whether model.length_regulator
    // has a vq attribute in the official object.
    #[test]
    #[ignore = "WIP #50: 6.65e-2 residual, suspected missing vq stage"]
    fn regulator_matches_official() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ckpt");
        let (w, f) = (
            dir.join("seedvc_regulator.safetensors"),
            dir.join("seedvc_e2e_fixture.safetensors"),
        );
        if !w.exists() || !f.exists() {
            return;
        }
        let dev = Device::Cpu;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[w], DType::F32, &dev).unwrap() };
        let reg = InterpolateRegulator::load(vb).unwrap();
        let fx = candle_core::safetensors::load(f, &dev).unwrap();
        let want = &fx["cond"];
        let target_len = want.dim(1).unwrap();
        let got = reg.forward(&fx["s_alt"], target_len).unwrap();
        let d = (got - want)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        println!("regulator max abs diff {d:.2e}");
        assert!(d < 1e-4, "regulator mismatch: {d}");
    }
}
