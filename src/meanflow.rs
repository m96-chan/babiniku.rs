//! Mean flows: training objective and 1-NFE sampling.
//!
//! Conditional flow matching builds the path `z_t = (1 - t) x + t ε` with
//! conditional velocity `v_t = ε - x`. Mean flows (Geng et al., 2025)
//! instead regress the *average* velocity over an interval `[r, t]`:
//!
//! ```text
//! u(z_t, r, t) = 1 / (t - r) * ∫_r^t v(z_τ, τ) dτ
//! ```
//!
//! Differentiating yields the mean-flows identity used as the training
//! target (Eq. 3 of the paper):
//!
//! ```text
//! u = v_t - (t - r) * d/dt u(z_t, r, t)
//! ```
//!
//! where the total derivative expands to the Jacobian–vector product
//! `dz/dt ∂_z u + ∂_t u` with tangent `(v_t, 0, 1)`. The loss is
//! `|| f_θ(z_t, r, t) - sg(u_tgt) ||²` (Eq. 4); at `r == t` it reduces to
//! standard conditional flow matching.
//!
//! At inference the clean sample is recovered with a **single** function
//! evaluation: `z_0 = z_1 - f_θ(z_1, 0, 1)` with `z_1 = ε ~ N(0, I)`.
//!
//! ### Note on the JVP
//!
//! `candle` has no forward-mode autodiff, so [`mean_flow_loss`] approximates
//! the JVP with a central-free forward finite difference along the tangent
//! `(v_t, 0, 1)`. This adds one extra forward pass and a step-size
//! hyper-parameter (`delta`, default `1e-2`) but keeps the whole objective
//! expressible in stable candle ops.

use candle_core::Tensor;

use crate::model::MeanVc2;
use crate::Result;

/// Sampled interval endpoints for mean-flows training, shape `[batch]` each,
/// with `r <= t`.
#[derive(Debug, Clone)]
pub struct RtSample {
    pub r: Tensor,
    pub t: Tensor,
}

/// Samples `(r, t)` pairs uniformly with `r <= t`.
///
/// With probability `cfm_ratio` the pair is collapsed to `r = t`, in which
/// case the objective reduces to standard conditional flow matching — mixing
/// both regimes stabilizes training (Geng et al., 2025 use ~75%).
pub fn sample_rt(
    batch: usize,
    cfm_ratio: f64,
    device: &candle_core::Device,
) -> Result<RtSample> {
    let a = Tensor::rand(0f32, 1f32, (batch,), device)?;
    let b = Tensor::rand(0f32, 1f32, (batch,), device)?;
    let t = a.maximum(&b)?;
    let r = a.minimum(&b)?;
    let collapse = Tensor::rand(0f32, 1f32, (batch,), device)?
        .lt(cfm_ratio as f32)?;
    let r = collapse.where_cond(&t, &r)?;
    Ok(RtSample { r, t })
}

/// Everything produced by one training step of the mean-flows objective.
#[derive(Debug)]
pub struct MeanFlowLoss {
    /// Scalar MSE loss `|| f_θ(z_t, r, t) - sg(u_tgt) ||²`.
    pub loss: Tensor,
    /// The network prediction `f_θ(z_t, r, t)`.
    pub prediction: Tensor,
    /// The (detached) regression target `u_tgt`.
    pub target: Tensor,
}

/// Computes the mean-flows training loss (Eq. 4 of the paper).
///
/// * `x`: clean mel-spectrogram `[batch, time, n_mels]`
/// * `cond_bnf`: timbre-aware BNFs `[batch, time, bnf_dim]`
/// * `speaker`: `[batch, speaker_dim]`
/// * `masks`: per-layer FRC masks (chunked training, Sec. 3.2) or `None`
/// * `rt`: interval endpoints from [`sample_rt`]
/// * `delta`: finite-difference step for the JVP approximation
pub fn mean_flow_loss(
    model: &MeanVc2,
    x: &Tensor,
    cond_bnf: &Tensor,
    speaker: &Tensor,
    masks: Option<&[Tensor]>,
    rt: &RtSample,
    delta: f64,
) -> Result<MeanFlowLoss> {
    let (b, _, _) = x.dims3()?;
    let noise = x.randn_like(0.0, 1.0)?;
    let t3 = rt.t.reshape((b, 1, 1))?;

    // z_t = (1 - t) x + t ε, v_t = ε - x.
    let z_t = x
        .broadcast_mul(&(1.0 - &t3)?)?
        .broadcast_add(&noise.broadcast_mul(&t3)?)?;
    let v_t = (&noise - x)?;

    let u = model.forward(&z_t, cond_bnf, speaker, &rt.r, &rt.t, masks)?;

    // JVP along the tangent (dz, dr, dt) = (v_t, 0, 1), forward differences:
    // (f(z + δ v, r, t + δ) - f(z, r, t)) / δ.
    let z_shift = (&z_t + &(v_t.clone() * delta)?)?;
    let t_shift = (&rt.t + delta)?;
    let u_shift = model.forward(&z_shift, cond_bnf, speaker, &rt.r, &t_shift, masks)?;
    let jvp = ((&u_shift - &u)? / delta)?;

    // u_tgt = v_t - (t - r) * jvp, with stop-gradient.
    let span = (&rt.t - &rt.r)?.reshape((b, 1, 1))?;
    let target = (&v_t - &jvp.broadcast_mul(&span)?)?.detach();

    let loss = (&u - &target)?.sqr()?.mean_all()?;
    Ok(MeanFlowLoss {
        loss,
        prediction: u,
        target,
    })
}

/// One-step (1-NFE) sampling: `z_0 = z_1 - f_θ(z_1, 0, 1)`.
///
/// * `noise`: `z_1 ~ N(0, I)`, `[batch, time, n_mels]`
/// * `cond_bnf`: timbre-aware BNFs `[batch, time, bnf_dim]`
///
/// Returns the generated mel-spectrogram `[batch, time, n_mels]`.
pub fn sample_1nfe(
    model: &MeanVc2,
    noise: &Tensor,
    cond_bnf: &Tensor,
    speaker: &Tensor,
    masks: Option<&[Tensor]>,
) -> Result<Tensor> {
    let b = noise.dim(0)?;
    let device = noise.device();
    let r = Tensor::zeros((b,), candle_core::DType::F32, device)?;
    let t = Tensor::ones((b,), candle_core::DType::F32, device)?;
    let u = model.forward(noise, cond_bnf, speaker, &r, &t, masks)?;
    Ok((noise - &u)?)
}
