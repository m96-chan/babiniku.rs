//! Quantitative speaker-similarity check, Vevo-Timbre vs Seed-VC on the
//! same source/reference pair: field report (2026-07) that Vevo's
//! reference-following ("リファレンス力") feels weaker than Seed-VC's.
//! Architecturally plausible — Vevo has **no discriminative speaker
//! embedding at all** (unlike Seed-VC's CAM++), it relies entirely on
//! prompt-prefix in-context conditioning — but "plausible" isn't
//! "measured." This borrows CosyVoice2's CAM++ (Apache-2.0, independent
//! of both engines under test) purely as a speaker-similarity yardstick,
//! same technique as `cosyvoice::examples::speaker_sim_probe` (see
//! docs/cosyvoice.md's own speaker-similarity field investigation).
//!
//! ```sh
//! cargo run --release -p babiniku --features cuda,seedvc \
//!     --example vevo_vs_seedvc_similarity -- <source.wav> <reference.wav>
//! ```

use candle_core::{Device, Tensor};
use cosyvoice::CosyVoiceEngine;
use seedvc::pipeline::{resample as seedvc_resample, SeedVcEngine};
use vevo::pipeline::{resample as vevo_resample, VevoEngine};

fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let mut r = hound::WavReader::open(path).unwrap();
    let spec = r.spec();
    let s: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .step_by(spec.channels as usize)
                .map(|v| v.unwrap() as f32 / scale)
                .collect()
        }
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .step_by(spec.channels as usize)
            .map(|v| v.unwrap())
            .collect(),
    };
    (s, spec.sample_rate)
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        eprintln!("usage: vevo_vs_seedvc_similarity <source.wav> <reference.wav>");
        std::process::exit(2);
    }
    let dev = Device::cuda_if_available(0).unwrap();
    let ckpt = "ckpt";

    let (src, src_sr) = read_wav(&a[0]);
    let (rf, ref_sr) = read_wav(&a[1]);

    // ---- Vevo-Timbre (24 kHz native, 32-step offline) ----
    let vevo = VevoEngine::load(ckpt, &dev).unwrap();
    let src24 = vevo_resample(&src, src_sr as usize, 24_000);
    let ref24 = vevo_resample(&rf, ref_sr as usize, 24_000);
    let vevo_out = vevo.inference_fm(&src24, &ref24, 32, None).unwrap();
    let vevo_out16 = vevo_resample(&vevo_out, 24_000, 16_000);

    // ---- Seed-VC (22.05 kHz native, 10-step offline, cfg 0.7) ----
    let seedvc = SeedVcEngine::load(ckpt, &dev).unwrap();
    let src22 = seedvc_resample(&src, src_sr as usize, 22_050);
    let ref22 = seedvc_resample(&rf, ref_sr as usize, 22_050);
    let mel_len = |n: usize| (n.saturating_sub(256)) / 256 + 1;
    let t = mel_len(ref22.len()) + mel_len(src22.len());
    let noise = Tensor::randn(0f32, 1f32, (1, 80, t), &dev).unwrap();
    let seedvc_out = seedvc.convert_offline(&src22, &ref22, 10, 0.7, &noise).unwrap();
    let seedvc_out16 = seedvc_resample(&seedvc_out, 22_050, 16_000);

    // ---- CAM++ (CosyVoice2's, used purely as an independent yardstick) ----
    let judge = CosyVoiceEngine::load(ckpt, &dev).unwrap();
    let src16 = vevo_resample(&src, src_sr as usize, 16_000);
    let ref16 = vevo_resample(&rf, ref_sr as usize, 16_000);
    let embed = |audio: &[f32]| judge.embed_for_debug(audio).unwrap();
    let (src_e, ref_e) = (embed(&src16), embed(&ref16));
    let (vevo_e, seedvc_e) = (embed(&vevo_out16), embed(&seedvc_out16));

    println!("src vs ref (how different are the two speakers): {:.4}", cos(&src_e, &ref_e));
    println!("vevo   out vs ref (should be HIGH if conversion worked): {:.4}", cos(&vevo_e, &ref_e));
    println!("seedvc out vs ref (should be HIGH if conversion worked): {:.4}", cos(&seedvc_e, &ref_e));
    println!("vevo   out vs src (should be LOW — did it leave the source's identity behind?): {:.4}", cos(&vevo_e, &src_e));
    println!("seedvc out vs src (should be LOW — did it leave the source's identity behind?): {:.4}", cos(&seedvc_e, &src_e));
}
