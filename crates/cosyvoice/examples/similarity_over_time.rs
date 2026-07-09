//! Sliding-window CAM++ speaker-similarity trace: segments a recording
//! and reports cosine similarity to a fixed reference per window, to
//! find where a live session's output drifts away from the target
//! speaker (field-debugging aid).
//!
//! ```sh
//! cargo run --release -p cosyvoice --features cuda --example similarity_over_time -- \
//!     <recording.wav> <reference.wav> [window_s] [hop_s]
//! ```
use candle_core::Device;
use cosyvoice::CosyVoiceEngine;
use vc_core::profile::resample_analysis;

fn read16(p: &str) -> Vec<f32> {
    let mut r = hound::WavReader::open(p).unwrap();
    let spec = r.spec();
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let sc = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .step_by(spec.channels as usize)
                .map(|v| v.unwrap() as f32 / sc)
                .collect()
        }
        hound::SampleFormat::Float => r
            .samples::<f32>()
            .step_by(spec.channels as usize)
            .map(|v| v.unwrap())
            .collect(),
    };
    if spec.sample_rate == 16_000 {
        audio
    } else {
        resample_analysis(&audio, spec.sample_rate as usize, 16_000)
    }
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|s| s * s).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let window_s: f32 = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(3.0);
    let hop_s: f32 = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(1.5);

    let dev = Device::cuda_if_available(0).unwrap();
    let eng = CosyVoiceEngine::load("ckpt", &dev).unwrap();

    let rec = read16(&a[0]);
    let reference = read16(&a[1]);
    let ref_emb = eng.embed_for_debug(&reference).unwrap();

    let win = (window_s * 16_000.0) as usize;
    let hop = (hop_s * 16_000.0) as usize;
    let mut i = 0;
    while i + win <= rec.len() {
        let seg = &rec[i..i + win];
        let level = rms(seg);
        if level > 5e-4 {
            let emb = eng.embed_for_debug(seg).unwrap();
            let sim = cos(&emb, &ref_emb);
            let flag = if sim < 0.4 { " <-- LOW" } else { "" };
            println!(
                "t={:5.1}s rms={:6.1}dB sim_to_ref={:.3}{}",
                i as f32 / 16_000.0,
                20.0 * level.log10(),
                sim,
                flag
            );
        }
        i += hop;
    }
}
