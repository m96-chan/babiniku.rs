//! Pairwise CAM++ similarity across sliding windows of a single
//! recording, to distinguish "genuinely fluctuating between voices"
//! from "one consistent voice that doesn't match some external
//! reference".
//!
//! ```sh
//! cargo run --release -p cosyvoice --features cuda --example pairwise_similarity -- \
//!     <recording.wav> [window_s] [hop_s]
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
    let window_s: f32 = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(3.0);
    let hop_s: f32 = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(3.0);

    let dev = Device::cuda_if_available(0).unwrap();
    let eng = CosyVoiceEngine::load("ckpt", &dev).unwrap();
    let rec = read16(&a[0]);

    let win = (window_s * 16_000.0) as usize;
    let hop = (hop_s * 16_000.0) as usize;
    let mut embs = Vec::new();
    let mut times = Vec::new();
    let mut i = 0;
    while i + win <= rec.len() {
        let seg = &rec[i..i + win];
        if rms(seg) > 5e-4 {
            embs.push(eng.embed_for_debug(seg).unwrap());
            times.push(i as f32 / 16_000.0);
        }
        i += hop;
    }
    println!("{} usable windows", embs.len());
    // similarity of each window to the FIRST usable window
    for (t, e) in times.iter().zip(&embs) {
        println!("t={t:5.1}s sim_to_first={:.3}", cos(&embs[0], e));
    }
    // overall stats: mean pairwise similarity
    let mut sims = Vec::new();
    for i in 0..embs.len() {
        for j in (i + 1)..embs.len() {
            sims.push(cos(&embs[i], &embs[j]));
        }
    }
    let mean = sims.iter().sum::<f32>() / sims.len().max(1) as f32;
    let min = sims.iter().cloned().fold(1.0f32, f32::min);
    let max = sims.iter().cloned().fold(-1.0f32, f32::max);
    println!(
        "pairwise: mean={mean:.3} min={min:.3} max={max:.3} (n={})",
        sims.len()
    );

    // full matrix, for eyeballing block/cluster structure
    print!("      ");
    for t in &times {
        print!("{t:5.0} ");
    }
    println!();
    for i in 0..embs.len() {
        print!("t={:4.0} ", times[i]);
        for j in 0..embs.len() {
            print!("{:.2}  ", cos(&embs[i], &embs[j]));
        }
        println!();
    }
}
