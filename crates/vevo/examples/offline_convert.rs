//! True offline (whole-utterance, single-pass) conversion via
//! `VevoEngine::inference_fm` directly — bypasses `VevoStream` entirely.
//!
//! This is the golden-parity-verified path (`cargo test -p vevo`); the
//! TUI's `--wav ... --out ...` flow does **not** use it (it feeds the
//! file through the same 320 ms/0.5 s-context streaming driver used for
//! live mic — see #77/field reports on why that's a lower-quality,
//! lower-latency-budget path). Use this example for a demo that
//! reflects the port's actual ceiling.
//!
//! ```sh
//! cargo run --release -p vevo --features cuda --example offline_convert -- \
//!     <src.wav> <ref.wav> <out.wav> [steps]
//! ```

use candle_core::Device;
use vevo::pipeline::{resample, VevoEngine};

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

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        eprintln!("usage: offline_convert <src.wav> <ref.wav> <out.wav> [steps]");
        std::process::exit(2);
    }
    let steps: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(32);
    let dev = Device::cuda_if_available(0).unwrap();
    let ckpt = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ckpt");
    let engine = VevoEngine::load(&ckpt, &dev).unwrap();

    let (src, sr1) = read_wav(&a[0]);
    let (rf, sr2) = read_wav(&a[1]);
    let src24 = resample(&src, sr1 as usize, 24_000);
    let ref24 = resample(&rf, sr2 as usize, 24_000);

    let t0 = std::time::Instant::now();
    let out = engine.inference_fm(&src24, &ref24, steps, None).unwrap();
    eprintln!(
        "converted in {:.2}s ({:.1}s src, RTF {:.3})",
        t0.elapsed().as_secs_f32(),
        src24.len() as f32 / 24_000.0,
        t0.elapsed().as_secs_f32() / (src24.len() as f32 / 24_000.0)
    );

    let out48 = resample(&out, 24_000, 48_000);
    let mut w = hound::WavWriter::create(
        &a[2],
        hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for s in &out48 {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
    }
    w.finalize().unwrap();
    println!("wrote {} ({} samples)", a[2], out48.len());
}
