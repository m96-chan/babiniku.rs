use candle_core::Device;
use meanvc2::backends::{Vocos, VocosConfig};
use meanvc2::encoders::Vocoder;
use meanvc2::v1::MelV1;

fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let fx = candle_core::safetensors::load("ckpt/copysyn_fixture.safetensors", &dev)?;

    // 1) mel front-end parity
    let mut r = hound::WavReader::open("ckpt/test.wav")?;
    let wav: Vec<f32> = r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    let ours = MelV1::new().compute(&wav, &dev)?;
    let theirs = &fx["mel_raw"];
    let n = ours.dim(0)?.min(theirs.dim(0)?);
    let d = (ours.narrow(0,0,n)? - theirs.narrow(0,0,n)?)?.abs()?;
    println!("mel diff: max {:.5} mean {:.6}", d.max_all()?.to_scalar::<f32>()?, d.mean_all()?.to_scalar::<f32>()?);

    // 2) vocoder parity on the official mel
    let vocos = Vocos::load(VocosConfig::official_meanvc1(), "ckpt/vocos.safetensors", &dev)?;
    let y = vocos.synthesize(&fx["mel01"])?;
    let yref: Vec<f32> = fx["wav_ref"].to_vec1()?;
    let m = y.len().min(yref.len());
    // official istft output includes padding trim differences; align by trimming both
    let diff: f32 = y[..m].iter().zip(&yref[..m]).map(|(a,b)| (a-b).abs()).fold(0f32, f32::max);
    let rms_ref = (yref[..m].iter().map(|s| s*s).sum::<f32>()/m as f32).sqrt();
    println!("vocos: ours {} ref {} samples, max diff {diff:.5} (ref rms {rms_ref:.4})", y.len(), yref.len());

    let spec = hound::WavSpec { channels: 1, sample_rate: 16000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create("ckpt/copysyn_rust.wav", spec)?;
    for s in &y { w.write_sample((s.clamp(-1.0,1.0) * 32767.0) as i16)?; }
    w.finalize()?;
    println!("wrote ckpt/copysyn_rust.wav");
    Ok(())
}
