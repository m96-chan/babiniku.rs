use candle_core::Device;
use meanvc2::backends::{FastU2pp, FastU2ppConfig};
use meanvc2::v1::{interpolate_linear, KaldiFbank};
fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let fx = candle_core::safetensors::load("ckpt/pipeline_ref.safetensors", &dev)?;
    let mut r = hound::WavReader::open("ckpt/test.wav")?;
    let wav: Vec<f32> = r.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    let fbank = KaldiFbank::new().compute(&wav, &dev)?.unsqueeze(0)?;
    println!("fbank: {:?}", fbank.dims());
    let asr = FastU2pp::load(FastU2ppConfig::official_meanvc1(),
        "ckpt/fastu2pp.safetensors", &dev)?;
    let bn = interpolate_linear(&asr.forward(&fbank)?, 4)?;
    let bref = &fx["bn_ref"];
    println!("bn ours {:?} ref {:?}", bn.dims(), bref.dims());
    let n = bn.dim(1)?.min(bref.dim(1)?);
    let d = (bn.narrow(1,0,n)? - bref.narrow(1,0,n)?)?.abs()?;
    let scale = bref.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("bn diff: max {:.4} mean {:.5} (ref |max| {scale:.3})",
        d.max_all()?.to_scalar::<f32>()?, d.mean_all()?.to_scalar::<f32>()?);
    // per-frame profile
    let pf = d.max(2)?.squeeze(0)?.to_vec1::<f32>()?;
    println!("frame diffs: 0..5 {:?} mid {:?}", &pf[..5], &pf[n/2..n/2+5]);
    Ok(())
}
