use candle_core::Device;
use meanvc2::v1::{KvCache, MeanVc1, MeanVc1Config};
fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let fx = candle_core::safetensors::load("ckpt/dit_stream_fixture.safetensors", &dev)?;
    let model = MeanVc1::load(MeanVc1Config::default(), "ckpt/model_200ms.safetensors", &dev)?;
    let timbre = model.timbre_cond(&fx["bn"], &fx["prompts"], &fx["spks"])?;
    let cs = 20usize;
    let mut kv = KvCache::default();
    let mut prev: Option<candle_core::Tensor> = None;
    for q in 0..8 {
        let noise = &fx[&format!("n{q}")];
        let u = model.forward_stream(noise, &timbre.narrow(1, q*cs, cs)?, &fx["spks"],
            prev.as_ref(), q*cs, &mut kv)?;
        let d = (&u - &fx[&format!("u{q}")])?.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("chunk {q}: u diff {d:.6}");
        prev = Some((noise - u)?);
    }
    Ok(())
}
