use candle_core::{Device, Tensor};
use meanvc2::v1::{CacheLayout, MeanVc1, MeanVc1Config};
fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let fx = candle_core::safetensors::load("ckpt/dit_fixture.safetensors", &dev)?;
    let model = MeanVc1::load(MeanVc1Config::default(), "ckpt/model_200ms.safetensors", &dev)?;
    let timbre = model.timbre_cond(&fx["bn"], &fx["prompts"], &fx["spks"])?;
    let r = Tensor::zeros((1,), candle_core::DType::F32, &dev)?;
    let t = Tensor::ones((1,), candle_core::DType::F32, &dev)?;
    let u = model.forward(&fx["x"], &timbre, &fx["spks"], CacheLayout::None, &r, &t)?;
    let d = (&u - &fx["u_ref"])?.abs()?;
    println!("u diff: max {:.5} mean {:.6} (ref |max| {:.3})",
        d.max_all()?.to_scalar::<f32>()?, d.mean_all()?.to_scalar::<f32>()?,
        fx["u_ref"].abs()?.max_all()?.to_scalar::<f32>()?);
    Ok(())
}
