use candle_core::Device;
use meanvc2::backends::{FastU2pp, FastU2ppConfig};
fn main() -> anyhow::Result<()> {
    let dev = Device::Cpu;
    let fx = candle_core::safetensors::load("ckpt/asr_chunk0_fixture.safetensors", &dev)?;
    let asr = FastU2pp::load(FastU2ppConfig::official_meanvc1(),
        "ckpt/fastu2pp.safetensors", &dev)?;
    let emb = asr.subsample(&fx["fb23"])?;
    let de = (&emb - &fx["embed_ref"])?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("embed diff: {de:.5} (ref |max| {:.1})", fx["embed_ref"].abs()?.max_all()?.to_scalar::<f32>()?);
    let pe = meanvc2::backends::debug_sinusoidal_pe(5, 256, &dev)?;
    let dpe = (&pe - &fx["pos"])?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("pos-emb diff: {dpe:.6}");
    let l0 = asr.debug_layer0(&fx["x_embed"], &fx["pos"])?;
    let dl = (&l0 - &fx["layer0_ref"])?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("layer0 diff: {dl:.5} (ref |max| {:.2})", fx["layer0_ref"].abs()?.max_all()?.to_scalar::<f32>()?);
    let bn = asr.forward(&fx["fb23"])?;
    let d = (&bn - &fx["bn0_ref"])?.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("chunk0 bn diff: {d:.5} (ref |max| {:.3})", fx["bn0_ref"].abs()?.max_all()?.to_scalar::<f32>()?);
    Ok(())
}
