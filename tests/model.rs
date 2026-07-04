use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use meanvc2::{meanflow, MeanVc2, MeanVc2Config, StreamingConverter};

fn tiny_config() -> MeanVc2Config {
    let mut cfg = MeanVc2Config::default();
    cfg.decoder.n_mels = 20;
    cfg.mel.n_mels = 20;
    cfg.decoder.hidden_dim = 64;
    cfg.decoder.time_embed_dim = 64;
    cfg.decoder.bnf_dim = 32;
    cfg.utte.hidden_dim = 32;
    cfg.utte.bnf_dim = 24;
    cfg.utte.num_tokens = 8;
    cfg.decoder.speaker_dim = 16;
    cfg.utte.speaker_dim = 16;
    cfg
}

fn tiny_model(cfg: &MeanVc2Config) -> MeanVc2 {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    MeanVc2::new(cfg.clone(), vb).unwrap()
}

#[test]
fn offline_conversion_shapes() {
    let cfg = tiny_config();
    let model = tiny_model(&cfg);
    let dev = Device::Cpu;

    // 6 chunks of 4 frames.
    let time = 6 * cfg.decoder.chunk_frames;
    let bnf = Tensor::randn(0f32, 1f32, (1, time, cfg.utte.bnf_dim), &dev).unwrap();
    let speaker = Tensor::randn(0f32, 1f32, (1, cfg.utte.speaker_dim), &dev).unwrap();

    let mel = model.convert(&bnf, &speaker).unwrap();
    assert_eq!(mel.dims(), &[1, time, cfg.decoder.n_mels]);
    let flat: Vec<f32> = mel.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|x| x.is_finite()));
}

#[test]
fn mean_flow_loss_is_finite_and_scalar() {
    let cfg = tiny_config();
    let model = tiny_model(&cfg);
    let dev = Device::Cpu;

    let time = 4 * cfg.decoder.chunk_frames;
    let x = Tensor::randn(0f32, 1f32, (2, time, cfg.decoder.n_mels), &dev).unwrap();
    let bnf = Tensor::randn(0f32, 1f32, (2, time, cfg.utte.bnf_dim), &dev).unwrap();
    let speaker = Tensor::randn(0f32, 1f32, (2, cfg.utte.speaker_dim), &dev).unwrap();

    let cond = model.timbre_aware_bnf(&bnf, &speaker).unwrap();
    let masks = model.decoder.frc_masks(time, &dev).unwrap();
    let rt = meanflow::sample_rt(2, 0.75, &dev).unwrap();
    let out =
        meanflow::mean_flow_loss(&model, &x, &cond, &speaker, Some(&masks), &rt, 1e-2).unwrap();

    assert_eq!(out.loss.dims(), &[] as &[usize]);
    let loss: f32 = out.loss.to_scalar().unwrap();
    assert!(loss.is_finite());
    assert_eq!(out.prediction.dims(), x.dims());
}

#[test]
fn rt_sample_orders_endpoints() {
    let dev = Device::Cpu;
    let rt = meanflow::sample_rt(64, 0.5, &dev).unwrap();
    let r: Vec<f32> = rt.r.to_vec1().unwrap();
    let t: Vec<f32> = rt.t.to_vec1().unwrap();
    assert!(r.iter().zip(&t).all(|(r, t)| r <= t));
}

#[test]
fn streaming_emits_every_chunk_with_lookahead() {
    let cfg = tiny_config();
    let model = tiny_model(&cfg);
    let dev = Device::Cpu;

    let mut conv = StreamingConverter::new(
        &model,
        &Tensor::randn(0f32, 1f32, (cfg.utte.speaker_dim,), &dev).unwrap(),
    )
    .unwrap();
    assert_eq!(conv.lookahead_chunks(), 1);

    let num_chunks = 10;
    let mut emitted = 0;
    for i in 0..num_chunks {
        let chunk = Tensor::randn(
            0f32,
            1f32,
            (1, cfg.decoder.chunk_frames, cfg.utte.bnf_dim),
            &dev,
        )
        .unwrap();
        let ready = conv.push(&chunk).unwrap();
        for mel in &ready {
            assert_eq!(mel.dims(), &[1, cfg.decoder.chunk_frames, cfg.decoder.n_mels]);
        }
        emitted += ready.len();
        // With a 1-chunk look-ahead, the first chunk is emitted on push #2.
        if i == 0 {
            assert_eq!(emitted, 0);
        }
    }
    emitted += conv.finish().unwrap().len();
    assert_eq!(emitted, num_chunks);
}
