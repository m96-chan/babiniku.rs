//! `babiniku-fetch` (issue #65): downloads the official checkpoints
//! from Hugging Face and converts them to the fp32 safetensors the
//! engines load — pure Rust, no Python required, per the CLAUDE.md
//! tooling policy ("anything a user runs is Rust").
//!
//! Golden/fixture generators are NOT this tool's job: those stay
//! Python by design (they must run the official implementations).
//!
//! ```sh
//! babiniku-fetch seedvc [--ckpt-dir <dir>] [--yes]
//! ```
//!
//! Nested-checkpoint reading relies on the candle fork's recursive
//! pickle traversal (`read_pth_tensor_info` descends `{net: {cfm: …}}`
//! with dotted keys).

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut cmd: Option<String> = None;
    let mut ckpt_dir: Option<PathBuf> = None;
    let mut yes = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ckpt-dir" => ckpt_dir = Some(PathBuf::from(args.next().context("--ckpt-dir <dir>")?)),
            "--yes" | "-y" => yes = true,
            "--help" | "-h" => {
                println!("usage: babiniku-fetch <seedvc> [--ckpt-dir <dir>] [--yes]");
                return Ok(());
            }
            c if cmd.is_none() && !c.starts_with('-') => cmd = Some(c.to_string()),
            other => bail!("unknown argument {other:?} (try --help)"),
        }
    }
    let ckpt = match ckpt_dir {
        Some(d) => d,
        None => default_ckpt_dir()?,
    };
    std::fs::create_dir_all(&ckpt)
        .with_context(|| format!("cannot create {}", ckpt.display()))?;

    match cmd.as_deref() {
        Some("seedvc") => fetch_seedvc(&ckpt, yes),
        Some(other) => bail!("unknown engine {other:?} — supported: seedvc (meanvc/xvc: #65)"),
        None => bail!("usage: babiniku-fetch <seedvc> [--ckpt-dir <dir>] [--yes]"),
    }
}

/// Mirror of the demo's resolution: `./ckpt` in a checkout, else the
/// platform data directory.
fn default_ckpt_dir() -> Result<PathBuf> {
    let local = PathBuf::from("ckpt");
    if local.is_dir() {
        return Ok(local);
    }
    let base = if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
    };
    Ok(base.context("cannot determine a data directory")?.join("babiniku/ckpt"))
}

fn confirm_gpl(yes: bool) -> Result<()> {
    eprintln!("Seed-VC weights and the seedvc engine crate are GPL-3.0.");
    eprintln!("Downloading is fine for local use; distributing builds that");
    eprintln!("include them carries GPL obligations (see crates/seedvc).");
    if yes {
        return Ok(());
    }
    eprint!("continue? [y/N] ");
    std::io::stderr().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    if !matches!(line.trim(), "y" | "Y" | "yes") {
        bail!("aborted");
    }
    Ok(())
}

fn hf_file(repo: &str, file: &str) -> Result<PathBuf> {
    eprintln!("fetching {repo} :: {file} …");
    let api = hf_hub::api::sync::Api::new()?;
    Ok(api.model(repo.to_string()).get(file)?)
}

/// Reads a `.pth`/`.bin`/`.pt` into name → fp32 tensor, keeping only
/// keys under `prefix` (stripped) when given.
fn read_pth(path: &Path, prefix: Option<&str>) -> Result<HashMap<String, Tensor>> {
    let all = candle_core::pickle::read_all(path)?;
    let mut out = HashMap::new();
    for (name, t) in all {
        let name = match prefix {
            Some(p) => match name.strip_prefix(p) {
                Some(rest) => rest.to_string(),
                None => continue,
            },
            None => name,
        };
        out.insert(name, t.to_dtype(DType::F32)?);
    }
    if out.is_empty() {
        bail!("no tensors under prefix {prefix:?} in {}", path.display());
    }
    Ok(out)
}

/// Folds `weight_g`/`weight_v` pairs into plain `weight` tensors
/// (`torch.nn.utils.remove_weight_norm`): `w = v · g / ‖v‖₂` with the
/// norm over all dims except 0.
fn fold_weight_norm(sd: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    let mut done: Vec<String> = Vec::new();
    for (k, g) in sd.iter() {
        let Some(base) = k.strip_suffix("weight_g") else {
            continue;
        };
        let v = sd
            .get(&format!("{base}weight_v"))
            .with_context(|| format!("missing weight_v for {k}"))?;
        let dims: Vec<usize> = (1..v.dims().len()).collect();
        let mut norm = v.sqr()?;
        for d in dims {
            norm = norm.sum_keepdim(d)?;
        }
        let norm = norm.sqrt()?;
        let w = v.broadcast_mul(&g.broadcast_div(&norm)?)?;
        out.insert(format!("{base}weight"), w);
        done.push(k.clone());
        done.push(format!("{base}weight_v"));
    }
    for (k, t) in sd {
        if !done.contains(&k) {
            out.insert(k, t);
        }
    }
    Ok(out)
}

fn save(map: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    candle_core::safetensors::save(map, path)?;
    eprintln!("wrote {} ({} tensors)", path.display(), map.len());
    Ok(())
}

fn fetch_seedvc(ckpt: &Path, yes: bool) -> Result<()> {
    confirm_gpl(yes)?;
    let dev = Device::Cpu;
    let _ = dev;

    // Main checkpoint: DiT (cfm) + length regulator. The checkpoint's
    // style_encoder is dead weight; the real speaker encoder is the
    // standalone funasr CAM++ (see docs/seedvc.md).
    let main = hf_file(
        "Plachta/Seed-VC",
        "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
    )?;
    save(&read_pth(&main, Some("net.cfm."))?, &ckpt.join("seedvc_dit.safetensors"))?;
    save(
        &read_pth(&main, Some("net.length_regulator."))?,
        &ckpt.join("seedvc_regulator.safetensors"),
    )?;

    // CAM++ speaker encoder (flat state dict).
    let camp = hf_file("funasr/campplus", "campplus_cn_common.bin")?;
    save(&read_pth(&camp, None)?, &ckpt.join("seedvc_campplus.safetensors"))?;

    // BigVGAN v2 vocoder (weight norm folded like remove_weight_norm()).
    let bv = hf_file("nvidia/bigvgan_v2_22khz_80band_256x", "bigvgan_generator.pt")?;
    let bv_sd = read_pth(&bv, Some("generator."))
        .or_else(|_| read_pth(&bv, None))?;
    save(&fold_weight_norm(bv_sd)?, &ckpt.join("seedvc_bigvgan.safetensors"))?;

    // Whisper-small encoder (already safetensors upstream; subset).
    let wh = hf_file("openai/whisper-small", "model.safetensors")?;
    let all = candle_core::safetensors::load(&wh, &Device::Cpu)?;
    let enc: HashMap<String, Tensor> = all
        .into_iter()
        .filter(|(k, _)| k.starts_with("model.encoder."))
        .map(|(k, t)| Ok((k, t.to_dtype(DType::F32)?)))
        .collect::<Result<_>>()?;
    save(&enc, &ckpt.join("seedvc_whisper.safetensors"))?;

    eprintln!("seedvc checkpoints ready under {}", ckpt.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_norm_fold_matches_torch_semantics() {
        // w = v * g / ||v||_2 with the norm over dims 1.. (torch
        // remove_weight_norm); checked against a hand computation.
        let dev = Device::Cpu;
        let v = Tensor::from_vec(vec![3f32, 4.0, 0.0, 5.0], (2, 2, 1), &dev).unwrap();
        let g = Tensor::from_vec(vec![10f32, 1.0], (2, 1, 1), &dev).unwrap();
        let mut sd = HashMap::new();
        sd.insert("conv.weight_v".to_string(), v);
        sd.insert("conv.weight_g".to_string(), g);
        sd.insert("conv.bias".to_string(), Tensor::zeros(2, DType::F32, &dev).unwrap());
        let out = fold_weight_norm(sd).unwrap();
        assert!(out.contains_key("conv.weight"));
        assert!(out.contains_key("conv.bias"));
        assert!(!out.contains_key("conv.weight_v"));
        let w: Vec<f32> = out["conv.weight"].flatten_all().unwrap().to_vec1().unwrap();
        // Row 0: ||(3,4)|| = 5, g=10 → (6, 8). Row 1: ||(0,5)|| = 5, g=1 → (0, 1).
        let want = [6.0f32, 8.0, 0.0, 1.0];
        for (a, b) in w.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn default_ckpt_dir_prefers_checkout() {
        // Run from the workspace root in CI/dev: ./ckpt may or may not
        // exist, but the function must return SOME sensible path.
        let d = default_ckpt_dir().unwrap();
        assert!(d.to_string_lossy().contains("ckpt"));
    }
}
