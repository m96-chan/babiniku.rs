import os, torch
os.environ["HF_HUB_CACHE"] = os.path.abspath("checkpoints/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
from safetensors.torch import load_file, save_file
from modules.bigvgan.bigvgan import BigVGAN

torch.set_grad_enabled(False)
model = BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False)
model.remove_weight_norm()
model.eval()

fx = load_file("/home/m96-chan/project/m96-chan/babiniku.rs/ckpt/seedvc_e2e_fixture.safetensors")
mel = fx["vc_mel"]

# vc_mel + conv_pre (cheap first-stage diagnostic) + final wave; the
# intermediate stages were only needed once for bisecting and are too
# large (~150 MB) to keep in the checked-in fixture.
out = {"vc_mel": mel, "conv_pre": model.conv_pre(mel)}
out["wave"] = model(mel)

diff = (out["wave"].squeeze(1) - fx["vc_wave"]).abs().max().item()
print("official standalone vs fixture vc_wave max abs diff:", diff)

save_file({k: v.contiguous() for k, v in out.items()},
          "/home/m96-chan/project/m96-chan/babiniku.rs/ckpt/seedvc_bigvgan_fixture.safetensors")
print("saved")
