"""Per-project model setup.

config.json holds the defaults. A project can override any *slot* below in
settings["models"] (slot id -> value). An empty value means "use the default".

Model files can come from several places; the loader node is chosen per file:
  diffusion_models/            -> UNETLoader
  checkpoints/  (all-in-one)   -> CheckpointLoaderSimple   (text encoder / VAE may be "@checkpoint")
  unet/*.gguf                  -> UnetLoaderGGUF            (needs ComfyUI-GGUF)
  text_encoders/ (or clip/)    -> CLIPLoader
  text_encoders/*.gguf         -> CLIPLoaderGGUF            (needs ComfyUI-GGUF)
"""
from __future__ import annotations

import copy

FROM_CKPT = "@checkpoint"
NONE = "@none"

# id, group, label, kind, config path
#   kind: diffusion | text_encoder | vae | dtype | lora | int | float
SLOTS = [
    # Qwen Image 2.1
    ("image.model", "qwen21", "Diffusion model", "diffusion", ("qwen21", "model")),
    ("image.text_encoder", "qwen21", "Text encoder (Qwen3-VL-8B)", "text_encoder", ("qwen21", "text_encoder")),
    ("image.vae", "qwen21", "VAE", "vae", ("qwen21", "vae")),
    ("image.weight_dtype", "qwen21", "Weight dtype", "dtype", ("qwen21", "weight_dtype")),
    ("image.fast_lora", "qwen21", "Fast-mode LoRA (none for a distilled/Turbo model)", "lora", ("qwen21", "fast", "lora")),
    ("image.fast_steps", "qwen21", "Fast steps", "int", ("qwen21", "fast", "steps")),
    ("image.quality_steps", "qwen21", "Quality steps", "int", ("qwen21", "quality", "steps")),
    ("image.cfg", "qwen21", "CFG (quality)", "float", ("qwen21", "quality", "cfg")),
    # legacy Qwen pair
    ("qwen.t2i_model", "qwen", "Text→image model", "diffusion", ("qwen", "t2i_model")),
    ("qwen.edit_model", "qwen", "Edit model", "diffusion", ("qwen", "edit_model")),
    ("qwen.text_encoder", "qwen", "Text encoder (Qwen2.5-VL)", "text_encoder", ("qwen", "text_encoder")),
    ("qwen.vae", "qwen", "VAE", "vae", ("qwen", "vae")),
    ("qwen.weight_dtype", "qwen", "Weight dtype", "dtype", ("qwen", "weight_dtype")),
    # Wan
    ("wan.text_encoder", "wan", "Text encoder (umt5-xxl)", "text_encoder", ("wan", "text_encoder")),
    ("wan.vae", "wan", "VAE", "vae", ("wan", "vae")),
    ("wan.weight_dtype", "wan", "Weight dtype", "dtype", ("wan", "weight_dtype")),
    ("wan.i2v.high", "wan", "Image→video / first-last: high noise", "diffusion", ("wan", "i2v", "high")),
    ("wan.i2v.low", "wan", "Image→video / first-last: low noise", "diffusion", ("wan", "i2v", "low")),
    ("wan.i2v.fast_lora_high", "wan", "Image→video fast LoRA: high", "lora", ("wan", "i2v", "fast_lora_high")),
    ("wan.i2v.fast_lora_low", "wan", "Image→video fast LoRA: low", "lora", ("wan", "i2v", "fast_lora_low")),
    ("wan.t2v.high", "wan", "Text→video: high noise", "diffusion", ("wan", "t2v", "high")),
    ("wan.t2v.low", "wan", "Text→video: low noise", "diffusion", ("wan", "t2v", "low")),
    ("wan.t2v.fast_lora_high", "wan", "Text→video fast LoRA: high", "lora", ("wan", "t2v", "fast_lora_high")),
    ("wan.t2v.fast_lora_low", "wan", "Text→video fast LoRA: low", "lora", ("wan", "t2v", "fast_lora_low")),
    ("wan.vace.high", "wan", "VACE model (high noise, or the only one)", "diffusion", ("wan", "vace", "high")),
    ("wan.vace.low", "wan", "VACE low noise (none = single-stage)", "diffusion", ("wan", "vace", "low")),
    ("wan.vace.weight_dtype", "wan", "VACE weight dtype", "dtype", ("wan", "vace", "weight_dtype")),
    ("wan.vace.fast_lora_high", "wan", "VACE fast LoRA: high", "lora", ("wan", "vace", "fast_lora_high")),
    ("wan.vace.fast_lora_strength", "wan", "VACE fast LoRA strength", "float", ("wan", "vace", "fast_lora_strength")),
    ("wan.vace.fill", "wan", "VACE fill (0.5 grey, 1.0 white)", "float", ("wan", "vace", "fill")),
    ("wan.vace.color_match", "wan", "VACE colour match to context (0–1)", "float", ("wan", "vace", "color_match")),
    ("wan.vace.fast_lora_low", "wan", "VACE fast LoRA: low", "lora", ("wan", "vace", "fast_lora_low")),
    ("wan.inp.high", "wan_fun", "Fun InP (first/last frame): high noise", "diffusion", ("wan", "inp", "high")),
    ("wan.inp.low", "wan_fun", "Fun InP: low noise", "diffusion", ("wan", "inp", "low")),
    ("wan.inp.fast_lora_high", "wan_fun", "Fun InP fast LoRA: high", "lora", ("wan", "inp", "fast_lora_high")),
    ("wan.inp.fast_lora_low", "wan_fun", "Fun InP fast LoRA: low", "lora", ("wan", "inp", "fast_lora_low")),
    ("wan.camera.high", "wan_fun", "Fun Camera: high noise", "diffusion", ("wan", "camera", "high")),
    ("wan.camera.low", "wan_fun", "Fun Camera: low noise", "diffusion", ("wan", "camera", "low")),
    ("wan.camera.fast_lora_high", "wan_fun", "Fun Camera fast LoRA: high", "lora", ("wan", "camera", "fast_lora_high")),
    ("wan.camera.fast_lora_low", "wan_fun", "Fun Camera fast LoRA: low", "lora", ("wan", "camera", "fast_lora_low")),
    ("wan.control.high", "wan_fun", "Fun Control: high noise", "diffusion", ("wan", "control", "high")),
    ("wan.control.low", "wan_fun", "Fun Control: low noise", "diffusion", ("wan", "control", "low")),
    ("wan.control.fast_lora_high", "wan_fun", "Fun Control fast LoRA: high", "lora", ("wan", "control", "fast_lora_high")),
    ("wan.control.fast_lora_low", "wan_fun", "Fun Control fast LoRA: low", "lora", ("wan", "control", "fast_lora_low")),
    ("wan.fast_steps", "wan", "Fast steps (i2v/t2v)", "int", ("wan", "fast", "steps")),
    ("wan.quality_steps", "wan", "Quality steps (i2v/t2v)", "int", ("wan", "quality", "steps")),
    ("wan.cfg", "wan", "CFG (quality, i2v/t2v)", "float", ("wan", "quality", "cfg")),
    ("wan.shift", "wan", "Shift", "float", ("wan", "shift")),
]
SLOT_BY_ID = {s[0]: s for s in SLOTS}
GROUPS = {"qwen21": "Keyframes – Qwen Image 2.1", "qwen": "Keyframes – Qwen-Image 2512 / Edit 2511", "wan": "Video – Wan",
          "wan_fun": "Video – Wan 2.2 Fun (InP, Camera, Control)"}

# one-click slot sets
PRESETS = {
    "vace_21": {"label": "VACE: Wan 2.1 VACE 14B (single stage)", "group": "wan", "values": {
        "wan.vace.high": "", "wan.vace.low": "", "wan.vace.fast_lora_high": "", "wan.vace.fast_lora_low": ""}},
    "vace_fun22": {"label": "VACE: Wan 2.2 Fun VACE A14B (two stage)", "group": "wan", "values": {
        "wan.vace.high": "wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors",
        "wan.vace.low": "wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors",
        "wan.vace.weight_dtype": "default",
        "wan.vace.fast_lora_high": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
        "wan.vace.fast_lora_low": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"}},
}
DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]


def _get(d, path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _set(d, path, value):
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def effective_config(cfg: dict, settings: dict) -> dict:
    """Global config with the project's model overrides applied."""
    eff = copy.deepcopy(cfg)
    if settings.get("image_model"):
        eff["image_model"] = settings["image_model"]
    for sid, value in (settings.get("models") or {}).items():
        slot = SLOT_BY_ID.get(sid)
        if slot is None or value in (None, ""):
            continue
        kind = slot[3]
        if value == NONE:
            value = None
        elif kind == "int":
            value = int(value)
        elif kind == "float":
            value = float(value)
        _set(eff, slot[4], value)
    return eff


# ------------------------------------------------------------------ file lists

def _list(folder):
    try:
        import folder_paths
        return list(folder_paths.get_filename_list(folder))
    except Exception:
        return []


def _node_available(name):
    try:
        import nodes
        return name in nodes.NODE_CLASS_MAPPINGS
    except Exception:
        return False


def available():
    gguf = _node_available("UnetLoaderGGUF")
    diff = _list("diffusion_models")
    ckpt = _list("checkpoints")
    unet_gguf = _list("unet_gguf") if gguf else []
    te = _list("text_encoders")
    te_gguf = _list("clip_gguf") if _node_available("CLIPLoaderGGUF") else []
    return {
        "diffusion": [{"value": f, "src": "diffusion_models"} for f in diff]
                     + [{"value": f, "src": "checkpoints"} for f in ckpt if f not in diff]
                     + [{"value": f, "src": "gguf"} for f in unet_gguf if f not in diff],
        "text_encoder": [{"value": f, "src": "text_encoders"} for f in te if not f.endswith(".gguf")]
                        + [{"value": f, "src": "gguf"} for f in te_gguf],
        "vae": [{"value": f, "src": "vae"} for f in _list("vae")],
        "lora": [{"value": f, "src": "loras"} for f in _list("loras")],
        "upscale": [{"value": f, "src": "upscale_models"} for f in _list("upscale_models")],
        "interpolation": [{"value": f, "src": "frame_interpolation"} for f in _list("frame_interpolation")],
        "gguf_installed": gguf,
    }


def slot_info(cfg):
    return [{"id": s[0], "group": s[1], "label": s[2], "kind": s[3], "default": _get(cfg, s[4])} for s in SLOTS]


# ------------------------------------------------------------------ loaders

class Loaders:
    """Adds the right loader node for each file. '@checkpoint' text encoders / VAEs come from
    the CheckpointLoaderSimple of the given checkpoint (the Graph de-duplicates loader nodes,
    so the checkpoint is loaded once)."""

    def __init__(self, g):
        self.g = g
        self.lists = None

    def _lists(self):
        if self.lists is None:
            self.lists = {k: set(_list(k)) for k in ("diffusion_models", "checkpoints", "unet_gguf")}
        return self.lists

    def is_checkpoint(self, name):
        L = self._lists()
        return bool(name) and name in L["checkpoints"] and name not in L["diffusion_models"]

    def _ckpt(self, name, title=None):
        return self.g.add("CheckpointLoaderSimple", _title=title or "Checkpoint", ckpt_name=name)

    def model(self, name, dtype="default", title=None):
        if self.is_checkpoint(name):
            return [self._ckpt(name, title), 0]
        if name.endswith(".gguf") and name not in self._lists()["diffusion_models"]:
            if not _node_available("UnetLoaderGGUF"):
                raise ValueError(f"{name} is a GGUF file - install ComfyUI-GGUF to use it")
            return [self.g.add("UnetLoaderGGUF", _title=title, unet_name=name), 0]
        return [self.g.add("UNETLoader", _title=title, unet_name=name, weight_dtype=dtype or "default"), 0]

    def clip(self, name, clip_type, ckpt=None, title=None):
        if name == FROM_CKPT:
            if not self.is_checkpoint(ckpt):
                raise ValueError("text encoder is set to 'from checkpoint' but the model is not an all-in-one checkpoint")
            return [self._ckpt(ckpt), 1]
        if name.endswith(".gguf"):
            if not _node_available("CLIPLoaderGGUF"):
                raise ValueError(f"{name} is a GGUF file - install ComfyUI-GGUF to use it")
            return [self.g.add("CLIPLoaderGGUF", _title=title, clip_name=name, type=clip_type), 0]
        return [self.g.add("CLIPLoader", _title=title, clip_name=name, type=clip_type, device="default"), 0]

    def vae(self, name, ckpt=None, title=None):
        if name == FROM_CKPT:
            if not self.is_checkpoint(ckpt):
                raise ValueError("VAE is set to 'from checkpoint' but the model is not an all-in-one checkpoint")
            return [self._ckpt(ckpt), 2]
        return [self.g.add("VAELoader", _title=title, vae_name=name), 0]
