"""Builds ComfyUI API-format prompts for every Director operation.

Each builder returns a dict {node_id: {class_type, inputs}} ready for POST /prompt.
Only core ComfyUI nodes plus this pack's Director* nodes are used.
"""
from __future__ import annotations

import json
import random

from . import models as M
from . import project as P


# loader nodes with identical inputs are shared, so a model or checkpoint is loaded once per graph
DEDUP = {"UNETLoader", "CLIPLoader", "VAELoader", "CheckpointLoaderSimple", "UnetLoaderGGUF", "CLIPLoaderGGUF",
         "DirectorLoadImage", "LoraLoaderModelOnly"}


class Graph:
    def __init__(self):
        self.nodes = {}
        self._n = 0
        self._memo = {}
        self.load = M.Loaders(self)

    def add(self, class_type, _title=None, **inputs):
        inputs = {k: v for k, v in inputs.items() if v is not None}
        key = None
        if class_type in DEDUP:
            key = (class_type, json.dumps(inputs, sort_keys=True))
            if key in self._memo:
                return self._memo[key]
        self._n += 1
        nid = str(self._n)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs, "_meta": {"title": _title or class_type}}
        if key:
            self._memo[key] = nid
        return nid


def out(nid, i=0):
    return [nid, i]


def rand_seed():
    return random.randint(1, 2**48)


# ---------------------------------------------------------------- shared bits

def merge_loras(*lists):
    seen, outl = set(), []
    for lst in lists:
        for l in lst or []:
            if not l or not l.get("name") or l.get("enabled") is False:
                continue
            key = (l["name"], l.get("stage", "both"))
            if key in seen:
                continue
            seen.add(key)
            outl.append(l)
    return outl


def apply_loras(g, model, loras, stage=None):
    for l in loras:
        st = l.get("stage", "both")
        if stage is not None and st not in ("both", stage):
            continue
        model = out(g.add("LoraLoaderModelOnly", _title=f"LoRA {l['name']}", model=model,
                          lora_name=l["name"], strength_model=float(l.get("strength", 1.0))))
    return model


def join_prompt(*parts):
    return ", ".join(p.strip().rstrip(",") for p in parts if p and p.strip())


def meta_json(**kw):
    return json.dumps(kw, ensure_ascii=False)


# ---------------------------------------------------------------- Qwen-Image 2512 / Edit 2511 (legacy, image_model="qwen")

def _qwen_base(g, cfg, settings, which, loras):
    q = cfg["qwen"]
    fast = settings.get("fast_image", True)
    mname = q["t2i_model"] if which == "t2i" else q["edit_model"]
    unet = g.load.model(mname, q.get("weight_dtype", "default"), "Qwen model")
    clip = g.load.clip(q["text_encoder"], "qwen_image", ckpt=mname)
    vae = g.load.vae(q["vae"], ckpt=mname)
    model = unet
    if fast:
        fl = q["fast"].get("t2i_lora" if which == "t2i" else "edit_lora")
        if fl:
            model = apply_loras(g, model, [{"name": fl, "strength": 1.0}])
    model = apply_loras(g, model, loras)
    model = out(g.add("ModelSamplingAuraFlow", model=model, shift=float(q.get("shift", 3.1))))
    samp = q["fast"] if fast else q["quality"]
    return model, clip, vae, samp


def build_t2i(cfg, proj, kf, seed=None):
    s = proj["settings"]
    rec = kf["recipe"]
    seed = seed or rand_seed()
    w, h = P.keyframe_size(s)
    loras = merge_loras(s.get("image_loras"), rec.get("loras"))
    g = Graph()
    model, clip, vae, samp = _qwen_base(g, cfg, s, "t2i", loras)
    prompt = join_prompt(rec.get("prompt", ""), s.get("style_prompt", ""))
    pos = out(g.add("CLIPTextEncode", _title="Prompt", clip=clip, text=prompt))
    neg = out(g.add("CLIPTextEncode", _title="Negative", clip=clip, text=cfg["qwen"].get("negative", "")))
    lat = out(g.add("EmptySD3LatentImage", width=w, height=h, batch_size=1))
    smp = out(g.add("KSampler", model=model, seed=seed, steps=samp["steps"], cfg=samp["cfg"],
                    sampler_name=samp["sampler"], scheduler=samp["scheduler"], positive=pos, negative=neg,
                    latent_image=lat, denoise=1.0))
    img = out(g.add("VAEDecode", samples=smp, vae=vae))
    g.add("DirectorSaveImage", images=img, project=proj["id"], target=f"keyframe:{kf['id']}",
          meta=meta_json(op="t2i", seed=seed, prompt=prompt, loras=loras))
    return g.nodes


def build_edit(cfg, proj, kf, seed=None, source_file=None, ref_files=None, mask_file=None, prompt=None,
               grow=16, feather=24, padding=96):
    """Qwen Image Edit. Used for derive (new keyframe from 1-3 parents) and for
    inpainting a region of an existing take (mask_file set)."""
    s = proj["settings"]
    rec = kf["recipe"]
    seed = seed or rand_seed()
    kw, kh = P.keyframe_size(s)
    loras = merge_loras(s.get("image_loras"), rec.get("loras"))
    if source_file is None:
        parents = rec.get("parents", [])
        source_file = P.kf_image(proj, parents[0])
        ref_files = [P.kf_image(proj, p) for p in parents[1:3]]
        if mask_file is None and rec.get("mask"):
            mask_file = rec["mask"]
            grow = int(rec.get("grow", grow))
    ref_files = ref_files or []
    prompt = prompt if prompt is not None else rec.get("prompt", "")
    g = Graph()
    model, clip, vae, samp = _qwen_base(g, cfg, s, "edit", loras)
    model = out(g.add("CFGNorm", model=model, strength=1.0))
    src = out(g.add("DirectorLoadImage", _title="Source", project=proj["id"], path=source_file,
                    width=kw, height=kh, fit="cover"))
    work, crop = src, None
    if mask_file:
        m = out(g.add("DirectorLoadMask", project=proj["id"], path=mask_file, width=kw, height=kh))
        crop = g.add("DirectorMaskCrop", image=src, mask=m, padding=padding, grow=grow, megapixels=1.0, multiple=16)
        work = out(crop, 0)
    refs = {}
    for i, rf in enumerate(ref_files[:2]):
        refs[f"image{i + 2}"] = out(g.add("DirectorLoadImage", _title=f"Reference {i + 2}", project=proj["id"],
                                          path=rf, width=0, height=0, fit="none"))
    method = cfg["qwen"].get("edit_reference_method", "index_timestep_zero")
    pos = out(g.add("TextEncodeQwenImageEditPlus", _title="Prompt", clip=clip, prompt=prompt, vae=vae,
                    image1=work, **refs))
    neg = out(g.add("TextEncodeQwenImageEditPlus", _title="Negative", clip=clip, prompt="", vae=vae,
                    image1=work, **refs))
    if method:
        pos = out(g.add("FluxKontextMultiReferenceLatentMethod", conditioning=pos, reference_latents_method=method))
        neg = out(g.add("FluxKontextMultiReferenceLatentMethod", conditioning=neg, reference_latents_method=method))
    lat = out(g.add("VAEEncode", pixels=work, vae=vae))
    if crop:
        lat = out(g.add("SetLatentNoiseMask", samples=lat, mask=out(crop, 1)))
    smp = out(g.add("KSampler", model=model, seed=seed, steps=samp["steps"], cfg=samp["cfg"],
                    sampler_name=samp["sampler"], scheduler=samp["scheduler"], positive=pos, negative=neg,
                    latent_image=lat, denoise=1.0))
    img = out(g.add("VAEDecode", samples=smp, vae=vae))
    if crop:
        img = out(g.add("DirectorMaskStitch", original=src, patch=img, crop_info=out(crop, 2), feather=feather))
    op = "inpaint" if mask_file else "edit"
    g.add("DirectorSaveImage", images=img, project=proj["id"], target=f"keyframe:{kf['id']}",
          meta=meta_json(op=op, seed=seed, prompt=prompt, source=source_file, refs=ref_files, mask=mask_file,
                         loras=loras))
    return g.nodes


# ---------------------------------------------------------------- Qwen Image 2.1 (images)
#
# One model for text->image and editing. Reference images go into TextEncodeQwenImage21
# (images.image_1 .. image_16); image_1 is the image being edited and sets the output size.
# Text->image uses its own empty latent at the keyframe size so the aspect matches the video.
# Inpainting: crop around the mask, edit the crop with a latent noise mask, stitch back.

QI21_MAX_REFS = 16


def _qi21_base(g, cfg, settings, loras, width, height):
    q = cfg["qwen21"]
    fast = settings.get("fast_image", True)
    model = g.load.model(q["model"], q.get("weight_dtype", "default"), "Qwen Image 2.1")
    clip = g.load.clip(q["text_encoder"], "qwen_image", ckpt=q["model"])
    vae = g.load.vae(q["vae"], ckpt=q["model"])
    samp = q["fast"] if fast else q["quality"]
    if fast and samp.get("lora"):
        model = apply_loras(g, model, [{"name": samp["lora"], "strength": float(samp.get("lora_strength", 1.0))}])
    model = apply_loras(g, model, loras)
    if q.get("resolution_aware_shift", True):
        # the model's exponential shift is fitted at 1024x1024 (mu 0.5 @ 256 tokens, 0.9 @ 8192 tokens);
        # ModelSamplingFlux reproduces that line for the actual output size.
        mu_max = 0.5 + 0.4 * (4096 - 256) / (8192 - 256)
        model = out(g.add("ModelSamplingFlux", model=model, max_shift=round(mu_max, 4), base_shift=0.5,
                          width=int(width), height=int(height)))
    cache = q.get("cache") or {}
    if cache.get("device", "auto") != "auto" or cache.get("dtype", "default") != "default":
        model = out(g.add("QwenImage21Cache", model=model, device=cache.get("device", "auto"),
                          dtype=cache.get("dtype", "default")))
    return model, clip, vae, samp


def build_qi21(cfg, proj, kf, seed=None, source_file=None, ref_files=None, mask_file=None, prompt=None,
               grow=16, feather=24, padding=96):
    """Qwen Image 2.1 keyframe graph.
    - recipe t2i : parents are optional references
    - recipe edit: parents[0] is the base image (image_1), the rest are references
    - mask_file  : inpaint a region of source_file (the take being improved)
    Project-wide reference keyframes are appended after the recipe's own images."""
    s = proj["settings"]
    rec = kf["recipe"]
    q = cfg["qwen21"]
    seed = seed or rand_seed()
    kw, kh = P.keyframe_size(s)
    loras = merge_loras(s.get("image_loras"), rec.get("loras"))
    prompt = prompt if prompt is not None else rec.get("prompt", "")
    full_prompt = join_prompt(prompt, s.get("style_prompt", ""))
    parents = [p for p in rec.get("parents", []) if p != kf["id"]]
    if source_file is None and rec.get("op") == "edit":
        source_file = P.kf_image(proj, parents[0])
        parents = parents[1:]
        if mask_file is None and rec.get("mask"):
            mask_file = rec["mask"]  # derived keyframe that only repaints a region of image1
            grow = int(rec.get("grow", grow))
    if ref_files is None:
        ref_files = [P.kf_image(proj, p) for p in parents]
    ref_files = list(ref_files) + [f for f in P.project_refs(proj, exclude={kf["id"], *rec.get("parents", [])})
                                   if f not in ref_files and f != source_file]
    ref_files = ref_files[:QI21_MAX_REFS - (1 if source_file else 0)]

    g = Graph()
    model, clip, vae, samp = _qi21_base(g, cfg, s, loras, kw, kh)
    images, src, crop = {}, None, None
    slot = 1
    if source_file:
        src = out(g.add("DirectorLoadImage", _title="Base image (image1)", project=proj["id"], path=source_file,
                        width=kw, height=kh, fit="cover"))
        work = src
        if mask_file:
            m = out(g.add("DirectorLoadMask", project=proj["id"], path=mask_file, width=kw, height=kh))
            crop = g.add("DirectorMaskCrop", image=src, mask=m, padding=padding, grow=grow,
                         megapixels=float(q.get("inpaint_megapixels", 1.0)), multiple=32)
            work = out(crop, 0)
        images["images.image_1"] = work
        slot = 2
    for rf in ref_files:
        images[f"images.image_{slot}"] = out(g.add("DirectorLoadImage", _title=f"Reference (image{slot})",
                                                   project=proj["id"], path=rf, width=0, height=0, fit="none",
                                                   max_megapixels=float(q.get("reference_megapixels", 1.0))))
        slot += 1
    enc = g.add("TextEncodeQwenImage21", _title="Prompt", clip=clip, prompt=full_prompt,
                negative_prompt=q.get("negative", ""), vae=vae if images else None,
                resolution=int(q.get("edit_resolution", 0)), **images)
    pos, neg = out(enc, 0), out(enc, 1)
    if crop:
        lat = out(g.add("VAEEncode", pixels=out(crop, 0), vae=vae))
        lat = out(g.add("SetLatentNoiseMask", samples=lat, mask=out(crop, 1)))
    elif source_file:
        lat = out(enc, 2)  # empty latent at image_1's size
    else:
        lat = out(g.add("DirectorEmptyLatent", width=kw, height=kh, channels=64, downscale=16, batch_size=1))
    smp = out(g.add("KSampler", model=model, seed=seed, steps=samp["steps"], cfg=samp["cfg"],
                    sampler_name=samp["sampler"], scheduler=samp["scheduler"], positive=pos, negative=neg,
                    latent_image=lat, denoise=1.0))
    img = out(g.add("VAEDecode", samples=smp, vae=vae))
    if crop:
        img = out(g.add("DirectorMaskStitch", original=src, patch=img, crop_info=out(crop, 2), feather=feather))
    op = "inpaint" if mask_file else ("edit" if source_file else "t2i")
    g.add("DirectorSaveImage", images=img, project=proj["id"], target=f"keyframe:{kf['id']}",
          meta=meta_json(op=op, model="qwen_image_2.1", seed=seed, prompt=full_prompt, source=source_file,
                         refs=ref_files, mask=mask_file, loras=loras))
    return g.nodes


# ---------------------------------------------------------------- Wan (video)

def _wan_sample(g, cfg, settings, family, loras, pos, neg, latent, seed, vace=False):
    """Two-stage (high/low noise) or single-stage Wan sampling. Returns latent."""
    w = cfg["wan"]
    fam = w[family]
    fast = settings.get("fast_video", True)
    samp = (w["vace_fast"] if fast else w["vace_quality"]) if vace else (w["fast"] if fast else w["quality"])
    dtype = fam.get("weight_dtype", w.get("weight_dtype", "default"))
    fstr = float(fam.get("fast_lora_strength", 1.0))

    def stage_model(fname, fast_lora, stage):
        m = g.load.model(fname, dtype, f"Wan {family} {stage}")
        if fast and fast_lora:
            m = apply_loras(g, m, [{"name": fast_lora, "strength": fstr}])
        m = apply_loras(g, m, loras, stage=stage if fam.get("low") else None)
        return out(g.add("ModelSamplingSD3", model=m, shift=float(w.get("shift", 5.0))))

    steps = int(samp["steps"])
    if fam.get("low"):
        hi = stage_model(fam["high"], fam.get("fast_lora_high"), "high")
        lo = stage_model(fam["low"], fam.get("fast_lora_low"), "low")
        split = max(1, steps // 2)
        a = out(g.add("KSamplerAdvanced", _title="High noise", model=hi, add_noise="enable", noise_seed=seed,
                      steps=steps, cfg=samp["cfg"], sampler_name=samp["sampler"], scheduler=samp["scheduler"],
                      positive=pos, negative=neg, latent_image=latent, start_at_step=0, end_at_step=split,
                      return_with_leftover_noise="enable"))
        return out(g.add("KSamplerAdvanced", _title="Low noise", model=lo, add_noise="disable", noise_seed=seed,
                         steps=steps, cfg=samp["cfg"], sampler_name=samp["sampler"], scheduler=samp["scheduler"],
                         positive=pos, negative=neg, latent_image=a, start_at_step=split, end_at_step=10000,
                         return_with_leftover_noise="disable"))
    m = stage_model(fam["high"], fam.get("fast_lora_high"), "both")
    return out(g.add("KSampler", model=m, seed=seed, steps=steps, cfg=samp["cfg"], sampler_name=samp["sampler"],
                     scheduler=samp["scheduler"], positive=pos, negative=neg, latent_image=latent, denoise=1.0))


def _wan_ckpt(g, cfg, family):
    """Checkpoint that '@checkpoint' text encoder / VAE should come from: this graph's own model if it
    is an all-in-one checkpoint, otherwise any Wan slot that is one."""
    w = cfg["wan"]
    fams = [f for f in ("i2v", "t2v", "vace", "inp", "camera", "control") if isinstance(w.get(f), dict)]
    for name in [w[family]["high"]] + [w[f].get(k) for f in fams for k in ("high", "low")]:
        if name and g.load.is_checkpoint(name):
            return name
    return w[family]["high"]


def _load_frame(g, pid, spec, w, h, title):
    if spec is None:
        return None
    if "file" in spec:
        return out(g.add("DirectorLoadImage", _title=title, project=pid, path=spec["file"], width=w, height=h,
                         fit="cover"))
    return out(g.add("DirectorLoadVideoFrames", _title=title, project=pid, path=spec["video"],
                     start=int(spec["frame"]), count=1, width=w, height=h), 0)


def _load_ctx(g, pid, spec, w, h, title, side):
    if spec is None:
        return None
    if "image" in spec:
        return _load_frame(g, pid, spec["image"], w, h, title)
    if side == "before":
        start = max(0, int(spec["end"]) - int(spec["count"]) + 1)
        count = int(spec["end"]) - start + 1
    else:
        start, count = int(spec["start"]), int(spec["count"])
    return out(g.add("DirectorLoadVideoFrames", _title=title, project=pid, path=spec["file"], start=start,
                     count=count, width=w, height=h), 0)


def build_segment(cfg, proj, seg_index, seed=None):
    s = proj["settings"]
    seg = proj["timeline"][seg_index]
    r = P.resolve(proj, seg_index)
    mode = r["mode"]
    seed = seed or rand_seed()
    W, H, fps = int(s["width"]), int(s["height"]), int(s["fps"])
    loras = merge_loras(s.get("video_loras"), seg.get("loras"))
    prompt = join_prompt(seg.get("prompt", ""), s.get("video_style_prompt", ""))
    pid = proj["id"]
    g = Graph()
    ck = _wan_ckpt(g, cfg, {"t2v": "t2v", "vace": "vace", "camera": "camera", "control": "control", "inp": "inp"}.get(mode, "i2v"))
    clip = g.load.clip(cfg["wan"]["text_encoder"], "wan", ckpt=ck)
    vae = g.load.vae(cfg["wan"]["vae"], ckpt=ck)
    pos = out(g.add("CLIPTextEncode", _title="Prompt", clip=clip, text=prompt))
    neg = out(g.add("CLIPTextEncode", _title="Negative", clip=clip, text=cfg["wan"].get("negative", "")))
    length = P.wan_length(seg["frames"])
    trim_start = trim_end = 0
    first_anchor = last_anchor = False

    if mode == "t2v":
        lat = out(g.add("EmptyHunyuanLatentVideo", width=W, height=H, length=length, batch_size=1))
        smp = _wan_sample(g, cfg, s, "t2v", loras, pos, neg, lat, seed)
        family = "t2v"
    elif mode == "camera":
        st = _load_frame(g, pid, r["start_image"], W, H, "Start frame")
        mo = seg.get("motion") or {}
        cam = g.add("WanCameraEmbedding", camera_pose=mo.get("camera", "Zoom In"), width=W, height=H, length=length,
                    speed=float(mo.get("speed", 1.0)))
        c = g.add("WanCameraImageToVideo", positive=pos, negative=neg, vae=vae, width=W, height=H, length=length,
                  batch_size=1, start_image=st, camera_conditions=out(cam, 0))
        smp = _wan_sample(g, cfg, s, "camera", loras, out(c, 0), out(c, 1), out(c, 2), seed)
        first_anchor = True
    elif mode == "control":
        ctl = r["control"]
        cv = out(g.add("DirectorLoadVideoFrames", _title="Control video", project=pid, path=ctl["file"],
                       start=ctl["offset"], count=length, width=W, height=H), 0)
        if ctl["preprocess"] == "canny":
            cv = out(g.add("Canny", image=cv, low_threshold=ctl["low"], high_threshold=ctl["high"]))
        ref = _load_frame(g, pid, r["start_image"], W, H, "Reference (start frame)")
        c = g.add("Wan22FunControlToVideo", positive=pos, negative=neg, vae=vae, width=W, height=H, length=length,
                  batch_size=1, ref_image=ref, control_video=cv)
        smp = _wan_sample(g, cfg, s, "control", loras, out(c, 0), out(c, 1), out(c, 2), seed)
    elif mode == "inp":
        st = _load_frame(g, pid, r["start_image"], W, H, "Start frame")
        en = _load_frame(g, pid, r["end_image"], W, H, "End frame")
        first_anchor, last_anchor = st is not None, en is not None
        c = g.add("WanFunInpaintToVideo", positive=pos, negative=neg, vae=vae, width=W, height=H, length=length,
                  batch_size=1, start_image=st, end_image=en)
        smp = _wan_sample(g, cfg, s, "inp", loras, out(c, 0), out(c, 1), out(c, 2), seed)
    elif mode in ("i2v", "flf2v"):
        st = _load_frame(g, pid, r["start_image"], W, H, "Start frame")
        en = _load_frame(g, pid, r["end_image"], W, H, "End frame")
        first_anchor, last_anchor = st is not None, en is not None
        if mode == "i2v":
            c = g.add("WanImageToVideo", positive=pos, negative=neg, vae=vae, width=W, height=H, length=length,
                      batch_size=1, start_image=st)
        else:
            c = g.add("WanFirstLastFrameToVideo", positive=pos, negative=neg, vae=vae, width=W, height=H,
                      length=length, batch_size=1, start_image=st, end_image=en)
        smp = _wan_sample(g, cfg, s, "i2v", loras, out(c, 0), out(c, 1), out(c, 2), seed)
    elif mode == "vace":
        before = _load_ctx(g, pid, r["before"], W, H, "Context before", "before")
        after = _load_ctx(g, pid, r["after"], W, H, "Context after", "after")
        vc = cfg["wan"]["vace"]
        br = g.add("DirectorVaceBridge", width=W, height=H, new_frames=int(seg["frames"]),
                   keep_single_anchor=True, before_frames=before, after_frames=after,
                   fill=float(vc.get("fill", 0.5)))
        c = g.add("WanVaceToVideo", positive=pos, negative=neg, vae=vae, width=W, height=H, length=out(br, 2),
                  batch_size=1, strength=float(cfg["wan"]["vace"].get("strength", 1.0)),
                  control_video=out(br, 0), control_masks=out(br, 1))
        smp = _wan_sample(g, cfg, s, "vace", loras, out(c, 0), out(c, 1), out(c, 2), seed, vace=True)
        smp = out(g.add("TrimVideoLatent", samples=smp, trim_amount=out(c, 3)))
        trim_start, trim_end = out(br, 3), out(br, 4)
        vace_ref = (out(br, 0), out(br, 5))  # true context frames + where they are
        first_anchor = bool(r["before"] and "image" in r["before"])
        last_anchor = bool(r["after"] and "image" in r["after"])
    else:
        raise ValueError(mode)

    img = out(g.add("VAEDecode", samples=smp, vae=vae))
    cm = float(cfg["wan"]["vace"].get("color_match", 1.0))
    if mode == "vace" and cm > 0:
        img = out(g.add("DirectorColorMatch", _title="Colour match to context", images=img,
                        reference=vace_ref[0], ref_mask=vace_ref[1], strength=cm))
    g.add("DirectorSaveVideo", images=img, project=pid, target=f"segment:{seg['id']}", fps=fps,
          trim_start=trim_start, trim_end=trim_end, splice_source="", splice_start=0,
          meta=meta_json(op="generate", mode=mode, seed=seed, prompt=prompt, loras=loras,
                         first_is_anchor=first_anchor, last_is_anchor=last_anchor,
                         start_ref=r["start_ref"], end_ref=r["end_ref"]))
    return g.nodes


def build_video_inpaint(cfg, proj, seg_index, take_id, mask_file, frame_start, frame_end, prompt=None,
                        grow=16, seed=None):
    """VACE inpaint of a masked region over a frame range; result is spliced back into a new take."""
    s = proj["settings"]
    seg = proj["timeline"][seg_index]
    take = next(t for t in seg["takes"] if t["id"] == take_id)
    seed = seed or rand_seed()
    W, H = int(s["width"]), int(s["height"])
    n = int(take.get("frames", 81))
    a = max(0, int(frame_start))
    b = min(n - 1, int(frame_end) if frame_end is not None and int(frame_end) >= 0 else n - 1)
    count = min(b - a + 1, int(s["max_segment_frames"]))
    loras = merge_loras(s.get("video_loras"), seg.get("loras"))
    prompt = join_prompt(prompt if prompt is not None else seg.get("prompt", ""), s.get("video_style_prompt", ""))
    pid = proj["id"]
    g = Graph()
    ck = _wan_ckpt(g, cfg, "vace")
    clip = g.load.clip(cfg["wan"]["text_encoder"], "wan", ckpt=ck)
    vae = g.load.vae(cfg["wan"]["vae"], ckpt=ck)
    pos = out(g.add("CLIPTextEncode", _title="Prompt", clip=clip, text=prompt))
    neg = out(g.add("CLIPTextEncode", _title="Negative", clip=clip, text=cfg["wan"].get("negative", "")))
    frames = out(g.add("DirectorLoadVideoFrames", _title="Take frames", project=pid, path=take["file"], start=a,
                       count=count, width=W, height=H), 0)
    mask = out(g.add("DirectorLoadMask", project=pid, path=mask_file, width=W, height=H))
    vc = cfg["wan"]["vace"]
    vi = g.add("DirectorVaceInpaint", frames=frames, mask=mask, grow=grow, fill=float(vc.get("fill", 0.5)))
    c = g.add("WanVaceToVideo", positive=pos, negative=neg, vae=vae, width=W, height=H, length=out(vi, 2),
              batch_size=1, strength=float(cfg["wan"]["vace"].get("strength", 1.0)),
              control_video=out(vi, 0), control_masks=out(vi, 1))
    smp = _wan_sample(g, cfg, s, "vace", loras, out(c, 0), out(c, 1), out(c, 2), seed, vace=True)
    smp = out(g.add("TrimVideoLatent", samples=smp, trim_amount=out(c, 3)))
    dec = out(g.add("VAEDecode", samples=smp, vae=vae))
    cm = float(vc.get("color_match", 1.0))
    if cm > 0:
        dec = out(g.add("DirectorColorMatch", _title="Colour match to surroundings", images=dec,
                        reference=frames, ref_mask=out(vi, 4), strength=cm))
    comp = out(g.add("DirectorCompositeFrames", original=frames, generated=dec, mask=out(vi, 3)))
    g.add("DirectorSaveVideo", images=comp, project=pid, target=f"segment:{seg['id']}", fps=int(s["fps"]),
          trim_start=0, trim_end=0, splice_source=take["file"], splice_start=a,
          meta=meta_json(op="inpaint", seed=seed, prompt=prompt, parent_take=take_id, mask=mask_file,
                         range=[a, a + count - 1], loras=loras,
                         first_is_anchor=take.get("first_is_anchor", False),
                         last_is_anchor=take.get("last_is_anchor", False),
                         start_ref=take.get("start_ref"), end_ref=take.get("end_ref")))
    return g.nodes


def build_interpolate(cfg, proj, render_file, multiplier=2):
    s = proj["settings"]
    g = Graph()
    frames = out(g.add("DirectorLoadVideoFrames", project=proj["id"], path=render_file, start=0, count=-1,
                       width=0, height=0), 0)
    m = out(g.add("FrameInterpolationModelLoader", model_name=cfg.get("interpolation_model", "film_net_fp16.safetensors")))
    fi = out(g.add("FrameInterpolate", interp_model=m, images=frames, multiplier=int(multiplier)))
    g.add("DirectorSaveVideo", images=fi, project=proj["id"], target="render:interp",
          fps=int(s["fps"]) * int(multiplier), trim_start=0, trim_end=0, splice_source="", splice_start=0,
          meta=meta_json(op="interpolate", source=render_file))
    return g.nodes


# ---------------------------------------------------------------- dispatcher

def build(cfg, proj, req):
    """req: {action, id, count, ...} -> list of {prompt, target, seed}"""
    cfg = M.effective_config(cfg, proj["settings"])
    action = req["action"]
    count = max(1, min(8, int(req.get("count") or proj["settings"].get("takes_per_run", 1))))
    jobs = []
    for _ in range(count):
        seed = rand_seed()
        if action == "keyframe.generate":
            kf = proj["keyframes"][req["id"]]
            ok, why = P.keyframe_ready(proj, kf)
            if not ok:
                raise P.NotReady(why)
            if cfg.get("image_model", "qwen21") == "qwen21":
                g = build_qi21(cfg, proj, kf, seed)
            elif kf["recipe"]["op"] == "edit":
                g = build_edit(cfg, proj, kf, seed)
            else:
                g = build_t2i(cfg, proj, kf, seed)
            target = f"keyframe:{kf['id']}"
        elif action == "keyframe.inpaint":
            kf = proj["keyframes"][req["id"]]
            take = next(t for t in kf["takes"] if t["id"] == req["take"])
            if cfg.get("image_model", "qwen21") == "qwen21":
                g = build_qi21(cfg, proj, kf, seed, source_file=take["file"], ref_files=[], mask_file=req["mask"],
                               prompt=req.get("prompt", ""), grow=int(req.get("grow", 16)))
            else:
                g = build_edit(cfg, proj, kf, seed, source_file=take["file"], ref_files=[], mask_file=req["mask"],
                               prompt=req.get("prompt", ""), grow=int(req.get("grow", 16)))
            target = f"keyframe:{kf['id']}"
        elif action == "segment.generate":
            i, seg = P.find_segment(proj, req["id"])
            g = build_segment(cfg, proj, i, seed)
            target = f"segment:{seg['id']}"
        elif action == "segment.inpaint":
            i, seg = P.find_segment(proj, req["id"])
            g = build_video_inpaint(cfg, proj, i, req["take"], req["mask"], req.get("frame_start", 0),
                                    req.get("frame_end", -1), req.get("prompt"), int(req.get("grow", 16)), seed)
            target = f"segment:{seg['id']}"
        elif action == "render.interpolate":
            g = build_interpolate(cfg, proj, req["file"], int(req.get("multiplier", 2)))
            target = "render"
            count = 1
        else:
            raise ValueError(f"unknown action {action}")
        jobs.append({"prompt": g, "target": target, "seed": seed})
        if action == "render.interpolate":
            break
    return jobs
