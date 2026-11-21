"""Director nodes: project-aware IO plus small mask/VACE helpers.

These are ordinary ComfyUI nodes, so every graph the Director panel queues can
also be opened, inspected and tweaked by hand.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .director import media
from .director import project as P

CAT = "director"


def _to_tensor(arr):
    return torch.from_numpy(np.asarray(arr).astype(np.float32) / 255.0)


def _to_uint8(t):
    return (t.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()


def _dilate(mask, px):
    """mask: (N,H,W). Separable max filter: two 1-D passes instead of a k*k window."""
    if px <= 0:
        return mask
    px = int(px)
    k = 2 * px + 1
    m = mask.unsqueeze(1)
    m = F.max_pool2d(m, (1, k), stride=1, padding=(0, px))
    m = F.max_pool2d(m, (k, 1), stride=1, padding=(px, 0))
    return m.squeeze(1)


def _feather(mask, px):
    if px <= 0:
        return mask
    m = mask.unsqueeze(1)
    k = 2 * int(px) + 1
    for _ in range(2):  # separable box blur, twice (~gaussian)
        m = F.avg_pool2d(F.pad(m, (px, px, 0, 0), mode="replicate"), (1, k), stride=1)
        m = F.avg_pool2d(F.pad(m, (0, 0, px, px), mode="replicate"), (k, 1), stride=1)
    return m.squeeze(1).clamp(0, 1)


def _mtime(project, path):
    try:
        return os.path.getmtime(P.safe_path(project, path))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------- loaders

class DirectorLoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project": ("STRING", {"default": ""}),
            "path": ("STRING", {"default": ""}),
            "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
            "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
            "fit": (["cover", "contain", "stretch", "none"],),
        }, "optional": {
            "max_megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.1,
                                         "tooltip": "With fit=none: shrink larger images to this size (0 = off). Keeps big reference uploads fast."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = CAT

    @classmethod
    def IS_CHANGED(cls, project, path, **kw):
        return _mtime(project, path)

    def load(self, project, path, width, height, fit, max_megapixels=0.0):
        img = Image.open(P.safe_path(project, path))
        img = media.fit_image(img, width, height, fit if width and height else "none")
        if max_megapixels and img.width * img.height > max_megapixels * 1024 * 1024:
            k = (max_megapixels * 1024 * 1024 / (img.width * img.height)) ** 0.5
            img = img.resize((max(32, round(img.width * k)), max(32, round(img.height * k))), Image.LANCZOS)
        return (_to_tensor(img)[None],)


class DirectorLoadMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project": ("STRING", {"default": ""}),
            "path": ("STRING", {"default": ""}),
            "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
            "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "load"
    CATEGORY = CAT

    @classmethod
    def IS_CHANGED(cls, project, path, **kw):
        return _mtime(project, path)

    def load(self, project, path, width, height):
        img = Image.open(P.safe_path(project, path))
        if img.mode == "RGBA" and np.asarray(img)[..., 3].min() < 255:
            m = img.getchannel("A")
        else:
            m = img.convert("L")
        if width and height:
            m = media.fit_image(m.convert("RGB"), width, height, "cover").convert("L")
        return (_to_tensor(m)[None],)


class DirectorLoadVideoFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project": ("STRING", {"default": ""}),
            "path": ("STRING", {"default": ""}),
            "start": ("INT", {"default": 0, "min": -100000, "max": 100000}),
            "count": ("INT", {"default": -1, "min": -1, "max": 100000}),
            "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
            "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "count")
    FUNCTION = "load"
    CATEGORY = CAT

    @classmethod
    def IS_CHANGED(cls, project, path, **kw):
        return _mtime(project, path)

    def load(self, project, path, start, count, width, height):
        frames = media.read_frames(P.safe_path(project, path), start, count)
        if not frames:
            raise ValueError(f"no frames read from {path} (start={start}, count={count})")
        if width and height:
            frames = [np.asarray(media.fit_image(Image.fromarray(f), width, height, "cover")) for f in frames]
        t = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
        return (t, t.shape[0])


# --------------------------------------------------------------------- VACE helpers

class DirectorVaceBridge:
    """Builds VACE control video/masks: [context before] + [new frames] + [context after].
    Context frames are kept (mask 0); new frames are generated (mask 1). Length is padded
    to Wan's 4k+1. trim_start/trim_end tell the save node which context frames to drop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
            "new_frames": ("INT", {"default": 81, "min": 1, "max": 1000}),
            "keep_single_anchor": ("BOOLEAN", {"default": True,
                                                "tooltip": "A single context image (keyframe) stays in the clip as its first/last frame"}),
        }, "optional": {
            "before_frames": ("IMAGE",),
            "after_frames": ("IMAGE",),
            "fill": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                               "tooltip": "Value of frames to generate: 0.5 grey (VACE convention), 1.0 white (ComfyUI templates)"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "MASK")
    RETURN_NAMES = ("control_video", "control_masks", "length", "trim_start", "trim_end", "context_mask")
    FUNCTION = "build"
    CATEGORY = CAT

    def build(self, width, height, new_frames, keep_single_anchor, before_frames=None, after_frames=None, fill=0.5):
        def fit(t):
            if t is None:
                return None
            if t.shape[1] != height or t.shape[2] != width:
                t = F.interpolate(t.movedim(-1, 1), size=(height, width), mode="bilinear",
                                  align_corners=False).movedim(1, -1)
            return t[..., :3]
        b, a = fit(before_frames), fit(after_frames)
        nb = 0 if b is None else b.shape[0]
        na = 0 if a is None else a.shape[0]
        length = P.wan_length(nb + new_frames + na)
        nn = length - nb - na
        gray = torch.full((nn, height, width, 3), float(fill))
        parts = [x for x in (b, gray, a) if x is not None]
        video = torch.cat(parts, 0)
        mask = torch.cat([torch.zeros(nb, height, width), torch.ones(nn, height, width),
                          torch.zeros(na, height, width)], 0)
        ts = 0 if (nb == 1 and keep_single_anchor) else nb
        te = 0 if (na == 1 and keep_single_anchor) else na
        return (video, mask, length, ts, te, 1.0 - mask)


class DirectorVaceInpaint:
    """Masks a region in existing frames for VACE. A single mask is applied to all frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "mask": ("MASK",),
            "grow": ("INT", {"default": 16, "min": 0, "max": 256}),
        }, "optional": {
            "fill": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                               "tooltip": "Value of the masked area: 0.5 grey (VACE convention), 1.0 white (ComfyUI templates)"}),
            "band": ("INT", {"default": 48, "min": 4, "max": 256,
                             "tooltip": "Width of the ring around the mask used to measure colour drift"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "MASK", "MASK")
    RETURN_NAMES = ("control_video", "control_masks", "length", "composite_mask", "band_mask")
    FUNCTION = "build"
    CATEGORY = CAT

    def build(self, frames, mask, grow, fill=0.5, band=48):
        n, H, W, _ = frames.shape
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode="bilinear").squeeze(1)
        mask = (mask > 0.5).float()
        static = mask.shape[0] == 1
        base = mask[:1] if static else mask[:n]
        base = _dilate(base.contiguous(), grow)                       # mask work happens once for a static mask
        composite = _feather(base, max(2, grow // 2))
        ring = (_dilate(base, band) - _dilate(base, max(2, grow // 2) + 2)).clamp(0, 1)
        length = P.wan_length(n)
        if length > n:
            frames = torch.cat([frames, frames[-1:].expand(length - n, -1, -1, -1)], 0)
        def to_len(x, k):
            if x.shape[0] == 1:
                return x.expand(k, H, W).contiguous()
            return torch.cat([x, x[-1:].expand(k - x.shape[0], -1, -1)], 0) if x.shape[0] < k else x[:k]
        mask = to_len(base, length)
        m = mask.unsqueeze(-1)
        control = frames[..., :3] * (1 - m) + float(fill) * m
        return (control, mask, length, to_len(composite, n), to_len(ring, n))


class DirectorColorMatch:
    """Removes colour drift the sampler introduced. In every frame where ref_mask marks pixels that
    should equal the reference (context frames, or a ring around an inpaint mask), it measures a
    per-channel gain/offset that maps the generated pixels onto the reference, interpolates that
    correction over frames with no reference pixels, and applies it to the whole frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "reference": ("IMAGE",),
            "ref_mask": ("MASK",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, images, reference, ref_mask, strength):
        if strength <= 0:
            return (images,)
        n, H, W, _ = images.shape
        ref = reference[..., :3]
        if ref.shape[1:3] != (H, W):
            ref = F.interpolate(ref.movedim(-1, 1), size=(H, W), mode="bilinear").movedim(1, -1)
        rm = ref_mask
        if rm.shape[-2:] != (H, W):
            rm = F.interpolate(rm.unsqueeze(1), size=(H, W), mode="bilinear").squeeze(1)
        gains = torch.ones(n, 3)
        offs = torch.zeros(n, 3)
        known = torch.zeros(n, dtype=torch.bool)
        static = rm.shape[0] == 1 or bool((rm == rm[:1]).all())
        m_static = (rm[0] > 0.5) if static else None
        for i in range(min(n, ref.shape[0], rm.shape[0] if rm.shape[0] > 1 else n)):
            m = m_static if static else (rm[i] > 0.5)
            if m.sum() < 64:
                continue
            g = images[i, ..., :3][m].float()
            r = ref[i][m].to(g.device).float()
            mg, mr = g.mean(0), r.mean(0)
            sg, sr = g.std(0).clamp_min(1e-3), r.std(0).clamp_min(1e-3)
            gain = (sr / sg).clamp(0.75, 1.33)
            gains[i] = gain.cpu()
            offs[i] = (mr - gain * mg).cpu()
            known[i] = True
        if not known.any():
            return (images,)
        idx = torch.nonzero(known).flatten().tolist()
        for i in range(n):  # fill unknown frames by linear interpolation in time
            if known[i]:
                continue
            prev = max([k for k in idx if k < i], default=None)
            nxt = min([k for k in idx if k > i], default=None)
            if prev is None:
                gains[i], offs[i] = gains[nxt], offs[nxt]
            elif nxt is None:
                gains[i], offs[i] = gains[prev], offs[prev]
            else:
                t = (i - prev) / (nxt - prev)
                gains[i] = gains[prev] * (1 - t) + gains[nxt] * t
                offs[i] = offs[prev] * (1 - t) + offs[nxt] * t
        g = gains.to(images.device)[:, None, None, :]
        o = offs.to(images.device)[:, None, None, :]
        x = images[..., :3]
        corrected = (x + strength * (x * g + o - x)).clamp(0, 1)
        if images.shape[-1] > 3:
            corrected = torch.cat([corrected, images[..., 3:]], -1)
        return (corrected,)


class DirectorCompositeFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"original": ("IMAGE",), "generated": ("IMAGE",), "mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    def run(self, original, generated, mask):
        n = original.shape[0]
        g = generated[:n]
        if g.shape[1:3] != original.shape[1:3]:
            g = F.interpolate(g.movedim(-1, 1), size=original.shape[1:3], mode="bilinear").movedim(1, -1)
        if g.shape[0] < n:
            g = torch.cat([g, original[g.shape[0]:]], 0)
        m = mask
        if m.shape[0] == 1:
            m = m.expand(n, -1, -1)
        if m.shape[1:] != original.shape[1:3]:
            m = F.interpolate(m.unsqueeze(1), size=original.shape[1:3], mode="bilinear").squeeze(1)
        m = m[:n].unsqueeze(-1).to(original.device)
        return (original[..., :3] * (1 - m) + g[..., :3].to(original.device) * m,)


# --------------------------------------------------------------------- crop & stitch

class DirectorMaskCrop:
    """Crops the masked region (plus context padding) and scales it to ~megapixels,
    so inpainting runs at full model resolution on the region that matters."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "mask": ("MASK",),
            "padding": ("INT", {"default": 96, "min": 0, "max": 2048}),
            "grow": ("INT", {"default": 16, "min": 0, "max": 256}),
            "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
        }, "optional": {
            "multiple": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8,
                                 "tooltip": "Crop size is rounded to this (Qwen Image 2.1 works on multiples of 32)"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "CROP_INFO")
    RETURN_NAMES = ("image", "mask", "crop_info")
    FUNCTION = "crop"
    CATEGORY = CAT

    def crop(self, image, mask, padding, grow, megapixels, multiple=32):
        _, H, W, _ = image.shape
        m = mask[:1]
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="bilinear").squeeze(1)
        m = _dilate((m > 0.5).float(), grow)
        ys, xs = torch.nonzero(m[0] > 0.5, as_tuple=True)
        if len(ys) == 0:
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            x0, x1 = int(xs.min()) - padding, int(xs.max()) + 1 + padding
            y0, y1 = int(ys.min()) - padding, int(ys.max()) + 1 + padding
            # keep at least 384px of context so the model sees enough of the scene
            cw, ch = x1 - x0, y1 - y0
            if cw < 384:
                d = (384 - cw) // 2; x0 -= d; x1 += 384 - cw - d
            if ch < 384:
                d = (384 - ch) // 2; y0 -= d; y1 += 384 - ch - d
            # shift into the image
            if x0 < 0: x1 -= x0; x0 = 0
            if y0 < 0: y1 -= y0; y0 = 0
            if x1 > W: x0 -= x1 - W; x1 = W
            if y1 > H: y0 -= y1 - H; y1 = H
            x0, y0 = max(0, x0), max(0, y0)
        crop = image[:, y0:y1, x0:x1, :3]
        cm = m[:, y0:y1, x0:x1]
        ch, cw = crop.shape[1], crop.shape[2]
        scale = (megapixels * 1024 * 1024 / (cw * ch)) ** 0.5
        tw, th = max(multiple, round(cw * scale / multiple) * multiple), max(multiple, round(ch * scale / multiple) * multiple)
        crop = F.interpolate(crop.movedim(-1, 1), size=(th, tw), mode="bicubic", align_corners=False).clamp(0, 1).movedim(1, -1)
        cm = F.interpolate(cm.unsqueeze(1), size=(th, tw), mode="bilinear").squeeze(1)
        info = {"box": (x0, y0, x1, y1), "size": (W, H), "mask": m}
        return (crop, cm, info)


class DirectorMaskStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original": ("IMAGE",), "patch": ("IMAGE",), "crop_info": ("CROP_INFO",),
            "feather": ("INT", {"default": 24, "min": 0, "max": 256}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = CAT

    def stitch(self, original, patch, crop_info, feather):
        x0, y0, x1, y1 = crop_info["box"]
        outs = []
        m = _feather(crop_info["mask"].to(original.device), feather)[0]
        for i in range(patch.shape[0]):
            o = original[min(i, original.shape[0] - 1)].clone()
            p = F.interpolate(patch[i:i + 1].movedim(-1, 1), size=(y1 - y0, x1 - x0), mode="bicubic",
                              align_corners=False).clamp(0, 1).movedim(1, -1)[0].to(o.device)
            mm = m[y0:y1, x0:x1].unsqueeze(-1)
            o[y0:y1, x0:x1, :3] = o[y0:y1, x0:x1, :3] * (1 - mm) + p[..., :3] * mm
            outs.append(o)
        return (torch.stack(outs),)


class DirectorEmptyLatent:
    """Empty latent of any channel count / downscale (Qwen Image 2.1: 64 channels, 1/16)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
            "channels": ("INT", {"default": 64, "min": 1, "max": 256}),
            "downscale": ("INT", {"default": 16, "min": 1, "max": 64}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "make"
    CATEGORY = CAT

    def make(self, width, height, channels, downscale, batch_size):
        import comfy.model_management as mm
        return ({"samples": torch.zeros([batch_size, channels, height // downscale, width // downscale],
                                        device=mm.intermediate_device())},)


# --------------------------------------------------------------------- savers

def _parse_meta(meta):
    try:
        return json.loads(meta) if meta else {}
    except Exception:
        return {"meta_raw": meta}


class DirectorSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "project": ("STRING", {"default": ""}),
            "target": ("STRING", {"default": "", "tooltip": "keyframe:<id>"}),
            "meta": ("STRING", {"default": "{}", "multiline": True}),
        }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = CAT

    def save(self, images, project, target, meta):
        kind, _, eid = target.partition(":")
        rel_dir = f"keyframes/{eid}" if kind == "keyframe" else "renders"
        d = P.safe_path(project, rel_dir)
        os.makedirs(d, exist_ok=True)
        ui = []
        for img in images:
            tid = P.new_id("take")
            rel = f"{rel_dir}/{tid}.png"
            Image.fromarray(_to_uint8(img)).save(P.safe_path(project, rel), compress_level=4)
            take = _parse_meta(meta)
            take.update({"id": tid, "file": rel})
            if kind == "keyframe":
                P.add_take(project, target, take)
            ui.append({"filename": f"{tid}.png", "subfolder": f"director/{project}/{rel_dir}", "type": "output"})
        return {"ui": {"images": ui, "director": [{"project": project, "target": target}]}}


class DirectorSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "project": ("STRING", {"default": ""}),
            "target": ("STRING", {"default": "", "tooltip": "segment:<id> or render:<name>"}),
            "meta": ("STRING", {"default": "{}", "multiline": True}),
            "fps": ("INT", {"default": 16, "min": 1, "max": 120}),
            "trim_start": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "trim_end": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "splice_source": ("STRING", {"default": "", "tooltip": "If set, frames replace a range of this video"}),
            "splice_start": ("INT", {"default": 0, "min": 0, "max": 100000}),
        }}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = CAT

    def save(self, images, project, target, meta, fps, trim_start, trim_end, splice_source, splice_start):
        n = images.shape[0]
        frames = images[trim_start:n - trim_end if trim_end else n]
        arrs = [_to_uint8(f[..., :3]) for f in frames]
        if splice_source:
            src = media.read_frames(P.safe_path(project, splice_source))
            h, w = src[0].shape[:2]
            for i, a in enumerate(arrs):
                j = splice_start + i
                if j < len(src):
                    if a.shape[:2] != (h, w):
                        a = np.asarray(media.fit_image(Image.fromarray(a), w, h, "stretch"))
                    src[j] = a
            arrs = src
        kind, _, eid = target.partition(":")
        tid = P.new_id("take")
        rel_dir = f"clips/{eid}" if kind == "segment" else "renders"
        stem = tid if kind == "segment" else f"{eid or 'render'}_{time.strftime('%Y%m%d_%H%M%S')}"
        files = media.write_take_files(project, rel_dir, stem, arrs, fps)
        take = _parse_meta(meta)
        take.update(files)
        take["id"] = tid
        if kind == "segment":
            P.add_take(project, target, take)
        return {"ui": {"director": [{"project": project, "target": target, "file": files["file"]}],
                       "images": [{"filename": os.path.basename(files["first"]),
                                   "subfolder": f"director/{project}/{rel_dir}", "type": "output"}]}}


NODE_CLASS_MAPPINGS = {
    "DirectorLoadImage": DirectorLoadImage,
    "DirectorLoadMask": DirectorLoadMask,
    "DirectorLoadVideoFrames": DirectorLoadVideoFrames,
    "DirectorVaceBridge": DirectorVaceBridge,
    "DirectorVaceInpaint": DirectorVaceInpaint,
    "DirectorCompositeFrames": DirectorCompositeFrames,
    "DirectorColorMatch": DirectorColorMatch,
    "DirectorMaskCrop": DirectorMaskCrop,
    "DirectorMaskStitch": DirectorMaskStitch,
    "DirectorEmptyLatent": DirectorEmptyLatent,
    "DirectorSaveImage": DirectorSaveImage,
    "DirectorSaveVideo": DirectorSaveVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: k.replace("Director", "Director: ") for k in NODE_CLASS_MAPPINGS}
