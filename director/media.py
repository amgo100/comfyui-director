"""Video/image IO (PyAV + Pillow) and the streaming timeline renderer."""
from __future__ import annotations

import os
import time
from fractions import Fraction

import numpy as np
from PIL import Image

from . import project as P


# ------------------------------------------------------------------ images

def fit_image(img: Image.Image, w: int, h: int, fit="cover") -> Image.Image:
    img = img.convert("RGB")
    if not w or not h or fit == "none":
        return img
    sw, sh = img.size
    if fit == "stretch":
        return img.resize((w, h), Image.LANCZOS)
    scale = max(w / sw, h / sh) if fit == "cover" else min(w / sw, h / sh)
    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    if fit == "cover":
        l, t = (nw - w) // 2, (nh - h) // 2
        return img.crop((l, t, l + w, t + h))
    canvas = Image.new("RGB", (w, h))
    canvas.paste(img, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def save_png(arr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path, compress_level=4)


# ------------------------------------------------------------------ video

def probe(path):
    import av
    with av.open(path) as c:
        st = c.streams.video[0]
        n = st.frames
        fps = float(st.average_rate or st.guessed_rate or 16)
        if not n:
            n = sum(1 for _ in c.decode(video=0))
        return {"frames": int(n), "fps": fps, "width": st.codec_context.width, "height": st.codec_context.height}


def iter_frames(path, start=0, count=-1):
    """Yield HxWx3 uint8 arrays. start<0 counts from the end."""
    import av
    if start < 0:
        start = max(0, probe(path)["frames"] + start)
    with av.open(path) as c:
        i = 0
        emitted = 0
        for fr in c.decode(video=0):
            if i >= start:
                yield fr.to_ndarray(format="rgb24")
                emitted += 1
                if count >= 0 and emitted >= count:
                    return
            i += 1


def read_frames(path, start=0, count=-1):
    return list(iter_frames(path, start, count))


class VideoWriter:
    def __init__(self, path, fps, crf=14):
        import av
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.c = av.open(path, "w")
        self.fps = Fraction(fps).limit_denominator(1001)
        self.st = None
        self.crf = crf
        self.n = 0

    def write(self, arr: np.ndarray):
        import av
        if self.st is None:
            h, w = arr.shape[:2]
            self.w, self.h = w - w % 2, h - h % 2
            self.st = self.c.add_stream("libx264", rate=self.fps)
            self.st.width, self.st.height = self.w, self.h
            self.st.pix_fmt = "yuv420p"
            self.st.options = {"crf": str(self.crf), "preset": "medium"}
        if arr.shape[0] != self.h or arr.shape[1] != self.w:
            arr = np.asarray(fit_image(Image.fromarray(arr), self.w, self.h))
        fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
        for p in self.st.encode(fr):
            self.c.mux(p)
        self.n += 1

    def close(self):
        if self.st is not None:
            for p in self.st.encode():
                self.c.mux(p)
        self.c.close()


def write_take_files(pid, rel_dir, stem, frames, fps):
    """Writes <stem>.mp4 plus lossless first/last PNGs. Returns take fields."""
    d = P.safe_path(pid, rel_dir)
    os.makedirs(d, exist_ok=True)
    vw = VideoWriter(os.path.join(d, stem + ".mp4"), fps)
    first = last = None
    for f in frames:
        if first is None:
            first = f
        last = f
        vw.write(f)
    vw.close()
    save_png(first, os.path.join(d, stem + "_first.png"))
    save_png(last, os.path.join(d, stem + "_last.png"))
    return {"file": f"{rel_dir}/{stem}.mp4", "first": f"{rel_dir}/{stem}_first.png",
            "last": f"{rel_dir}/{stem}_last.png", "frames": vw.n, "width": vw.w, "height": vw.h}


def import_video(pid, src_path, settings, stem):
    """Normalise an imported clip to project fps and resolution."""
    info = probe(src_path)
    W, H, fps = int(settings["width"]), int(settings["height"]), float(settings["fps"])
    step = info["fps"] / fps

    def gen():
        nxt = 0.0
        for i, f in enumerate(iter_frames(src_path)):
            while i >= nxt - 1e-6 and i < nxt + 1 - 1e-6:
                yield np.asarray(fit_image(Image.fromarray(f), W, H))
                nxt += step
    return write_take_files(pid, "imports", stem, gen(), fps)


def grab_frame(pid, video_rel, index, out_rel):
    path = P.safe_path(pid, video_rel)
    frames = read_frames(path, int(index), 1)
    if not frames:
        frames = read_frames(path, -1, 1)
    save_png(frames[0], P.safe_path(pid, out_rel))
    return out_rel


# ------------------------------------------------------------------ render

def _ref_eq(a, b):
    return a is not None and b is not None and list(a) == list(b)


def render(pid, out_name=None, progress=None):
    proj = P.load(pid)
    s = proj["settings"]
    W, H, fps = int(s["width"]), int(s["height"]), int(s["fps"])
    out_name = out_name or time.strftime("render_%Y%m%d_%H%M%S.mp4")
    rel = f"renders/{out_name}"
    vw = VideoWriter(P.safe_path(pid, rel), fps, crf=16)
    prev_end = None  # what the previously emitted segment ended on
    skipped = []
    edl = []
    tl = proj["timeline"]
    for i, seg in enumerate(tl):
        if progress:
            progress(i, len(tl))
        start_frame = vw.n
        if seg["kind"] == "still":
            try:
                img = P.kf_image(proj, seg.get("keyframe"))
            except P.NotReady as e:
                skipped.append({"segment": seg["id"], "why": str(e)})
                continue
            arr = np.asarray(fit_image(Image.open(P.safe_path(pid, img)), W, H))
            n = int(seg["frames"])
            if _ref_eq(prev_end, ("kf", seg.get("keyframe"))):
                n -= 1
            for _ in range(max(0, n)):
                vw.write(arr)
            prev_end = ("kf", seg.get("keyframe"))
        else:
            t = P.selected_take(seg)
            if not t:
                skipped.append({"segment": seg["id"], "why": "no take"})
                prev_end = None
                continue
            n = int(t.get("frames") or P.probe(P.safe_path(pid, t["file"]))["frames"])
            a = int(seg.get("trim_in") or 0)
            b = n - int(seg.get("trim_out") or 0)
            sref = t.get("start_ref")
            if a == 0 and t.get("first_is_anchor") and sref and (
                    list(sref) == ["prev"] or _ref_eq(prev_end, sref)):
                a = 1
            eref = t.get("end_ref")
            if b == n and t.get("last_is_anchor") and eref and list(eref) == ["next"]:
                b = n - 1
            for f in iter_frames(P.safe_path(pid, t["file"]), a, max(0, b - a)):
                vw.write(f)
            if t.get("last_is_anchor") and eref and eref[0] == "kf" and b == n:
                prev_end = tuple(eref)
            else:
                prev_end = ("seg", seg["id"])
        edl.append({"segment": seg["id"], "start": start_frame, "frames": vw.n - start_frame})
    vw.close()
    return {"file": rel, "frames": vw.n, "seconds": vw.n / fps, "skipped": skipped, "edl": edl}
