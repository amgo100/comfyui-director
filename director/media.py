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
        self.size = None  # (w, h) to force; otherwise the first frame decides

    def write(self, arr: np.ndarray):
        import av
        if self.st is None:
            w, h = self.size or (arr.shape[1], arr.shape[0])
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


def _take_frames(path, a, e, m, interpolated):
    """Output frames a..e (inclusive) of a take on the m-times frame grid. An interpolated file
    already has (n-1)*m+1 frames; otherwise frame j of the grid is source frame j // m (held)."""
    if e < a:
        return
    if interpolated or m == 1:
        yield from iter_frames(path, a, e - a + 1)
        return
    k0 = a // m
    for k, f in enumerate(iter_frames(path, k0, e // m - k0 + 1), start=k0):
        for _ in range(min(e, k * m + m - 1) - max(a, k * m) + 1):
            yield f


def render(pid, out_name=None, progress=None, final=False, cfg=None):
    """Streams the selected takes into one mp4.

    Draft: project size and fps, the takes as generated. Final: the finishing size and fps; each clip
    uses its finished version when it matches the current finishing settings, and is otherwise
    resized (and its frames held for the interpolation) with a note in `fallbacks`.

    Frame bookkeeping works on a grid of m output frames per source frame (m = 1 for draft):
    a take of n frames spans grid frames 0..(n-1)*m. A clip that starts on the frame the previous
    one ended on skips grid frame 0 only, so the interpolated frames between them are kept. A clip
    followed by a cut gets m-1 extra copies of its last frame, which keeps every clip exactly
    m times its draft length."""
    proj = P.load(pid)
    s = proj["settings"]
    fs = P.finish_settings(proj, cfg) if final else None
    if final:
        W, H, m = fs["width"], fs["height"], fs["interpolate"]
        vkey = P.finish_key(fs, "video", s.get("fast_video", True))
        skey = P.finish_key(fs, "still")
    else:
        W, H, m = int(s["width"]), int(s["height"]), 1
    fps = int(s["fps"]) * m
    out_name = out_name or time.strftime(("final_" if final else "render_") + "%Y%m%d_%H%M%S.mp4")
    rel = f"renders/{out_name}"
    vw = VideoWriter(P.safe_path(pid, rel), fps, crf=16)
    vw.size = (W, H)
    prev_end = None  # what the previously emitted segment ended on
    hold = None      # (frame, count) still owed to the previous clip if the next one is a cut
    skipped, fallbacks, edl = [], [], []

    def flush_hold():
        nonlocal hold
        if hold:
            for _ in range(hold[1]):
                vw.write(hold[0])
            if edl:
                edl[-1]["frames"] += hold[1]
        hold = None

    def fit(arr):
        if arr.shape[0] != H or arr.shape[1] != W:
            arr = np.asarray(fit_image(Image.fromarray(arr), W, H))
        return arr

    tl = proj["timeline"]
    for i, seg in enumerate(tl):
        if progress:
            progress(i, len(tl))
        if seg["kind"] == "still":
            try:
                img = P.kf_image(proj, seg.get("keyframe"))
            except P.NotReady as e:
                skipped.append({"segment": seg["id"], "why": str(e)})
                continue
            if final:
                f = P.finished_of(P.selected_take(proj["keyframes"][seg["keyframe"]]), skey)
                if f:
                    img = f["file"]
                elif fs["method"] != "resize":
                    fallbacks.append({"segment": seg["id"], "why": "still not finished"})
            arr = np.asarray(fit_image(Image.open(P.safe_path(pid, img)), W, H))
            n = int(seg["frames"])
            if _ref_eq(prev_end, ("kf", seg.get("keyframe"))):
                n -= 1
            flush_hold()
            start_frame = vw.n
            if n > 0:  # like a clip: the last frame's m-1 extra copies are owed, not written yet
                for _ in range((n - 1) * m + 1):
                    vw.write(arr)
                hold = (arr, m - 1) if m > 1 else None
            prev_end = ("kf", seg.get("keyframe"))
        else:
            t = P.selected_take(seg)
            if not t:
                skipped.append({"segment": seg["id"], "why": "no take"})
                prev_end = None
                continue
            n = int(t.get("frames") or probe(P.safe_path(pid, t["file"]))["frames"])
            path, interpolated = t["file"], False
            if final:
                f = P.finished_of(t, vkey)
                if f and int(f.get("frames", -1)) == (n - 1) * f.get("multiplier", 1) + 1:
                    path, interpolated = f["file"], True
                elif fs["method"] != "resize" or m > 1:
                    fallbacks.append({"segment": seg["id"], "why": "outdated" if t.get("finished") else "not finished"})
            trim_in, trim_out = int(seg.get("trim_in") or 0), int(seg.get("trim_out") or 0)
            a = trim_in * m
            sref, eref = t.get("start_ref"), t.get("end_ref")
            if trim_in == 0 and t.get("first_is_anchor") and sref and (
                    list(sref) == ["prev"] or _ref_eq(prev_end, sref)):
                a = 1                       # first frame repeats the previous clip's last
                hold = None                 # ... whose transition frames now come from this clip
            flush_hold()
            start_frame = vw.n
            lead = trim_out == 0 and t.get("last_is_anchor") and eref and list(eref) == ["next"]
            if lead:
                e, owed = (n - 1) * m - 1, 0   # the next clip starts on this clip's last frame
            else:
                e, owed = (n - 1 - trim_out) * m, m - 1
            last = None
            for fr in _take_frames(P.safe_path(pid, path), a, e, m, interpolated):
                last = fit(fr)
                vw.write(last)
            if last is not None and owed:
                hold = (last, owed)
            if t.get("last_is_anchor") and eref and eref[0] == "kf" and trim_out == 0:
                prev_end = tuple(eref)
            else:
                prev_end = ("seg", seg["id"])
        edl.append({"segment": seg["id"], "start": start_frame, "frames": vw.n - start_frame})
    flush_hold()
    vw.close()
    return {"file": rel, "frames": vw.n, "seconds": vw.n / fps, "fps": fps, "width": W, "height": H,
            "final": bool(final), "skipped": skipped, "fallbacks": fallbacks, "edl": edl}
