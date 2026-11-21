"""Project storage and timeline model for ComfyUI Director.

A project lives in  <ComfyUI output>/director/<project_id>/  and consists of
project.json plus media folders (keyframes/, clips/, masks/, imports/, renders/).

Model
-----
keyframes : {kf_id: Keyframe}   -- still images with a recipe and a list of takes
timeline  : [Segment, ...]      -- ordered; each segment is one shot/clip

Keyframe
  recipe: {op: t2i | edit | import | grab, prompt, parents: [kf_id], loras: [...]}
  takes : [{id, file, seed, op, ...}]   -- candidates; one is `selected`

Segment
  kind  : generate | video | still
  frames: length in frames (generate/still); video length comes from the take
  start : {type: none | keyframe | prev, keyframe}
  end   : {type: none | keyframe | next, keyframe}
  mode  : auto | i2v | flf2v | t2v | vace
  context_frames: frames of neighbouring clips given to VACE (motion continuity)
  takes : [{id, file, first, last, frames, first_is_anchor, last_is_anchor, finished, ...}]

Finishing
  A take can carry a `finished` version: the same frames upscaled to the delivery size (optionally
  refined with Wan and frame-interpolated). It is stored with the settings `key` it was made with,
  so changing the finishing settings marks it outdated instead of silently mixing versions.
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import uuid

_LOCK = threading.RLock()
_ROOT_OVERRIDE = None


def root_dir() -> str:
    if _ROOT_OVERRIDE:
        return _ROOT_OVERRIDE
    import folder_paths  # ComfyUI
    return os.path.join(folder_paths.get_output_directory(), "director")


def set_root(path: str):
    global _ROOT_OVERRIDE
    _ROOT_OVERRIDE = path


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


_PID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def project_dir(pid: str) -> str:
    if not _PID_RE.match(pid or ""):
        raise ValueError(f"bad project id: {pid!r}")
    return os.path.join(root_dir(), pid)


def safe_path(pid: str, rel: str) -> str:
    """Resolve a project-relative path and refuse anything outside the project."""
    base = os.path.realpath(project_dir(pid))
    full = os.path.realpath(os.path.join(base, rel))
    if not (full == base or full.startswith(base + os.sep)):
        raise ValueError(f"path escapes project: {rel!r}")
    return full


DEFAULT_SETTINGS = {
    "width": 832,
    "height": 480,
    "fps": 16,
    "max_segment_frames": 81,
    "keyframe_megapixels": 1.0,
    "style_prompt": "",
    "video_style_prompt": "",
    "fast_image": True,
    "fast_video": True,
    "takes_per_run": 2,
    "image_loras": [],   # project-wide (character / style)
    "image_reference_keyframes": [],  # keyframes passed as reference images to every keyframe generation
    "video_loras": [],
    "context_frames": 9,
    "image_model": "",   # "" = config default, "qwen21" or "qwen"
    "flf_engine": "i2v",  # first/last-frame clips: "i2v" (Wan 2.2 I2V) or "fun_inp" (Wan 2.2 Fun InP)
    "models": {},        # model slot overrides, see director/models.py
    "finish": {},        # finishing (upscale / refine / interpolate) overrides, see finish_settings()
}

SUBDIRS = ["keyframes", "clips", "masks", "imports", "renders"]


# --------------------------------------------------------------------------- io

def list_projects():
    r = root_dir()
    if not os.path.isdir(r):
        return []
    out = []
    for name in sorted(os.listdir(r)):
        pj = os.path.join(r, name, "project.json")
        if os.path.isfile(pj):
            try:
                with open(pj, encoding="utf-8") as f:
                    d = json.load(f)
                out.append({"id": name, "name": d.get("name", name), "updated": d.get("updated", 0)})
            except Exception:
                pass
    out.sort(key=lambda p: -p["updated"])
    return out


def create_project(name: str) -> dict:
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "-", name.strip()).strip("-")[:40] or "project"
    pid = slug
    i = 2
    while os.path.exists(project_dir(pid)):
        pid = f"{slug}-{i}"
        i += 1
    d = project_dir(pid)
    for s in SUBDIRS:
        os.makedirs(os.path.join(d, s), exist_ok=True)
    proj = {"id": pid, "name": name, "version": 1, "created": time.time(), "updated": time.time(),
            "settings": copy.deepcopy(DEFAULT_SETTINGS), "keyframes": {}, "timeline": []}
    save(proj)
    return proj


def load(pid: str) -> dict:
    with _LOCK:
        with open(os.path.join(project_dir(pid), "project.json"), encoding="utf-8") as f:
            proj = json.load(f)
    s = copy.deepcopy(DEFAULT_SETTINGS)
    s.update(proj.get("settings", {}))
    proj["settings"] = s
    return proj


def save(proj: dict):
    with _LOCK:
        proj["updated"] = time.time()
        d = project_dir(proj["id"])
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "project.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(proj, f, indent=1, ensure_ascii=False)
        os.replace(tmp, os.path.join(d, "project.json"))


class edit:
    """with edit(pid) as proj: ... -> atomically load, mutate, save."""

    def __init__(self, pid):
        self.pid = pid

    def __enter__(self):
        _LOCK.acquire()
        self.proj = load(self.pid)
        return self.proj

    def __exit__(self, et, ev, tb):
        try:
            if et is None:
                save(self.proj)
        finally:
            _LOCK.release()
        return False


# ------------------------------------------------------------------- entities

def new_keyframe(**fields) -> dict:
    kf = {"id": new_id("kf"), "name": "", "recipe": {"op": "t2i", "prompt": "", "parents": [], "loras": []},
          "takes": [], "selected": None, "created": time.time()}
    recipe = fields.pop("recipe", None)
    if recipe:
        kf["recipe"].update(recipe)
    kf.update(fields)
    return kf


def new_segment(**fields) -> dict:
    seg = {"id": new_id("seg"), "kind": "generate", "name": "", "frames": 81, "prompt": "",
           "start": {"type": "none", "keyframe": None}, "end": {"type": "none", "keyframe": None},
           "mode": "auto", "use_context": True, "motion": {"type": "none"}, "context_frames": None, "loras": [],
           "takes": [], "selected": None, "trim_in": 0, "trim_out": 0, "created": time.time()}
    for k in ("start", "end"):
        if k in fields and isinstance(fields[k], dict):
            seg[k].update(fields.pop(k))
    seg.update(fields)
    return seg


def find_segment(proj, sid):
    for i, s in enumerate(proj["timeline"]):
        if s["id"] == sid:
            return i, s
    raise KeyError(f"segment {sid} not found")


def selected_take(entity):
    sel = entity.get("selected")
    for t in entity.get("takes", []):
        if t["id"] == sel:
            return t
    return None


def add_take(pid: str, target: str, take: dict) -> dict:
    """Called by the save nodes. target = 'keyframe:<id>' or 'segment:<id>'."""
    kind, _, eid = target.partition(":")
    with edit(pid) as proj:
        if kind == "keyframe":
            ent = proj["keyframes"].get(eid)
            if ent is None:
                raise KeyError(f"keyframe {eid} not found")
        elif kind == "segment":
            _, ent = find_segment(proj, eid)
        else:
            raise ValueError(f"bad target {target!r}")
        take.setdefault("id", new_id("take"))
        take.setdefault("created", time.time())
        ent.setdefault("takes", []).append(take)
        if not ent.get("selected"):
            ent["selected"] = take["id"]
    return take


# ------------------------------------------------------------------- operations

KF_FIELDS = {"name", "recipe", "selected", "notes"}
SEG_FIELDS = {"name", "kind", "frames", "prompt", "start", "end", "mode", "use_context", "context_frames", "motion",
              "loras", "selected", "trim_in", "trim_out", "keyframe", "notes"}
SETTINGS_FIELDS = set(DEFAULT_SETTINGS)


def apply_op(pid: str, op: dict) -> dict:
    """Fine-grained edits so the UI never overwrites takes written by running jobs."""
    name = op.get("op")
    with edit(pid) as proj:
        result = {}
        if name == "settings.update":
            for k, v in op.get("fields", {}).items():
                if k in SETTINGS_FIELDS:
                    proj["settings"][k] = v
        elif name == "project.rename":
            proj["name"] = op["name"]
        elif name == "keyframe.create":
            kf = new_keyframe(**{k: v for k, v in op.get("fields", {}).items() if k in KF_FIELDS})
            proj["keyframes"][kf["id"]] = kf
            result["id"] = kf["id"]
        elif name == "keyframe.update":
            kf = proj["keyframes"][op["id"]]
            for k, v in op.get("fields", {}).items():
                if k == "recipe":
                    kf["recipe"].update(v)
                elif k in KF_FIELDS:
                    kf[k] = v
        elif name == "keyframe.delete":
            proj["keyframes"].pop(op["id"], None)
            refs = proj["settings"].get("image_reference_keyframes") or []
            proj["settings"]["image_reference_keyframes"] = [k for k in refs if k != op["id"]]
            for k in proj["keyframes"].values():
                k["recipe"]["parents"] = [p for p in k["recipe"].get("parents", []) if p != op["id"]]
            for s in proj["timeline"]:
                for end in ("start", "end"):
                    if s[end].get("keyframe") == op["id"]:
                        s[end] = {"type": "none", "keyframe": None}
                if s.get("keyframe") == op["id"]:
                    s["keyframe"] = None
        elif name == "segment.insert":
            fields = {k: v for k, v in op.get("fields", {}).items() if k in SEG_FIELDS}
            seg = new_segment(**fields)
            idx = op.get("index")
            if idx is None or idx > len(proj["timeline"]):
                idx = len(proj["timeline"])
            proj["timeline"].insert(idx, seg)
            result["id"] = seg["id"]
        elif name == "segment.update":
            _, seg = find_segment(proj, op["id"])
            for k, v in op.get("fields", {}).items():
                if k in ("start", "end") and isinstance(v, dict):
                    seg[k].update(v)
                elif k in SEG_FIELDS:
                    seg[k] = v
        elif name == "segment.delete":
            i, _ = find_segment(proj, op["id"])
            proj["timeline"].pop(i)
        elif name == "segment.move":
            i, seg = find_segment(proj, op["id"])
            proj["timeline"].pop(i)
            j = max(0, min(len(proj["timeline"]), op["index"]))
            proj["timeline"].insert(j, seg)
        elif name == "segment.split":
            i, seg = find_segment(proj, op["id"])
            a = max(5, int(seg["frames"]) // 2)
            b = new_segment(kind="generate", frames=max(5, int(seg["frames"]) - a + 1), prompt=seg["prompt"],
                            loras=copy.deepcopy(seg["loras"]), start={"type": "prev"}, end=copy.deepcopy(seg["end"]))
            seg["frames"] = a
            seg["end"] = {"type": "none", "keyframe": None}
            proj["timeline"].insert(i + 1, b)
            result["id"] = b["id"]
        elif name == "anchor.set":
            # put keyframe K on the boundary between segment i-1 and i
            idx = op["index"]
            kid = op.get("keyframe")
            tl = proj["timeline"]
            ref = {"type": "keyframe", "keyframe": kid} if kid else {"type": "none", "keyframe": None}
            if 0 <= idx - 1 < len(tl) and tl[idx - 1]["kind"] == "generate":
                tl[idx - 1]["end"] = dict(ref)
            if 0 <= idx < len(tl) and tl[idx]["kind"] == "generate":
                tl[idx]["start"] = dict(ref)
        elif name in ("keyframe.select", "segment.select"):
            ent = proj["keyframes"][op["id"]] if name.startswith("keyframe") else find_segment(proj, op["id"])[1]
            ent["selected"] = op["take"]
        elif name in ("keyframe.delete_take", "segment.delete_take"):
            ent = proj["keyframes"][op["id"]] if name.startswith("keyframe") else find_segment(proj, op["id"])[1]
            ent["takes"] = [t for t in ent["takes"] if t["id"] != op["take"]]
            if ent.get("selected") == op["take"]:
                ent["selected"] = ent["takes"][-1]["id"] if ent["takes"] else None
        else:
            raise ValueError(f"unknown op {name!r}")
        result["project"] = proj
        return result


# ------------------------------------------------------------- anchor resolution

class NotReady(Exception):
    pass


def kf_image(proj, kid):
    kf = proj["keyframes"].get(kid)
    if not kf:
        raise NotReady(f"keyframe {kid} missing")
    t = selected_take(kf)
    if not t:
        raise NotReady(f"keyframe '{kf.get('name') or kid}' has no selected take")
    return t["file"]


def seg_take(proj, seg):
    if seg["kind"] == "still":
        return None
    t = selected_take(seg)
    if not t:
        raise NotReady(f"segment '{seg.get('name') or seg['id']}' has no selected take")
    return t


def seg_last_frame(proj, seg):
    """(image file, video file or None, frame index) for the last used frame of a segment."""
    if seg["kind"] == "still":
        return kf_image(proj, seg["keyframe"]), None, None
    t = seg_take(proj, seg)
    n = int(t.get("frames", 0))
    idx = n - 1 - int(seg.get("trim_out") or 0)
    if not seg.get("trim_out") and t.get("last"):
        return t["last"], t["file"], idx
    return None, t["file"], idx


def seg_first_frame(proj, seg):
    if seg["kind"] == "still":
        return kf_image(proj, seg["keyframe"]), None, None
    t = seg_take(proj, seg)
    idx = int(seg.get("trim_in") or 0)
    if not idx and t.get("first"):
        return t["first"], t["file"], 0
    return None, t["file"], idx


def resolve(proj, seg_index):
    """Work out what a generate-segment needs. Returns dict describing inputs and mode."""
    tl = proj["timeline"]
    seg = tl[seg_index]
    s = proj["settings"]
    prev = tl[seg_index - 1] if seg_index > 0 else None
    nxt = tl[seg_index + 1] if seg_index + 1 < len(tl) else None
    st, en = seg["start"], seg["end"]
    use_ctx = seg.get("use_context", True)
    ctx_n = int(seg.get("context_frames") or s.get("context_frames") or 9)

    r = {"start_image": None, "end_image": None, "before": None, "after": None,
         "start_ref": None, "end_ref": None}

    if st["type"] == "keyframe" and st.get("keyframe"):
        r["start_image"] = {"file": kf_image(proj, st["keyframe"])}
        r["start_ref"] = ("kf", st["keyframe"])
    elif st["type"] == "prev":
        if prev is None:
            raise NotReady("'continue from previous' but this is the first segment")
        img, vid, idx = seg_last_frame(proj, prev)
        if use_ctx and vid is not None:
            r["before"] = {"file": vid, "end": idx, "count": ctx_n}
        elif img:
            r["start_image"] = {"file": img}
        else:
            r["start_image"] = {"video": vid, "frame": idx}
        r["start_ref"] = ("prev",)

    mtype_early = (seg.get("motion") or {}).get("type", "none")
    if mtype_early in ("camera", "control") and en["type"] != "none":
        r["note"] = f"the end frame is ignored with a {'camera move' if mtype_early == 'camera' else 'control video'}"
    elif en["type"] == "keyframe" and en.get("keyframe"):
        r["end_image"] = {"file": kf_image(proj, en["keyframe"])}
        r["end_ref"] = ("kf", en["keyframe"])
    elif en["type"] == "next":
        if nxt is None:
            raise NotReady("'lead into next' but this is the last segment")
        img, vid, idx = seg_first_frame(proj, nxt)
        if use_ctx and vid is not None:
            r["after"] = {"file": vid, "start": idx, "count": ctx_n}
        elif img:
            r["end_image"] = {"file": img}
        else:
            r["end_image"] = {"video": vid, "frame": idx}
        r["end_ref"] = ("next",)

    motion = seg.get("motion") or {}
    mtype = motion.get("type", "none")
    mode = seg.get("mode", "auto")
    if mtype == "camera":
        mode = "camera"          # Wan 2.2 Fun Camera: start frame + camera move
    elif mtype == "control":
        mode = "control"         # Wan 2.2 Fun Control: control video (+ start frame as reference)
    elif mode == "auto":
        if r["before"] or r["after"]:
            mode = "vace"
        elif r["start_image"] and r["end_image"]:
            mode = "flf2v"
        elif r["start_image"]:
            mode = "i2v"
        elif r["end_image"]:
            mode = "flf2v"
        else:
            mode = "t2v"
    if mode == "flf2v" and s.get("flf_engine") == "fun_inp":
        mode = "inp"
    if mode == "vace":
        # VACE takes single images as one-frame context too
        if r["start_image"] and not r["before"]:
            r["before"] = {"image": r["start_image"]}
        if r["end_image"] and not r["after"]:
            r["after"] = {"image": r["end_image"]}
    elif r["before"] or r["after"]:
        # non-VACE modes: fall back to boundary frames
        if r["before"]:
            b = r.pop("before")
            r["start_image"] = {"video": b["file"], "frame": b["end"]}
            r["before"] = None
        if r["after"]:
            a = r.pop("after")
            r["end_image"] = {"video": a["file"], "frame": a["start"]}
            r["after"] = None
    if mode in ("i2v", "camera") and not r["start_image"]:
        raise NotReady(("A camera move" if mode == "camera" else "Image→video") + " needs a start frame")
    if mode == "camera" and r["end_image"]:
        r["note"] = "the end frame is ignored with a camera move"
        r["end_image"], r["end_ref"] = None, None
    if mode == "control":
        src = motion.get("source") or {}
        if src.get("segment"):
            if src["segment"] == seg["id"]:
                raise NotReady("a clip cannot be its own control video")
            _, cseg = find_segment(proj, src["segment"])
            t = selected_take(cseg)
            if not t:
                raise NotReady(f"control clip '{cseg.get('name') or cseg['id']}' has no selected take")
            r["control"] = {"file": t["file"]}
        elif src.get("file"):
            r["control"] = {"file": src["file"]}
        else:
            raise NotReady("choose a control video")
        r["control"].update(offset=int(motion.get("offset") or 0), preprocess=motion.get("preprocess", "none"),
                            low=float(motion.get("canny_low", 0.4)), high=float(motion.get("canny_high", 0.8)))
        if r["end_image"]:
            r["note"] = "the end frame is ignored with a control video"
            r["end_image"], r["end_ref"] = None, None
    r["mode"] = mode
    return r


def wan_length(n: int) -> int:
    """Wan needs 4k+1 frames."""
    n = max(5, int(n))
    return ((n - 1 + 3) // 4) * 4 + 1


def keyframe_size(settings):
    """Keyframe resolution: video aspect ratio at ~keyframe_megapixels, multiples of 32."""
    w, h = settings["width"], settings["height"]
    mp = float(settings.get("keyframe_megapixels") or 1.0) * 1024 * 1024
    ar = w / h
    kh = (mp / ar) ** 0.5
    kw = kh * ar
    return int(round(kw / 32) * 32), int(round(kh / 32) * 32)


# -------------------------------------------------------------------- planning

def project_refs(proj, exclude=()):
    """Selected takes of the project-wide reference keyframes (skips ones without a take)."""
    files = []
    for kid in proj["settings"].get("image_reference_keyframes") or []:
        if kid in exclude:
            continue
        try:
            files.append(kf_image(proj, kid))
        except NotReady:
            pass
    return files


def keyframe_ready(proj, kf):
    rec = kf.get("recipe", {})
    if rec.get("op") in ("import", "grab"):
        return False, "imported"
    for p in rec.get("parents", []):
        try:
            kf_image(proj, p)
        except NotReady as e:
            return False, str(e)
    if rec.get("op") == "edit" and not rec.get("parents"):
        return False, "edit needs at least one source keyframe"
    if not (rec.get("prompt") or "").strip():
        return False, "no prompt"
    return True, ""


def status(proj):
    """Per-entity readiness for the UI and for auto-fill."""
    out = {"keyframes": {}, "segments": {}}
    for kid, kf in proj["keyframes"].items():
        ok, why = keyframe_ready(proj, kf)
        out["keyframes"][kid] = {"ready": ok, "why": why, "has_take": bool(kf.get("takes"))}
    for i, seg in enumerate(proj["timeline"]):
        if seg["kind"] != "generate":
            out["segments"][seg["id"]] = {"ready": False, "why": seg["kind"], "has_take": bool(seg.get("takes"))}
            continue
        try:
            r = resolve(proj, i)
            ok, why, mode = True, r.get("note", ""), r["mode"]
            if seg["frames"] > proj["settings"]["max_segment_frames"]:
                ok, why = False, "too long - split the segment"
            if not (seg.get("prompt") or "").strip() and mode in ("t2v",):
                ok, why = False, "text-to-video needs a prompt"
        except NotReady as e:
            ok, why, mode = False, str(e), None
        out["segments"][seg["id"]] = {"ready": ok, "why": why, "mode": mode, "has_take": bool(seg.get("takes"))}
    return out


def plan(proj):
    """Everything that can be generated right now and has no takes yet."""
    st = status(proj)
    jobs = []
    for kid, v in st["keyframes"].items():
        if v["ready"] and not v["has_take"]:
            jobs.append({"action": "keyframe.generate", "id": kid})
    for sid, v in st["segments"].items():
        if v["ready"] and not v["has_take"]:
            jobs.append({"action": "segment.generate", "id": sid})
    return jobs


# -------------------------------------------------------------------- finishing

FINISH_DEFAULTS = {
    "width": 0, "height": 0,      # 0 = derived from short_side and the project's aspect ratio
    "short_side": 1080,
    "method": "model",            # resize | model | refine
    "upscale_model": "4x-UltraSharp.pth",
    "interpolate": 1,             # 1 = off, 2..4 = FILM/RIFE multiplier (fps x multiplier)
    "interpolation_model": "",    # "" = config interpolation_model
    "refine": {
        "short_side": 720,        # Wan refines at this size, the upscale model takes it the rest of the way
        "denoise": 0.3,
        "model": "",              # "" = Wan text->video low-noise model
        "fast_lora": "",          # "" = that model's fast LoRA; "@none" = none
        "color_match": 1.0,
        "window": 81, "overlap": 8,
        "fast": {"steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
        "quality": {"steps": 10, "cfg": 3.5, "sampler": "euler", "scheduler": "simple"},
    },
}
FINISH_METHODS = ("resize", "model", "refine")


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif v is not None and v != "":   # empty = keep the default
            out[k] = v
    return out


def even(x):
    return max(2, int(round(x / 2)) * 2)


def size_for_short_side(settings, short):
    """Size with the project's aspect ratio and the given short side. Within 3% of 16:9 / 9:16
    it snaps to the standard delivery size (832x480 -> 1920x1080), cropping a sliver."""
    w, h = int(settings["width"]), int(settings["height"])
    short = int(short)
    ar = w / h
    if abs(ar - 16 / 9) / (16 / 9) < 0.03:
        return even(short * 16 / 9), short
    if abs(ar - 9 / 16) / (9 / 16) < 0.03:
        return short, even(short * 16 / 9)
    if w >= h:
        return even(short * ar), short
    return short, even(short / ar)


def finish_settings(proj, cfg=None):
    """Effective finishing settings: built-in defaults <- config.json 'finish' <- project settings."""
    fs = _merge(FINISH_DEFAULTS, (cfg or {}).get("finish") or {})
    fs = _merge(fs, proj["settings"].get("finish") or {})
    if fs.get("method") not in FINISH_METHODS:
        fs["method"] = "model"
    if not fs.get("interpolation_model"):
        fs["interpolation_model"] = (cfg or {}).get("interpolation_model", "film_net_fp16.safetensors")
    fs["interpolate"] = max(1, min(4, int(fs.get("interpolate") or 1)))
    if fs.get("width") and fs.get("height"):
        fs["width"], fs["height"] = even(fs["width"]), even(fs["height"])
    else:
        fs["width"], fs["height"] = size_for_short_side(proj["settings"], fs.get("short_side") or 1080)
    rw, rh = size_for_short_side(proj["settings"], fs["refine"].get("short_side") or 720)
    # Wan works on multiples of 16; never refine above the output size
    if rw * rh > fs["width"] * fs["height"]:
        rw, rh = fs["width"], fs["height"]
    fs["refine"]["width"], fs["refine"]["height"] = int(rw // 16 * 16), int(rh // 16 * 16)
    fs["fps"] = int(proj["settings"]["fps"]) * fs["interpolate"]
    return fs


def finish_key(fs, kind="video", fast_video=True):
    """Identifies what a finished file was made with. Stills ignore refine and interpolation."""
    k = {"w": fs["width"], "h": fs["height"], "m": fs["method"]}
    if fs["method"] != "resize":
        k["u"] = fs.get("upscale_model") or ""
    if kind == "video":
        k["i"] = fs["interpolate"]
        if fs["interpolate"] > 1:
            k["im"] = fs["interpolation_model"]
        if fs["method"] == "refine":
            r = fs["refine"]
            k["r"] = [r["width"], r["height"], float(r["denoise"]), r.get("model") or "", r.get("fast_lora") or "",
                      bool(fast_video), float(r.get("color_match", 1.0))]
    elif fs["method"] == "refine":
        k["m"] = "model"
    return json.dumps(k, sort_keys=True)


def finished_of(take, key):
    f = (take or {}).get("finished")
    return f if f and f.get("key") == key else None


def set_finished(pid: str, target: str, info: dict):
    """target = 'finish:<segment|keyframe>:<entity id>:<take id>'."""
    parts = target.split(":")
    if len(parts) != 4 or parts[0] != "finish":
        raise ValueError(f"bad finish target {target!r}")
    _, kind, eid, tid = parts
    with edit(pid) as proj:
        ent = proj["keyframes"].get(eid) if kind == "keyframe" else find_segment(proj, eid)[1]
        if ent is None:
            raise KeyError(f"{kind} {eid} not found")
        take = next((t for t in ent.get("takes", []) if t["id"] == tid), None)
        if take is None:
            raise KeyError(f"take {tid} not found")
        old = take.get("finished") or {}
        take["finished"] = dict(info, created=time.time())
    for k in ("file", "first", "last"):  # the replaced version is not referenced anywhere else
        if old.get(k) and old[k] not in (info.get("file"), info.get("first"), info.get("last")):
            try:
                os.remove(safe_path(pid, old[k]))
            except OSError:
                pass
    return take


def finish_items(proj, cfg=None):
    """Everything the final render needs: the selected take of every clip, and of every keyframe
    held as a still. Each item: {kind, id, take, name, state: ok|outdated|missing|no_take}."""
    fs = finish_settings(proj, cfg)
    fast = proj["settings"].get("fast_video", True)
    vkey, skey = finish_key(fs, "video", fast), finish_key(fs, "still")
    items, seen = [], set()
    # with 'resize only' the render resizes on the fly: stills never need finishing, clips only to interpolate
    need_stills = fs["method"] != "resize"
    need_clips = fs["method"] != "resize" or fs["interpolate"] > 1
    for seg in proj["timeline"]:
        if (seg["kind"] == "still" and not need_stills) or (seg["kind"] != "still" and not need_clips):
            continue
        if seg["kind"] == "still":
            kid = seg.get("keyframe")
            if not kid or kid in seen or kid not in proj["keyframes"]:
                continue
            seen.add(kid)
            kf = proj["keyframes"][kid]
            t = selected_take(kf)
            kind, eid, name, key = "keyframe", kid, kf.get("name") or kid, skey
        else:
            t = selected_take(seg)
            kind, eid, name, key = "segment", seg["id"], seg.get("name") or seg["id"], vkey
        if not t:
            state = "no_take"
        elif finished_of(t, key):
            state = "ok"
        elif t.get("finished"):
            state = "outdated"
        else:
            state = "missing"
        items.append({"kind": kind, "id": eid, "take": t["id"] if t else None, "name": name, "state": state})
    return items


def finish_status(proj, cfg=None):
    fs = finish_settings(proj, cfg)
    items = finish_items(proj, cfg)
    count = {s: sum(1 for i in items if i["state"] == s) for s in ("ok", "outdated", "missing", "no_take")}
    return {"settings": fs, "items": items, "count": count,
            "video_key": finish_key(fs, "video", proj["settings"].get("fast_video", True)),
            "still_key": finish_key(fs, "still")}
