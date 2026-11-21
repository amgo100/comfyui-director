"""HTTP API used by the Director panel (mounted on ComfyUI's own server under /director)."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import traceback

from aiohttp import web

from . import graphs, media
from . import models as M
from . import project as P

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def load_config():
    with open(os.path.join(PACK_DIR, "config.example.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    user = os.path.join(PACK_DIR, "config.json")
    if os.path.isfile(user):
        with open(user, encoding="utf-8") as f:
            _deep_merge(cfg, json.load(f))
    return cfg


def _ok(**kw):
    return web.json_response(kw)


def _err(e, status=400):
    return web.json_response({"error": str(e)}, status=status)


def _full(pid, extra=None):
    proj = P.load(pid)
    cfg = load_config()
    d = {"project": proj, "status": P.status(proj, M.effective_config(cfg, proj["settings"])),
         "finish": P.finish_status(proj, cfg)}
    if extra:
        d.update(extra)
    return d


def register(routes):
    @routes.get("/director/config")
    async def get_config(request):
        cfg = load_config()
        try:
            import folder_paths
            cfg["_loras"] = folder_paths.get_filename_list("loras")
        except Exception:
            cfg["_loras"] = []
        return web.json_response(cfg)

    @routes.get("/director/models")
    async def get_models(request):
        """Model slots with their config defaults, plus the files ComfyUI can see, for the Settings dialog."""
        cfg = load_config()
        return web.json_response({"slots": M.slot_info(cfg), "groups": M.GROUPS, "dtypes": M.DTYPES,
                                  "image_model": cfg.get("image_model", "qwen21"), "files": M.available(),
                                  "presets": M.PRESETS, "finish_defaults": P.finish_settings(
                                      {"settings": dict(P.DEFAULT_SETTINGS, finish={})}, cfg)})

    @routes.get("/director/projects")
    async def list_projects(request):
        return _ok(projects=P.list_projects())

    @routes.post("/director/projects")
    async def create_project(request):
        body = await request.json()
        proj = P.create_project(body.get("name") or "Untitled")
        return web.json_response(_full(proj["id"]))

    @routes.get("/director/project/{pid}")
    async def get_project(request):
        try:
            return web.json_response(_full(request.match_info["pid"]))
        except FileNotFoundError:
            return _err("project not found", 404)

    @routes.post("/director/project/{pid}/op")
    async def op(request):
        pid = request.match_info["pid"]
        try:
            body = await request.json()
            ops = body if isinstance(body, list) else [body]
            res = {}
            cfg = load_config()
            for o in ops:
                res = P.apply_op(pid, o, M.effective_config(cfg, P.load(pid)["settings"]))
            res.pop("project", None)
            return web.json_response(_full(pid, {"result": res}))
        except Exception as e:
            traceback.print_exc()
            return _err(e)

    @routes.post("/director/project/{pid}/build")
    async def build(request):
        pid = request.match_info["pid"]
        try:
            req = await request.json()
            proj = P.load(pid)
            jobs = graphs.build(load_config(), proj, req)
            return _ok(jobs=jobs)
        except P.NotReady as e:
            return _err(f"Not ready: {e}")
        except Exception as e:
            traceback.print_exc()
            return _err(e)

    @routes.get("/director/project/{pid}/plan")
    async def plan(request):
        """Next auto-fill step: one model group (see graphs.plan_step). ?stale=1 includes stale takes,
        ?loaded=<model key> is what the last finished job had loaded."""
        pid = request.match_info["pid"]
        try:
            q = request.rel_url.query
            return _ok(**graphs.plan_step(load_config(), P.load(pid), stale=q.get("stale") == "1",
                                          loaded=q.get("loaded", ""),
                                          skip=set(filter(None, q.get("skip", "").split(",")))))
        except Exception as e:
            traceback.print_exc()
            return _err(e)

    @routes.post("/director/project/{pid}/upload")
    async def upload(request):
        pid = request.match_info["pid"]
        try:
            data = await request.post()
            f = data["file"]
            purpose = data.get("purpose", "keyframe")
            ext = os.path.splitext(f.filename or "")[1].lower() or ".png"
            uid = P.new_id("up")
            if purpose == "mask":
                rel = f"masks/{uid}.png"
            else:
                rel = f"imports/{uid}{ext}"
            path = P.safe_path(pid, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as out:
                out.write(f.file.read())
            result = {"path": rel}
            if purpose == "keyframe":
                kf = P.new_keyframe(name=data.get("name") or os.path.splitext(f.filename or "image")[0],
                                    recipe={"op": "import", "prompt": ""})
                kf["takes"] = [{"id": P.new_id("take"), "file": rel, "op": "import"}]
                kf["selected"] = kf["takes"][0]["id"]
                with P.edit(pid) as proj:
                    proj["keyframes"][kf["id"]] = kf
                result["keyframe"] = kf["id"]
            elif purpose == "control":
                proj = P.load(pid)
                loop = asyncio.get_running_loop()
                files = await loop.run_in_executor(None, media.import_video, pid, path, proj["settings"], uid + "_ctrl")
                result["path"] = files["file"]
                result["frames"] = files["frames"]
            elif purpose == "video":
                proj = P.load(pid)
                loop = asyncio.get_running_loop()
                files = await loop.run_in_executor(None, media.import_video, pid, path, proj["settings"], uid + "_n")
                take = dict(files, id=P.new_id("take"), op="import")
                idx = data.get("index")
                res = P.apply_op(pid, {"op": "segment.insert", "index": int(idx) if idx not in (None, "") else None,
                                       "fields": {"kind": "video", "name": os.path.splitext(f.filename or "clip")[0],
                                                  "frames": files["frames"]}})
                P.add_take(pid, f"segment:{res['id']}", take)
                result["segment"] = res["id"]
            return web.json_response(_full(pid, {"result": result}))
        except Exception as e:
            traceback.print_exc()
            return _err(e)

    @routes.post("/director/project/{pid}/grab")
    async def grab(request):
        """Turn a frame of a clip into a keyframe (for I2I, continuation, or re-use)."""
        pid = request.match_info["pid"]
        try:
            body = await request.json()
            proj = P.load(pid)
            _, seg = P.find_segment(proj, body["segment"])
            take = next(t for t in seg["takes"] if t["id"] == body.get("take", seg["selected"]))
            kf = P.new_keyframe(name=body.get("name") or f"{seg.get('name') or 'clip'} @ {body['frame']}",
                                recipe={"op": "grab", "prompt": "", "source": {
                                    "segment": seg["id"], "take": take["id"], "frame": int(body["frame"])}})
            tid = P.new_id("take")
            rel = f"keyframes/{kf['id']}/{tid}.png"
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, media.grab_frame, pid, take["file"], int(body["frame"]), rel)
            kf["takes"] = [{"id": tid, "file": rel, "op": "grab"}]
            kf["selected"] = tid
            with P.edit(pid) as p:
                p["keyframes"][kf["id"]] = kf
            return web.json_response(_full(pid, {"result": {"keyframe": kf["id"]}}))
        except Exception as e:
            traceback.print_exc()
            return _err(e)

    @routes.post("/director/project/{pid}/render")
    async def render(request):
        pid = request.match_info["pid"]
        try:
            body = await request.json() if request.can_read_body else {}
            final = bool((body or {}).get("final"))
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, lambda: media.render(pid, final=final, cfg=load_config()))
            return _ok(**res)
        except Exception as e:
            traceback.print_exc()
            return _err(e)

    @routes.get("/director/file/{pid}/{path:.*}")
    async def get_file(request):
        try:
            full = P.safe_path(request.match_info["pid"], request.match_info["path"])
        except ValueError as e:
            return _err(e, 403)
        if not os.path.isfile(full):
            return _err("not found", 404)
        return web.FileResponse(full, headers={"Cache-Control": "max-age=31536000, immutable"})
