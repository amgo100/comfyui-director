"""Render bookkeeping tests (no ComfyUI needed): python tests/test_render.py  (or pytest)

Synthetic clips encode (clip, frame) in their colour. The draft render must match the frame
sequence the boundary rules describe; the final render with interpolated finished versions must be
exactly m times as long, with every m-th frame equal to the draft and the frames between them
interpolated within a clip (or holding the last frame at a cut)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from director import media, project as P  # noqa: E402

W, H, N = 64, 48, 9


def col(v):
    return np.full((H, W, 3), v, np.uint8)


def frames_of(vals):
    return [col(v) for v in vals]


def means(path):
    return [tuple(int(round(x)) for x in f.reshape(-1, 3).mean(0)) for f in media.iter_frames(path)]


def close(a, b, tol=4):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def make_project(m=2):
    root = tempfile.mkdtemp(prefix="director_test_")
    P.set_root(root)
    proj = P.create_project("t")
    pid = proj["id"]
    s = proj["settings"]
    s.update(width=W, height=H, fps=8, finish={"method": "model", "interpolate": m, "width": W * 2, "height": H * 2,
                                               "upscale_model": "x.pth"})
    # clip values: each frame k of clip c gets grey level base_c + 6k
    A = [20 + 6 * k for k in range(N)]
    B = [A[-1]] + [A[-1] + 6 * k for k in range(1, N)]          # starts on A's last frame
    D = [200 - 6 * k for k in range(N)]
    C = [110 + 6 * k for k in range(N - 1)] + [D[0]]              # ends on D's first frame (lead into next)
    K = 15                                                         # keyframe colour of the still
    Fv = [K] + [K + 7 * k for k in range(1, N)]                    # starts on the still's keyframe
    vals = {"A": A, "B": B, "C": C, "D": D, "F": Fv}
    takes = {}
    for name, v in vals.items():
        takes[name] = media.write_take_files(pid, f"clips/{name}", "t", frames_of(v), 8)
    os.makedirs(P.safe_path(pid, "keyframes/kfK"), exist_ok=True)
    media.save_png(col(K), P.safe_path(pid, "keyframes/kfK/t.png"))
    proj["keyframes"]["kfK"] = P.new_keyframe(id="kfK", recipe={"op": "import"},
                                              takes=[{"id": "tk", "file": "keyframes/kfK/t.png"}], selected="tk")

    def seg(name, **extra):
        t = dict(takes[name], id=f"take{name}")
        t.update(extra)
        return P.new_segment(id=name, frames=N, takes=[t], selected=t["id"])
    tl = [seg("A"),
          seg("B", first_is_anchor=True, start_ref=["prev"]),
          seg("C", last_is_anchor=True, end_ref=["next"]),
          seg("D"),
          P.new_segment(id="E", kind="still", keyframe="kfK", frames=3),
          seg("F", first_is_anchor=True, start_ref=["kf", "kfK"])]
    tl[3]["trim_in"], tl[3]["trim_out"] = 2, 1
    proj["timeline"] = tl
    P.save(proj)
    return pid, vals, K


def expected_draft(vals, K):
    A, B, C, D, Fv = (vals[k] for k in "ABCDF")
    return A + B[1:] + C[:-1] + D[2:N - 1] + [K] * 3 + Fv[1:]


def test_draft():
    pid, vals, K = make_project()
    r = media.render(pid)
    got = [g[0] for g in means(P.safe_path(pid, r["file"]))]
    exp = expected_draft(vals, K)
    assert len(got) == len(exp) == r["frames"], (len(got), len(exp))
    assert all(abs(a - b) <= 3 for a, b in zip(got, exp)), list(zip(got, exp))
    assert sum(e["frames"] for e in r["edl"]) == r["frames"]
    return r


def finish_all(pid, m):
    """Fake finishing: 2x size, linear interpolation (what FrameInterpolate's frame grid looks like)."""
    proj = P.load(pid)
    fs = P.finish_settings(proj, {})
    vkey, skey = P.finish_key(fs, "video", True), P.finish_key(fs, "still")
    for seg in proj["timeline"]:
        if seg["kind"] == "still":
            continue
        t = P.selected_take(seg)
        src = list(media.iter_frames(P.safe_path(pid, t["file"])))
        outf = []
        for k in range(len(src) - 1):
            for j in range(m):
                a = src[k].astype(np.float32) * (1 - j / m) + src[k + 1].astype(np.float32) * (j / m)
                outf.append(np.kron(a, np.ones((2, 2, 1))).astype(np.uint8))
        outf.append(np.kron(src[-1], np.ones((2, 2, 1), np.uint8)))
        files = media.write_take_files(pid, f"clips/{seg['id']}", "t_fin", outf, 8 * m)
        P.set_finished(pid, f"finish:segment:{seg['id']}:{t['id']}", dict(files, key=vkey, multiplier=m))
    from PIL import Image
    kimg = np.asarray(Image.open(P.safe_path(pid, "keyframes/kfK/t.png")))
    media.save_png(np.kron(kimg, np.ones((2, 2, 1), np.uint8)), P.safe_path(pid, "keyframes/kfK/t_fin.png"))
    P.set_finished(pid, "finish:keyframe:kfK:tk", {"file": "keyframes/kfK/t_fin.png", "key": skey,
                                                     "width": W * 2, "height": H * 2})
    return vkey


def check_final(pid, m, vals, K, interpolated=True):
    r = media.render(pid, final=True, cfg={})
    assert (r["width"], r["height"], r["fps"]) == (W * 2, H * 2, 8 * m), r
    got = [g[0] for g in means(P.safe_path(pid, r["file"]))]
    draft = expected_draft(vals, K)
    assert len(got) == r["frames"] == m * len(draft), (len(got), m * len(draft))
    # clip boundaries in draft frames, and whether the next clip continues the motion
    A, B, C, D, Fv = (vals[k] for k in "ABCDF")
    cuts = {len(A + B[1:] + C[:-1] + D[2:N - 1]) - 1,        # D -> still: cut
            len(A + B[1:] + C[:-1] + D[2:N - 1]) + 2}         # still -> F: F starts on the still (same image)
    c_end = len(A + B[1:] + C[:-1]) - 1                       # C leads into D's frame 0, but D is trimmed by 2:
    cuts.add(len(draft) - 1)                                  # C's in-betweens still head for C's own last frame
    for j, v in enumerate(got):
        k, ph = divmod(j, m)
        if ph == 0 or not interpolated:
            exp = draft[k]
        elif k == c_end:
            exp = C[-2] + (C[-1] - C[-2]) * ph / m
        elif k in cuts:
            exp = draft[k]
        else:
            exp = draft[k] + (draft[k + 1] - draft[k]) * ph / m
        assert abs(v - exp) <= 4, (j, v, exp)
    assert sum(e["frames"] for e in r["edl"]) == r["frames"]
    return r


def test_final_interpolated():
    for m in (1, 2, 3):
        pid, vals, K = make_project(m)
        finish_all(pid, m)
        r = check_final(pid, m, vals, K)
        assert not r["fallbacks"], r["fallbacks"]


def test_final_fallback_holds_frames():
    pid, vals, K = make_project(2)
    r = check_final(pid, 2, vals, K, interpolated=False)   # nothing finished: resized, frames held
    assert len(r["fallbacks"]) == 6


def test_outdated_after_settings_change():
    pid, vals, K = make_project(2)
    finish_all(pid, 2)
    st = P.finish_status(P.load(pid), {})
    assert st["count"]["ok"] == 6, st["count"]
    P.apply_op(pid, {"op": "settings.update", "fields": {"finish": {"method": "model", "interpolate": 2,
                                                                     "width": W * 4, "height": H * 4}}})
    st = P.finish_status(P.load(pid), {})
    assert st["count"]["outdated"] == 6, st["count"]


def test_sizes():
    mk = lambda w, h: {"settings": dict(P.DEFAULT_SETTINGS, width=w, height=h)}  # noqa: E731
    assert P.finish_settings(mk(832, 480))[("width")] == 1920
    assert (P.finish_settings(mk(832, 480))["height"]) == 1080
    assert (lambda f: (f["width"], f["height"]))(P.finish_settings(mk(480, 832))) == (1080, 1920)
    assert (lambda f: (f["width"], f["height"]))(P.finish_settings(mk(1280, 720))) == (1920, 1080)
    assert (lambda f: (f["width"], f["height"]))(P.finish_settings(mk(640, 640))) == (1080, 1080)
    f = P.finish_settings(mk(832, 480))
    assert (f["refine"]["width"], f["refine"]["height"]) == (1280, 720)
    f = P.finish_settings(mk(960, 544))
    assert (f["width"], f["height"]) == (1920, 1080)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
