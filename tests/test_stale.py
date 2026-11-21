"""Stale tracking and model-grouped auto-fill (no ComfyUI needed): python tests/test_stale.py

Jobs are simulated: the real graph is built, and a take is saved with the metadata its save node
would write (inputs snapshot, select flag)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from director import graphs, project as P, routes  # noqa: E402

CFG = routes.load_config()


def new_project():
    P.set_root(tempfile.mkdtemp(prefix="director_stale_"))
    return P.create_project("t")["id"]


def run(pid, action, eid, select=False):
    """Build the job like the panel does and 'execute' it: add the take its save node would add."""
    proj = P.load(pid)
    job = graphs.build(CFG, proj, {"action": action, "id": eid, "count": 1, "select": select})[0]
    save = next(n for n in job["prompt"].values() if n["class_type"].startswith("DirectorSave"))
    meta = json.loads(save["inputs"]["meta"])
    tid = P.new_id("take")
    meta.update(id=tid, file=f"x/{tid}.mp4", first=f"x/{tid}_f.png", last=f"x/{tid}_l.png", frames=81)
    P.add_take(pid, job["target"], meta)
    return tid


def st(pid):
    proj = P.load(pid)
    return P.status(proj, CFG)


def build_chain(pid):
    """K1 (t2i) -> K2 (edit of K1); clip A: K1 -> K2; clip B continues A; clip C: t2v."""
    r = P.apply_op(pid, {"op": "keyframe.create", "fields": {"name": "K1", "recipe": {"op": "t2i", "prompt": "a lighthouse"}}})
    k1 = r["id"]
    r = P.apply_op(pid, {"op": "keyframe.create", "fields": {"name": "K2", "recipe": {"op": "edit", "prompt": "at night", "parents": [k1]}}})
    k2 = r["id"]
    a = P.apply_op(pid, {"op": "segment.insert", "fields": {"name": "A", "prompt": "waves", "start": {"type": "keyframe", "keyframe": k1},
                                                            "end": {"type": "keyframe", "keyframe": k2}}})["id"]
    b = P.apply_op(pid, {"op": "segment.insert", "fields": {"name": "B", "prompt": "more waves", "start": {"type": "prev"}, "use_context": False}})["id"]
    c = P.apply_op(pid, {"op": "segment.insert", "fields": {"name": "C", "prompt": "gulls over the sea"}})["id"]
    return k1, k2, a, b, c


def test_stale_cascade():
    pid = new_project()
    k1, k2, a, b, c = build_chain(pid)
    run(pid, "keyframe.generate", k1)
    run(pid, "keyframe.generate", k2)
    run(pid, "segment.generate", a)
    run(pid, "segment.generate", b)
    run(pid, "segment.generate", c)
    s = st(pid)
    assert not any(v["stale"] for v in list(s["keyframes"].values()) + list(s["segments"].values())), s
    take = P.selected_take(P.find_segment(P.load(pid), a)[1])
    assert take["inputs"] == {f"kf:{k1}": P.selected_take(P.load(pid)["keyframes"][k1])["id"],
                              f"kf:{k2}": P.selected_take(P.load(pid)["keyframes"][k2])["id"]}, take["inputs"]

    # a second take of K1, selected by hand: K2 and A are stale, B is downstream, C untouched
    run(pid, "keyframe.generate", k1)
    proj = P.load(pid)
    P.apply_op(pid, {"op": "keyframe.select", "id": k1, "take": proj["keyframes"][k1]["takes"][-1]["id"]})
    s = st(pid)
    assert s["keyframes"][k2]["stale"] == ["keyframe 'K1' has a different take selected"], s["keyframes"][k2]
    assert len(s["segments"][a]["stale"]) == 1 and s["segments"][b].get("downstream") and not s["segments"][b]["stale"]
    assert not s["segments"][c]["stale"] and not s["segments"][c].get("downstream")

    # the update run: stale jobs only for ready + stale items, with select
    jobs = P.plan(P.load(pid), CFG, stale=True)
    assert {(j["id"], j.get("select")) for j in jobs} == {(k2, True), (a, True)}, jobs
    assert P.plan(P.load(pid), CFG) == []          # plain auto-fill: no gaps
    run(pid, "keyframe.generate", k2, select=True)  # replaces K2's selection -> A still stale (K2 changed too)
    s = st(pid)
    assert not s["keyframes"][k2]["stale"] and len(s["segments"][a]["stale"]) == 2, s["segments"][a]
    run(pid, "segment.generate", a, select=True)
    s = st(pid)
    assert not s["segments"][a]["stale"] and s["segments"][b]["stale"] == ["clip 'A' has a different take selected"]
    run(pid, "segment.generate", b, select=True)
    s = st(pid)
    assert not any(v["stale"] or v.get("downstream") for v in list(s["keyframes"].values()) + list(s["segments"].values()))
    assert len(P.find_segment(P.load(pid), a)[1]["takes"]) == 2   # old takes are kept


def test_anchor_swap_keep_and_inpaint():
    pid = new_project()
    k1, k2, a, b, c = build_chain(pid)
    for kind, eid in (("keyframe", k1), ("keyframe", k2), ("segment", a)):
        run(pid, f"{kind}.generate", eid)
    P.apply_op(pid, {"op": "segment.update", "id": a, "fields": {"end": {"type": "none", "keyframe": None}}})
    assert st(pid)["segments"][a]["stale"] == ["no longer uses keyframe 'K2'"]
    P.apply_op(pid, {"op": "segment.keep", "id": a}, CFG)
    assert not st(pid)["segments"][a]["stale"]
    # an inpainted keyframe take keeps its parent's inputs, so K2 stays up to date ...
    proj = P.load(pid)
    t = P.selected_take(proj["keyframes"][k2])
    job = graphs.build(CFG, proj, {"action": "keyframe.inpaint", "id": k2, "take": t["id"], "mask": "masks/m.png", "count": 1})[0]
    meta = json.loads(next(n for n in job["prompt"].values() if n["class_type"] == "DirectorSaveImage")["inputs"]["meta"])
    assert meta["inputs"] == t["inputs"]
    # ... and selecting it makes clips built from K2 stale
    P.apply_op(pid, {"op": "segment.update", "id": a, "fields": {"end": {"type": "keyframe", "keyframe": k2}}})
    P.apply_op(pid, {"op": "segment.keep", "id": a}, CFG)
    P.add_take(pid, f"keyframe:{k2}", dict(meta, id="take_inp", file="x.png", select=True))
    s = st(pid)
    assert not s["keyframes"][k2]["stale"] and s["segments"][a]["stale"] == ["keyframe 'K2' has a different take selected"]


def test_migration_records_current_inputs():
    pid = new_project()
    k1, k2, a, b, c = build_chain(pid)
    for kind, eid in (("keyframe", k1), ("keyframe", k2), ("segment", a)):
        run(pid, f"{kind}.generate", eid)
    proj = P.load(pid)
    for t in [*(x for k in proj["keyframes"].values() for x in k["takes"]), *P.find_segment(proj, a)[1]["takes"]]:
        t.pop("inputs", None)
    proj["version"] = 1
    P.save(proj)
    proj = P.load(pid)  # migrates
    assert proj["version"] == 2 and "inputs" in P.selected_take(P.find_segment(proj, a)[1])
    assert P.load(pid)["version"] == 2  # saved
    assert not any(v["stale"] for v in st(pid)["segments"].values())


def test_model_grouping():
    pid = new_project()
    k1, k2, a, b, c = build_chain(pid)
    d = P.apply_op(pid, {"op": "segment.insert", "fields": {"name": "D", "prompt": "storm"}})["id"]
    # step 1: nothing loaded -> keyframes first (only K1 is ready)
    step = graphs.plan_step(CFG, P.load(pid))
    assert [j["id"] for j in step["jobs"]] == [k1], step
    assert {g["label"]: g["count"] for g in step["groups"]} == {"keyframes": 1, "text→video": 2}, step["groups"]
    # with the t2v weights loaded, the t2v clips go first even though a keyframe is ready
    t2v_key = step["groups"][1]["key"]
    assert "t2v_high" in t2v_key and "t2v_low" in t2v_key
    step = graphs.plan_step(CFG, P.load(pid), loaded=t2v_key)
    assert sorted(j["id"] for j in step["jobs"]) == sorted([c, d]) and step["group"] == "text→video"
    run(pid, "keyframe.generate", k1)
    run(pid, "keyframe.generate", k2)
    # Qwen still loaded: nothing left for it -> keyframes gone, largest group is t2v (2) over i2v (1)
    qwen_key = graphs.model_key(graphs.build(CFG, P.load(pid), {"action": "keyframe.generate", "id": k1, "count": 1})[0]["prompt"])
    step = graphs.plan_step(CFG, P.load(pid), loaded=qwen_key)
    assert step["group"] == "text→video" and len(step["jobs"]) == 2, step
    assert {g["label"] for g in step["groups"]} == {"first/last frame", "text→video"}
    # stale run skips ids already regenerated in this run
    run(pid, "segment.generate", a)
    run(pid, "keyframe.generate", k1)
    proj = P.load(pid)
    P.apply_op(pid, {"op": "keyframe.select", "id": k1, "take": proj["keyframes"][k1]["takes"][-1]["id"]})
    ids = {j["id"] for j in graphs.plan_step(CFG, P.load(pid), stale=True, loaded=qwen_key)["jobs"]}
    assert ids == {k2}, ids
    ids = {j["id"] for j in graphs.plan_step(CFG, P.load(pid), stale=True, loaded=qwen_key, skip={k2})["jobs"]}
    assert k2 not in ids


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
