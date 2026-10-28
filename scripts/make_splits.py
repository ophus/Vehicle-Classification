#!/usr/bin/env python3
"""Create the frozen train/val/test split -> <data_root>/splits/splits.json

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --block-frames 8 --guard 2 --seed 42
    python scripts/make_splits.py --overwrite        # ONLY if you accept that the test set changes

Rules (why):
  * Only frames marked "reviewed": true are used. An un-reviewed frame with no boxes is NOT a
    verified empty frame, so it must never become a negative example. JSON without the key = reviewed.
  * >= 3 sessions: whole sessions are assigned to splits (best protection against leakage).
  * < 3 sessions: each session is cut into blocks of --block-frames consecutive frames and whole
    blocks are assigned to splits. Neighbouring frames are nearly identical (static camera), so at every
    boundary between different splits, --guard frames are dropped from the lower-priority side
    (train < val < test) to keep val/test intact. This lowers, but does not remove, leakage.
  * The split is frozen: re-creating it with different data changes the test set and invalidates
    earlier results, hence --overwrite is required.
Ratios come from configs/default.json ("split"). Exit codes: 0 ok, 2 error.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.config import data_root, load_config  # noqa: E402

PRIORITY = {"train": 0, "val": 1, "test": 2}


def frame_number(fid):
    m = re.search(r"(\d+)$", fid)
    return int(m.group(1)) if m else 0


def load_sessions(ann_dir):
    """-> (sessions: {sid: [frame dicts sorted by time]}, excluded_unreviewed: {sid: [ids]})."""
    sessions, excluded = {}, {}
    for p in sorted(Path(ann_dir).glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        sid = doc["session_id"]
        frames = [f for f in doc["frames"] if f.get("reviewed", True)]
        skipped = [f["frame_id"] for f in doc["frames"] if not f.get("reviewed", True)]
        if skipped:
            excluded[sid] = skipped
        if frames:
            sessions[sid] = sorted(frames, key=lambda f: (frame_number(f["frame_id"]), f["frame_id"]))
    return sessions, excluded


def greedy_assign(sizes, ratios, rng):
    """Assign units (given by sizes) to splits so the frame share follows `ratios`."""
    order = list(range(len(sizes)))
    rng.shuffle(order)
    total = float(sum(sizes))
    got = {s: 0.0 for s in ratios}
    out = [None] * len(sizes)
    for u in order:
        # largest remaining deficit; ties resolved train < val < test
        s = max(ratios, key=lambda k: (ratios[k] * total - got[k], -PRIORITY[k]))
        out[u] = s
        got[s] += sizes[u]
    return out


def apply_guard(seq, guard):
    """seq: list of [frame_id, split] in time order. Return ids to drop at split boundaries."""
    dropped = set()
    for i in range(len(seq) - 1):
        a, b = seq[i][1], seq[i + 1][1]
        if a == b:
            continue
        if PRIORITY[a] < PRIORITY[b]:           # earlier side has lower priority -> drop backwards
            for k in range(i, max(-1, i - guard), -1):
                if seq[k][1] != a:
                    break
                dropped.add(seq[k][0])
        else:                                    # later side has lower priority -> drop forwards
            for k in range(i + 1, min(len(seq), i + 1 + guard)):
                if seq[k][1] != b:
                    break
                dropped.add(seq[k][0])
    return dropped


def make_splits(sessions, ratios, seed=42, block_frames=8, guard=2, min_sessions_for_session_mode=3):
    rng = random.Random(seed)
    sids = sorted(sessions)
    assign = {}            # (sid, frame_id) -> split
    dropped = defaultdict(list)

    if len(sids) >= min_sessions_for_session_mode:
        mode = "session"
        sizes = [len(sessions[s]) for s in sids]
        for sid, sp in zip(sids, greedy_assign(sizes, ratios, rng)):
            for f in sessions[sid]:
                assign[(sid, f["frame_id"])] = sp
    else:
        mode = "block"
        blocks = []  # (sid, [frame_ids])
        for sid in sids:
            ids = [f["frame_id"] for f in sessions[sid]]
            chunks = [ids[i:i + block_frames] for i in range(0, len(ids), block_frames)]
            if len(chunks) > 1 and len(chunks[-1]) < block_frames // 2:   # fold a tiny tail into the previous block
                chunks[-2].extend(chunks.pop())
            blocks += [(sid, c) for c in chunks]
        if len(blocks) < 3:
            raise ValueError(f"only {len(blocks)} block(s) of {block_frames} frames: need more reviewed frames "
                             "(or a smaller --block-frames) to fill train, val and test")
        for (sid, ids), sp in zip(blocks, greedy_assign([len(b[1]) for b in blocks], ratios, rng)):
            for fid in ids:
                assign[(sid, fid)] = sp
        for sid in sids:
            seq = [[f["frame_id"], assign[(sid, f["frame_id"])]] for f in sessions[sid]]
            for fid in apply_guard(seq, guard):
                dropped[sid].append(fid)
                del assign[(sid, fid)]

    splits = {s: defaultdict(list) for s in ratios}
    for (sid, fid), sp in assign.items():
        splits[sp][sid].append(fid)
    splits = {s: {sid: sorted(v, key=frame_number) for sid, v in d.items()} for s, d in splits.items()}
    empty = [s for s, d in splits.items() if not d]
    if empty:
        raise ValueError(f"split(s) {empty} ended up empty; add data or change --block-frames/--guard")

    box_lookup = {(sid, f["frame_id"]): f.get("boxes", []) for sid in sids for f in sessions[sid]}
    stats = {}
    for sp, d in splits.items():
        counts = Counter(b["label"] for sid, ids in d.items() for fid in ids for b in box_lookup[(sid, fid)])
        stats[sp] = {"frames": sum(len(v) for v in d.values()), "boxes_per_class": dict(counts)}
    return {"mode": mode, "splits": splits, "dropped_guard": dict(dropped), "stats": stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--block-frames", type=int, default=8)
    ap.add_argument("--guard", type=int, default=2)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--min-box-warn", type=int, default=20,
                    help="warn if a class has fewer boxes than this in val or test (heuristic)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    sc = cfg["split"]
    ratios = {k: sc[k] for k in ("train", "val", "test")}
    seed = args.seed if args.seed is not None else sc["seed"]
    root = data_root(cfg)
    out = root / "splits" / "splits.json"
    if out.exists() and not args.overwrite:
        print(f"ERROR: {out} exists. The split is frozen; use --overwrite only if you accept a new test set.",
              file=sys.stderr)
        return 2
    try:
        sessions, excluded = load_sessions(root / "annotations")
        if not sessions:
            raise ValueError(f"no reviewed frames found in {root / 'annotations'}")
        res = make_splits(sessions, ratios, seed, args.block_frames, args.guard)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    doc = {"version": 1, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed, "ratios": ratios,
           "block_frames": args.block_frames, "guard": args.guard, "excluded_unreviewed": excluded, **res}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)

    print(f"mode: {res['mode']} | seed {seed} | wrote {out}")
    n_drop = sum(len(v) for v in res["dropped_guard"].values())
    n_exc = sum(len(v) for v in excluded.values())
    print(f"dropped at boundaries (guard): {n_drop} | excluded un-reviewed: {n_exc}")
    classes = cfg["classes"]
    for sp in ("train", "val", "test"):
        st = res["stats"][sp]
        print(f"  {sp:5s}: {st['frames']:4d} frames | boxes {[st['boxes_per_class'].get(c, 0) for c in classes]}"
              f"  (order {classes})")
    for sp in ("val", "test"):
        for c in classes:
            n = res["stats"][sp]["boxes_per_class"].get(c, 0)
            if n < args.min_box_warn:
                print(f"WARNING: class '{c}' has only {n} boxes in {sp}: metrics for it will be unreliable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
