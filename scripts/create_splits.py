#!/usr/bin/env python3
"""Create leakage-resistant train/validation/test splits by recording session.

The tool refuses to split invalid annotations and requires at least three
independent sessions. It writes a JSON manifest; it never moves or copies data.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.config import data_root, load_config  # noqa: E402
from validate_dataset import validate_file  # noqa: E402


SPLITS = ("train", "val", "test")


def split_sessions(session_ids, ratios, seed=42):
    """Return deterministic session assignments using largest-remainder counts."""
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("session IDs must be unique")
    if len(session_ids) < len(SPLITS):
        raise ValueError(f"need at least {len(SPLITS)} independent sessions; found {len(session_ids)}")
    if set(ratios) != set(SPLITS):
        raise ValueError(f"split ratios must define exactly {SPLITS}")
    values = [float(ratios[name]) for name in SPLITS]
    if any(value <= 0 for value in values) or abs(sum(values) - 1.0) > 1e-8:
        raise ValueError("split ratios must be positive and sum to 1.0")

    # Reserve one session for each split, then apportion the remainder.
    remaining = len(session_ids) - len(SPLITS)
    quotas = [remaining * value for value in values]
    counts = [1 + int(quota) for quota in quotas]
    left = len(session_ids) - sum(counts)
    order = sorted(range(len(SPLITS)), key=lambda i: (-(quotas[i] % 1), i))
    for i in order[:left]:
        counts[i] += 1

    shuffled = list(session_ids)
    random.Random(seed).shuffle(shuffled)
    result, cursor = {}, 0
    for name, count in zip(SPLITS, counts):
        result[name] = sorted(shuffled[cursor:cursor + count])
        cursor += count
    return result


def create_manifest(annotations_dir, root, cfg, ratios, seed=42, deep=False):
    """Validate annotation files and build a split manifest without writing it."""
    files = sorted(Path(annotations_dir).glob("*.json"))
    if not files:
        raise ValueError(f"no annotation files in {annotations_dir}")

    ids, problems = [], []
    for path in files:
        result = validate_file(path, cfg, root, deep=deep)
        identity_warnings = [message for message in result["warnings"]
                             if message.startswith("session_id ")]
        if result["errors"] or identity_warnings:
            details = result["errors"] or identity_warnings
            problems.append(f"{path.name}: {len(details)} validation issue(s); first: {details[0]}")
        else:
            ids.append(path.stem)
    if problems:
        raise ValueError("cannot create splits until annotations validate:\n- " + "\n- ".join(problems))

    assignments = split_sessions(ids, ratios, seed)
    return {
        "schema_version": 1,
        "group_by": "session",
        "seed": seed,
        "ratios": {name: float(ratios[name]) for name in SPLITS},
        "sessions": assignments,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="config JSON (default configs/default.json)")
    ap.add_argument("--output", help="manifest path (default <data_root>/splits/session_split.json)")
    ap.add_argument("--seed", type=int, help="override config split.seed")
    ap.add_argument("--deep", action="store_true", help="also decode images and check duplicates")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing manifest")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = data_root(cfg)
    out = Path(args.output) if args.output else root / "splits" / "session_split.json"
    if out.exists() and not args.overwrite:
        print(f"ERROR: {out} already exists; pass --overwrite to replace it", file=sys.stderr)
        return 2
    split_cfg = cfg.get("split", {})
    ratios = {name: split_cfg.get(name) for name in SPLITS}
    seed = args.seed if args.seed is not None else split_cfg.get("seed", 42)
    try:
        manifest = create_manifest(root / "annotations", root, cfg, ratios, seed, deep=args.deep)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {out}")
    for name in SPLITS:
        print(f"{name}: {len(manifest['sessions'][name])} session(s): {', '.join(manifest['sessions'][name])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
