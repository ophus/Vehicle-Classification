#!/usr/bin/env python3
"""Train HOG+color -> Linear SVM on the patch dataset; select C on VALIDATION; never touches test.

Usage:
    python scripts/train_svm.py --name svm_4cls_v1
    python scripts/train_svm.py --name svm_2cls_v1 --drop truck,unknown
    python scripts/train_svm.py --name svm_3cls_v1 --merge truck=other,unknown=other
    options: --c-grid 0.01,0.1,1,10  --seed 42  --overwrite

Input : <data_root>/processed/patches/patches_{train,val}.npz   (build_patches.py)
Output: artifacts/models/<name>.joblib            scaler + SVM + class order + feature/patch config + versions
        artifacts/models/<name>_card.json         human-readable model card
        artifacts/metrics/<name>_val.json         validation metrics (accuracy, per-class P/R/F1, confusion matrix)
        artifacts/figures/<name>_val_confusion.png
--drop removes those classes' patches entirely; --merge maps several classes onto one name.
Library-provided: sklearn LinearSVC / StandardScaler, cv2.HOGDescriptor. TEST data is not read here:
the final test evaluation is a separate, one-time step.
Exit codes: 0 ok, 2 error.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.config import artifacts_dir, data_root, load_config  # noqa: E402
from features.hog_color import extract_features, feature_dim  # noqa: E402


def load_patches(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def remap(y, class_names, merge=None, drop=()):
    """Return (keep_mask, new_y, new_names). merge: {old_name: new_name}; drop: iterable of old names."""
    merge, drop = merge or {}, set(drop)
    unknown = [n for n in list(merge) + list(drop) if n not in class_names]
    if unknown:
        raise ValueError(f"unknown class name(s) {unknown}; available: {list(class_names)}")
    if "background" in drop:
        raise ValueError("cannot drop 'background'")
    new_names, mapping = [], {}
    for i, n in enumerate(class_names):
        if n in drop:
            mapping[i] = None
            continue
        nn = merge.get(n, n)
        if nn not in new_names:
            new_names.append(nn)
        mapping[i] = new_names.index(nn)
    keep = np.array([mapping[int(v)] is not None for v in y], dtype=bool)
    new_y = np.array([mapping[int(v)] for v in y[keep]], dtype=np.int64)
    return keep, new_y, new_names


def fit_select(Xtr, ytr, Xva, yva, c_grid, seed):
    """Fit scaler on train only; try each C; choose by validation macro-F1 (classes present in val)."""
    scaler = StandardScaler().fit(Xtr)
    A, B = scaler.transform(Xtr), scaler.transform(Xva)
    labels = sorted(set(yva.tolist()))
    results, best = [], None
    for C in c_grid:
        t0 = time.time()
        svm = LinearSVC(C=C, class_weight="balanced", random_state=seed, max_iter=20000, dual="auto").fit(A, ytr)
        pred = svm.predict(B)
        r = {"C": C, "val_macro_f1": float(f1_score(yva, pred, labels=labels, average="macro", zero_division=0)),
             "val_accuracy": float(accuracy_score(yva, pred)), "fit_seconds": round(time.time() - t0, 2)}
        results.append(r)
        if best is None or r["val_macro_f1"] > best[0]["val_macro_f1"]:
            best = (r, svm)
    return scaler, best[1], best[0], results


def metrics(yva, pred, names):
    idx = list(range(len(names)))
    rep = classification_report(yva, pred, labels=idx, target_names=names, output_dict=True, zero_division=0)
    counts = np.bincount(yva, minlength=len(names))
    return {"accuracy": float(accuracy_score(yva, pred)),
            "majority_class_accuracy": float(counts.max() / counts.sum()),
            "per_class": {n: {k: float(rep[n][k]) if k != "support" else int(rep[n][k])
                              for k in ("precision", "recall", "f1-score", "support")} for n in names},
            "confusion_matrix": confusion_matrix(yva, pred, labels=idx).tolist()}


def save_confusion_png(cm, names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2, 1.0 * len(names) + 2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_pairs(text):
    out = {}
    for item in filter(None, (text or "").split(",")):
        a, sep, b = item.partition("=")
        if not sep or not a or not b:
            raise argparse.ArgumentTypeError(f"bad --merge item {item!r}; use old=new,old2=new")
        out[a.strip()] = b.strip()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--drop", default="")
    ap.add_argument("--merge", type=parse_pairs, default={})
    ap.add_argument("--c-grid", default="0.01,0.1,1,10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    art = artifacts_dir(cfg)
    model_path = art / "models" / f"{args.name}.joblib"
    if model_path.exists() and not args.overwrite:
        print(f"ERROR: {model_path} exists; choose another --name or pass --overwrite", file=sys.stderr)
        return 2
    pdir = data_root(cfg) / "processed" / "patches"
    try:
        tr, va = load_patches(pdir / "patches_train.npz"), load_patches(pdir / "patches_val.npz")
        c_grid = [float(v) for v in args.c_grid.split(",")]
        drop = [d for d in args.drop.split(",") if d]
        names0 = [str(n) for n in tr["class_names"]]
        if [str(n) for n in va["class_names"]] != names0:
            raise ValueError("class_names differ between train and val patches; rebuild patches")
        ktr, ytr, names = remap(tr["y"], names0, args.merge, drop)
        kva, yva, _ = remap(va["y"], names0, args.merge, drop)
        for i, n in enumerate(names):
            if (ytr == i).sum() == 0:
                raise ValueError(f"class '{n}' has no TRAIN patches; drop or merge it")
        if len(yva) == 0:
            raise ValueError("no validation patches left")
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    ptc = cfg["patch"]
    if tr["X"].shape[1] != ptc["size"]:
        print(f"ERROR: patches are {tr['X'].shape[1]}px but config patch.size={ptc['size']}", file=sys.stderr)
        return 2
    t0 = time.time()
    Ftr = extract_features(tr["X"][ktr], cfg["features"])
    Fva = extract_features(va["X"][kva], cfg["features"])
    t_feat = time.time() - t0
    assert Ftr.shape[1] == feature_dim(ptc["size"], cfg["features"])
    scaler, svm, best, grid = fit_select(Ftr, ytr, Fva, yva, c_grid, args.seed)
    pred = svm.predict(scaler.transform(Fva))
    m = metrics(yva, pred, names)

    for d in ("models", "metrics", "figures"):
        (art / d).mkdir(parents=True, exist_ok=True)
    bundle = {"scaler": scaler, "svm": svm, "class_names": names, "feature_cfg": cfg["features"],
              "patch_cfg": ptc, "merge": args.merge, "drop": drop, "C": best["C"]}
    joblib.dump(bundle, model_path)
    reload = joblib.load(model_path)                      # round-trip check: same predictions after saving
    same = bool(np.array_equal(reload["svm"].predict(reload["scaler"].transform(Fva)), pred))
    card = {"name": args.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "device": "CPU",
            "class_order": names, "input": f"{ptc['size']}x{ptc['size']} uint8 BGR patch, margin {ptc['margin']}",
            "feature_dim": int(Ftr.shape[1]), "chosen_C": best["C"],
            "train_patches_per_class": {n: int((ytr == i).sum()) for i, n in enumerate(names)},
            "val_patches_per_class": {n: int((yva == i).sum()) for i, n in enumerate(names)},
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "numpy": np.__version__, "opencv": cv2.__version__, "joblib": joblib.__version__},
            "feature_seconds": round(t_feat, 2), "roundtrip_predictions_identical": same}
    (art / "models" / f"{args.name}_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    (art / "metrics" / f"{args.name}_val.json").write_text(json.dumps(
        {"split": "val", "C_grid": grid, "chosen": best, **m}, indent=1), encoding="utf-8")
    save_confusion_png(m["confusion_matrix"], names, art / "figures" / f"{args.name}_val_confusion.png",
                       f"{args.name} - validation")

    print(f"classes: {names} | feature dim {Ftr.shape[1]} | train {len(ytr)} / val {len(yva)} patches")
    for r in grid:
        print(f"  C={r['C']:<6} val macro-F1={r['val_macro_f1']:.3f}  acc={r['val_accuracy']:.3f}  ({r['fit_seconds']}s)")
    print(f"chosen C={best['C']} | val accuracy {m['accuracy']:.3f} (always-predict-majority = {m['majority_class_accuracy']:.3f})")
    for n in names:
        p = m["per_class"][n]
        print(f"  {n:11s} P={p['precision']:.2f} R={p['recall']:.2f} F1={p['f1-score']:.2f} (n_val={p['support']})")
    print(f"saved model: {model_path} | save/reload predictions identical: {same}")
    return 0 if same else 2


if __name__ == "__main__":
    sys.exit(main())
