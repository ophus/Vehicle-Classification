#!/usr/bin/env python3
"""Report the runtime environment and verify whether TensorFlow really uses a GPU.

Usage:
    python scripts/check_env.py                    # print report
    python scripts/check_env.py --save colab_t4    # also write artifacts/configs/env_colab_t4.json
    python scripts/check_env.py --require-gpu      # exit code 1 unless a GPU matmul succeeded

Verdict meanings:
    GPU_OK    TensorFlow sees a GPU AND a matmul was placed on it.
    CPU_ONLY  No GPU visible to TensorFlow (or TensorFlow missing); CPU path only.
    TF_MISSING TensorFlow could not be imported.
Do not claim "trained on GPU" unless the verdict is GPU_OK on the machine that trained.
"""
import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.config import artifacts_dir, load_config  # noqa: E402

LIBS = ["numpy", "cv2", "pandas", "sklearn", "matplotlib", "joblib", "tensorflow", "keras"]


def lib_info(name):
    try:
        mod = importlib.import_module(name)
        return {"version": getattr(mod, "__version__", "unknown")}
    except Exception as exc:  # missing or broken install
        return {"version": None, "error": f"{type(exc).__name__}: {exc}"}


def nvidia_smi():
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}
    if out.returncode != 0:
        return {"error": out.stderr.strip() or f"exit {out.returncode}"}
    return {"gpus": [line.strip() for line in out.stdout.splitlines() if line.strip()]}


def tf_check():
    try:
        import tensorflow as tf
    except Exception as exc:
        return {"verdict": "TF_MISSING", "error": f"{type(exc).__name__}: {exc}"}
    info = {"tf_version": tf.__version__, "built_with_cuda": tf.test.is_built_with_cuda()}
    with tf.device("/CPU:0"):
        a = tf.random.uniform((512, 512))
        info["cpu_matmul_device"] = tf.matmul(a, a).device
    gpus = tf.config.list_physical_devices("GPU")
    info["visible_gpus"] = [g.name for g in gpus]
    info["verdict"] = "CPU_ONLY"
    if gpus:
        try:
            with tf.device("/GPU:0"):
                a = tf.random.uniform((2048, 2048))
                b = tf.matmul(a, a)
                b.numpy()  # force execution
            info["gpu_matmul_device"] = b.device
            if "GPU" in b.device.upper():
                info["verdict"] = "GPU_OK"
            try:
                mem = tf.config.experimental.get_memory_info("GPU:0")
                info["gpu_mem_in_use_bytes"] = mem.get("current")
            except Exception:
                pass
        except Exception as exc:
            info["gpu_error"] = f"{type(exc).__name__}: {exc}"
    return info


def collect():
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "is_colab": any(k.startswith("COLAB_") for k in os.environ),
        "libs": {n: lib_info(n) for n in LIBS},
        "nvidia_smi": nvidia_smi(),
        "tensorflow_check": tf_check(),
    }
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--save", metavar="NAME", help="write artifacts/configs/env_NAME.json")
    ap.add_argument("--require-gpu", action="store_true", help="exit 1 unless verdict is GPU_OK")
    args = ap.parse_args()

    env = collect()
    print(json.dumps(env, indent=2, ensure_ascii=False))

    if args.save:
        out_dir = artifacts_dir(load_config()) / "configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"env_{args.save}.json"
        if out.exists():
            print(f"ERROR: {out} already exists; choose another --save name (no silent overwrite).",
                  file=sys.stderr)
            return 2
        out.write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved: {out}")

    verdict = env["tensorflow_check"]["verdict"]
    print(f"VERDICT: {verdict}")
    if args.require_gpu and verdict != "GPU_OK":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
