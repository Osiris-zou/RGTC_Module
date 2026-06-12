#!/usr/bin/env python3
"""
Reproduce Table 1 ViT-B/16 and ViT-L/16 model-only throughput with a mean-of-5 protocol.

Run from the classification/ directory:

python scripts/reproduce_table1_vit_throughput_mean5.py \
  --device cuda \
  --batch-size 64 \
  --warmup 50 \
  --timed-forwards 200 \
  --repeats 5 \
  --output-raw results/raw/table1_vit_throughput_mean5_raw.csv \
  --output-final results/final/table1_vit_fixed_beta_mean5.csv

This script keeps the manuscript accuracy/GFLOPs values unchanged and only replaces
throughput/speedup with freshly measured mean throughput.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch

# Enable TF32 for high-throughput FP32 inference on RTX 4090.
# AMP is still disabled; this only affects CUDA float32 matmul precision.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

print("[Runtime]")
print("  torch:", torch.__version__)
print("  cuda:", torch.version.cuda)
print("  gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("  tf32 matmul:", torch.backends.cuda.matmul.allow_tf32)
print("  tf32 cudnn:", torch.backends.cudnn.allow_tf32)
print("  matmul precision:", torch.get_float32_matmul_precision())

# Allow running without installing the package when launched from classification/.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_model  # noqa: E402


# Paper Table 1 constants. Accuracy and GFLOPs are intentionally kept as manuscript values.
# Only throughput and speedup are newly measured by this script.
TABLE1_ROWS: List[Dict] = [
    # ViT-B/16
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="Full patched", run_method="tome", r=0,  beta=None,  top1=84.41, top5=97.26, acc_drop=0.00,  gflops=17.58, flops_reduction="0.00%",  final_tokens=197),
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="ToMe",         run_method="tome", r=4,  beta=None,  top1=84.25, top5=97.21, acc_drop=0.16,  gflops=15.34, flops_reduction="12.73%", final_tokens=149),
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="Ours",         run_method="reliability_guided", r=4,  beta=0.000, top1=84.25, top5=97.21, acc_drop=0.16,  gflops=15.34, flops_reduction="12.73%", final_tokens=149),
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="ToMe",         run_method="tome", r=8,  beta=None,  top1=83.70, top5=97.02, acc_drop=0.71,  gflops=13.12, flops_reduction="25.37%", final_tokens=101),
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="Ours",         run_method="reliability_guided", r=8,  beta=0.035, top1=83.78, top5=97.05, acc_drop=0.63,  gflops=13.12, flops_reduction="25.37%", final_tokens=101),
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="ToMe",         run_method="tome", r=12, beta=None,  top1=82.69, top5=96.57, acc_drop=1.72,  gflops=10.92, flops_reduction="37.88%", final_tokens=53),
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="Ours",         run_method="reliability_guided", r=12, beta=0.050, top1=82.87, top5=96.65, acc_drop=1.54,  gflops=10.92, flops_reduction="37.88%", final_tokens=53),
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="ToMe",         run_method="tome", r=16, beta=None,  top1=80.17, top5=95.43, acc_drop=4.24,  gflops=8.78,  flops_reduction="50.06%", final_tokens=11),
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="Ours",         run_method="reliability_guided", r=16, beta=0.015, top1=80.27, top5=95.39, acc_drop=4.14,  gflops=8.78,  flops_reduction="50.06%", final_tokens=11),
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="ToMe",         run_method="tome", r=20, beta=None,  top1=63.57, top5=83.78, acc_drop=20.84, gflops=7.14,  flops_reduction="59.39%", final_tokens=4),
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="Ours",         run_method="reliability_guided", r=20, beta=0.040, top1=63.78, top5=83.95, acc_drop=20.63, gflops=7.14,  flops_reduction="59.39%", final_tokens=4),
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="ToMe",         run_method="tome", r=25, beta=None,  top1=21.15, top5=35.96, acc_drop=63.26, gflops=5.80,  flops_reduction="67.01%", final_tokens=2),
    dict(model="ViT-B/16", timm_model="vit_base_patch16_224", method="Ours",         run_method="reliability_guided", r=25, beta=0.020, top1=21.28, top5=36.01, acc_drop=63.13, gflops=5.80,  flops_reduction="67.01%", final_tokens=2),
    # ViT-L/16
    dict(model="ViT-L/16", timm_model="vit_large_patch16_224", method="Full patched", run_method="tome", r=0,  beta=None,  top1=85.68, top5=97.74, acc_drop=0.00,  gflops=61.60, flops_reduction="0.00%",  final_tokens=197),
    dict(model="ViT-L/16", timm_model="vit_large_patch16_224", method="ToMe",         run_method="tome", r=4,  beta=None,  top1=85.22, top5=97.49, acc_drop=0.46,  gflops=46.15, flops_reduction="25.08%", final_tokens=101),
    dict(model="ViT-L/16", timm_model="vit_large_patch16_224", method="Ours",         run_method="reliability_guided", r=4,  beta=0.020, top1=85.23, top5=97.51, acc_drop=0.45,  gflops=46.15, flops_reduction="25.08%", final_tokens=101),
    dict(model="ViT-L/16", timm_model="vit_large_patch16_224", method="ToMe",         run_method="tome", r=8,  beta=None,  top1=83.21, top5=96.60, acc_drop=2.47,  gflops=30.99, flops_reduction="49.69%", final_tokens=7),
    dict(model="ViT-L/16", timm_model="vit_large_patch16_224", method="Ours",         run_method="reliability_guided", r=8,  beta=0.055, top1=83.29, top5=96.71, acc_drop=2.39,  gflops=30.99, flops_reduction="49.69%", final_tokens=7),
    dict(model="ViT-L/16", timm_model="vit_large_patch16_224", method="ToMe",         run_method="tome", r=12, beta=None,  top1=20.41, top5=31.93, acc_drop=65.27, gflops=20.89, flops_reduction="66.09%", final_tokens=2),
    dict(model="ViT-L/16", timm_model="vit_large_patch16_224", method="Ours",         run_method="reliability_guided", r=12, beta=0.050, top1=20.62, top5=31.94, acc_drop=65.06, gflops=20.89, flops_reduction="66.09%", final_tokens=2),
]


def iter_rows(model_filter: str) -> Iterable[Dict]:
    wanted = {
        "all": {"ViT-B/16", "ViT-L/16"},
        "vit_b": {"ViT-B/16"},
        "vit_l": {"ViT-L/16"},
    }[model_filter]
    for row in TABLE1_ROWS:
        if row["model"] in wanted:
            yield row


def beta_for_run(row: Dict) -> float:
    return 0.0 if row["beta"] is None else float(row["beta"])


def benchmark_one(row: Dict, args: argparse.Namespace, repeat_id: int) -> float:
    device = torch.device(args.device)
    model = load_model(
        model_name=row["timm_model"],
        method=row["run_method"],
        r=int(row["r"]),
        beta=beta_for_run(row),
        checkpoint=None,
        pretrained=False,
        device=device,
        prop_attn=not args.no_prop_attn,
    )
    model.eval()
    x = torch.randn(args.batch_size, 3, 224, 224, device=device)

    with torch.inference_mode():
        # Warm-up for this exact model configuration.
        for _ in range(args.warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(args.timed_forwards):
                _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            elapsed_sec = starter.elapsed_time(ender) / 1000.0
        else:
            import time
            start = time.perf_counter()
            for _ in range(args.timed_forwards):
                _ = model(x)
            elapsed_sec = time.perf_counter() - start

    throughput = args.batch_size * args.timed_forwards / elapsed_sec
    del model, x
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(throughput)


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=["all", "vit_b", "vit_l"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--timed-forwards", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--no-prop-attn", action="store_true", help="Disable proportional attention. Keep default off for off-the-shelf timm models.")
    parser.add_argument("--output-raw", default="results/raw/table1_vit_throughput_mean5_raw.csv")
    parser.add_argument("--output-final", default="results/final/table1_vit_fixed_beta_mean5.csv")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu only for debugging, not for manuscript throughput.")

    raw_rows: List[Dict] = []
    final_rows: List[Dict] = []

    for row in iter_rows(args.models):
        label = f"{row['model']} | {row['method']} | r={row['r']} | beta={row['beta']}"
        vals: List[float] = []
        print(f"\n[Benchmark] {label}", flush=True)
        for rep in range(args.repeats):
            value = benchmark_one(row, args, rep)
            vals.append(value)
            print(f"  repeat {rep + 1}/{args.repeats}: {value:.2f} img/s", flush=True)
            raw_rows.append({
                "model": row["model"],
                "timm_model": row["timm_model"],
                "method": row["method"],
                "run_method": row["run_method"],
                "r": row["r"],
                "beta": "-" if row["beta"] is None else f"{row['beta']:.3f}",
                "repeat_id": rep,
                "batch_size": args.batch_size,
                "warmup": args.warmup,
                "timed_forwards": args.timed_forwards,
                "amp": "disabled",
                "prop_attn": not args.no_prop_attn,
                "throughput": f"{value:.6f}",
            })

        mean_v = statistics.mean(vals)
        std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
        median_v = statistics.median(vals)
        out = dict(row)
        out.update({
            "beta": "-" if row["beta"] is None else f"{row['beta']:.3f}",
            "throughput_mean": mean_v,
            "throughput_std": std_v,
            "throughput_median": median_v,
            "reported_throughput": mean_v,
            "reported_stat": "mean_of_5",
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "timed_forwards": args.timed_forwards,
            "repeats": args.repeats,
        })
        final_rows.append(out)

    # Compute speedup within each model relative to its newly measured Full patched row.
    full_by_model = {
        row["model"]: row["reported_throughput"]
        for row in final_rows
        if row["method"] == "Full patched"
    }
    for row in final_rows:
        base = full_by_model[row["model"]]
        row["speedup"] = row["reported_throughput"] / base if base > 0 else math.nan

    raw_fields = [
        "model", "timm_model", "method", "run_method", "r", "beta", "repeat_id",
        "batch_size", "warmup", "timed_forwards", "amp", "prop_attn", "throughput",
    ]
    final_fields = [
        "model", "method", "r", "beta", "top1", "top5", "acc_drop", "gflops",
        "flops_reduction", "final_tokens", "reported_throughput", "speedup",
        "throughput_mean", "throughput_std", "throughput_median", "reported_stat",
        "batch_size", "warmup", "timed_forwards", "repeats", "timm_model", "run_method",
    ]

    # Format numeric columns for manuscript-friendly CSV.
    formatted_final: List[Dict] = []
    for row in final_rows:
        f = dict(row)
        for k in ["top1", "top5", "acc_drop", "gflops", "reported_throughput", "throughput_mean", "throughput_std", "throughput_median"]:
            f[k] = f"{float(f[k]):.2f}" if k not in {"throughput_std"} else f"{float(f[k]):.4f}"
        f["speedup"] = f"{float(f['speedup']):.2f}x"
        formatted_final.append(f)

    write_csv(Path(args.output_raw), raw_rows, raw_fields)
    write_csv(Path(args.output_final), formatted_final, final_fields)
    print(f"\nSaved raw repeated results to:   {args.output_raw}")
    print(f"Saved final Table 1 source to: {args.output_final}")


if __name__ == "__main__":
    main()
