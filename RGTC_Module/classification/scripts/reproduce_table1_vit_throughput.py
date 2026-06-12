"""Reproduce or assemble the ViT throughput values used in Table 1.

The default ``assemble`` mode writes the manuscript-reported throughput values
for ViT-B/16 and ViT-L/16 to ``results/final/table1_vit_throughput.csv``.

Use ``run`` mode to repeat the model-only CUDA throughput benchmark with random
224x224 inputs. Runtime can vary across machines and driver states, so the raw
benchmark outputs are written separately under ``results/raw/table1_vit_throughput``.
The manuscript table remains the authoritative submitted result.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path



TABLE1_VIT_THROUGHPUT = [
    # model_label, timm_name, method_label, method_arg, r, beta, final_tokens, reported_throughput, reported_speedup
    ('ViT-B/16', 'vit_base_patch16_224', 'Full patched', 'full', 0, 0.0, 197, 1153.37, 1.00),
    ('ViT-B/16', 'vit_base_patch16_224', 'ToMe', 'tome', 4, 0.0, 149, 1192.38, 1.03),
    ('ViT-B/16', 'vit_base_patch16_224', 'Reliability-Guided', 'reliability_guided', 4, 0.0, 149, 1178.09, 1.02),
    ('ViT-B/16', 'vit_base_patch16_224', 'ToMe', 'tome', 8, 0.0, 101, 1430.93, 1.24),
    ('ViT-B/16', 'vit_base_patch16_224', 'Reliability-Guided', 'reliability_guided', 8, 0.035, 101, 1406.41, 1.22),
    ('ViT-B/16', 'vit_base_patch16_224', 'ToMe', 'tome', 12, 0.0, 53, 1713.21, 1.49),
    ('ViT-B/16', 'vit_base_patch16_224', 'Reliability-Guided', 'reliability_guided', 12, 0.05, 53, 1680.40, 1.46),
    ('ViT-B/16', 'vit_base_patch16_224', 'ToMe', 'tome', 16, 0.0, 11, 2077.79, 1.80),
    ('ViT-B/16', 'vit_base_patch16_224', 'Reliability-Guided', 'reliability_guided', 16, 0.015, 11, 2030.99, 1.76),
    ('ViT-B/16', 'vit_base_patch16_224', 'ToMe', 'tome', 20, 0.0, 4, 2487.78, 2.16),
    ('ViT-B/16', 'vit_base_patch16_224', 'Reliability-Guided', 'reliability_guided', 20, 0.04, 4, 2424.59, 2.10),
    ('ViT-B/16', 'vit_base_patch16_224', 'ToMe', 'tome', 25, 0.0, 2, 2932.20, 2.54),
    ('ViT-B/16', 'vit_base_patch16_224', 'Reliability-Guided', 'reliability_guided', 25, 0.02, 2, 2846.69, 2.47),
    ('ViT-L/16', 'vit_large_patch16_224', 'Full patched', 'full', 0, 0.0, 197, 247.74, 1.00),
    ('ViT-L/16', 'vit_large_patch16_224', 'ToMe', 'tome', 4, 0.0, 101, 318.51, 1.29),
    ('ViT-L/16', 'vit_large_patch16_224', 'Reliability-Guided', 'reliability_guided', 4, 0.02, 101, 315.14, 1.27),
    ('ViT-L/16', 'vit_large_patch16_224', 'ToMe', 'tome', 8, 0.0, 7, 474.66, 1.92),
    ('ViT-L/16', 'vit_large_patch16_224', 'Reliability-Guided', 'reliability_guided', 8, 0.055, 7, 467.98, 1.89),
    ('ViT-L/16', 'vit_large_patch16_224', 'ToMe', 'tome', 12, 0.0, 2, 682.49, 2.75),
    ('ViT-L/16', 'vit_large_patch16_224', 'Reliability-Guided', 'reliability_guided', 12, 0.05, 2, 670.88, 2.71),
]

def benchmark(model, batch_size, warmup, runs, repeats, device):
    import torch
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    values = []
    with torch.inference_mode():
        for _ in range(repeats):
            for _ in range(warmup):
                model(x)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(runs):
                model(x)
            end.record()
            torch.cuda.synchronize()
            values.append(batch_size * runs / (start.elapsed_time(end) / 1000.0))
    return values


def write_rows(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "method", "r", "beta", "final_tokens",
        "reported_throughput_img_s", "reported_speedup",
        "batch_size", "warmup", "timed_forward_passes", "repeats",
        "reported_stat", "source",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assemble(output: Path):
    rows = []
    for model_label, _, method_label, _, r, beta, final_tokens, throughput, speedup in TABLE1_VIT_THROUGHPUT:
        rows.append({
            "model": model_label,
            "method": method_label,
            "r": r,
            "beta": "" if method_label == "ToMe" or method_label == "Full patched" else beta,
            "final_tokens": final_tokens,
            "reported_throughput_img_s": throughput,
            "reported_speedup": speedup,
            "batch_size": 64,
            "warmup": 50,
            "timed_forward_passes": 200,
            "repeats": 5,
            "reported_stat": "mean",
            "source": "Table 1 throughput column in submitted manuscript; mean over five repeats",
        })
    write_rows(rows, output)


def run(args):
    import torch
    from common import load_model
    raw_dir = args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    device = torch.device(args.device)
    for model_label, timm_name, method_label, method_arg, r, beta, final_tokens, reported, reported_speedup in TABLE1_VIT_THROUGHPUT:
        model = load_model(
            timm_name,
            method_arg,
            r,
            beta,
            checkpoint=None,
            pretrained=False,
            device=device,
            stabilization="safe",
        )
        vals = benchmark(model, args.batch_size, args.warmup, args.runs, args.repeats, device)
        result = {
            "model": model_label,
            "timm_name": timm_name,
            "method": method_label,
            "method_arg": method_arg,
            "r": r,
            "beta": beta,
            "final_tokens": final_tokens,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "timed_forward_passes": args.runs,
            "repeats": args.repeats,
            "throughput_mean": statistics.mean(vals),
            "throughput_median": statistics.median(vals),
            "throughput_all": vals,
            "manuscript_reported_throughput": reported,
        }
        json_name = f"{model_label.replace('/','_')}_{method_arg}_r{r}_beta{beta}.json".replace(" ", "_")
        (raw_dir / json_name).write_text(json.dumps(result, indent=2))
        summary_rows.append(result)
        print(json.dumps(result, indent=2))

    summary_path = raw_dir / "table1_vit_throughput_rerun_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "timm_name", "method", "method_arg", "r", "beta", "final_tokens",
            "batch_size", "warmup", "timed_forward_passes", "repeats",
            "throughput_mean", "throughput_median", "manuscript_reported_throughput"
        ])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["assemble", "run"], default="assemble")
    parser.add_argument("--output", type=Path, default=Path("results/final/table1_vit_throughput.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw/table1_vit_throughput"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "assemble":
        assemble(args.output)
        print(f"Wrote {args.output}")
    else:
        run(args)


if __name__ == "__main__":
    main()
