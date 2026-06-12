"""Reproduce or assemble Table 6 stabilization-ablation results.

The default ``assemble`` mode writes the manuscript Table 6 values to
``results/final/table6_stabilization_ablation.csv``.

Use ``run`` mode to re-evaluate the three variants on ImageNet-1K and benchmark
their model-only throughput. This requires the ImageNet-1K validation directory
and may take time.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path



TABLE6_ROWS = [
    {
        "variant": "Without NaN-safe margin",
        "method_arg": "reliability_guided",
        "stabilization": "unsafe",
        "r": 8,
        "beta": 0.015,
        "top1": 68.18,
        "top5": 89.81,
        "throughput_img_s": 1391.73,
        "observation": "unstable",
    },
    {
        "variant": "Safe margin, beta=0",
        "method_arg": "reliability_guided",
        "stabilization": "safe",
        "r": 8,
        "beta": 0.0,
        "top1": 83.70,
        "top5": 97.02,
        "throughput_img_s": 1412.39,
        "observation": "stable",
    },
    {
        "variant": "Safe margin, beta=0.015",
        "method_arg": "reliability_guided",
        "stabilization": "safe",
        "r": 8,
        "beta": 0.015,
        "top1": 83.75,
        "top5": 97.04,
        "throughput_img_s": 1404.08,
        "observation": "improved",
    },
]


def write_final(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "model_name", "variant", "r", "beta", "stabilization",
        "top1", "top5", "throughput_img_s", "observation", "source"
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in TABLE6_ROWS:
            writer.writerow({
                "model": "ViT-B/16",
                "model_name": "vit_base_patch16_224",
                "variant": row["variant"],
                "r": row["r"],
                "beta": row["beta"],
                "stabilization": row["stabilization"],
                "top1": row["top1"],
                "top5": row["top5"],
                "throughput_img_s": row["throughput_img_s"],
                "observation": row["observation"],
                "source": "Table 6 in submitted manuscript",
            })


def evaluate(model, data_path, preprocess, batch_size, workers, device):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from tqdm import tqdm
    from common import transform
    dataset = ImageFolder(data_path, transform=transform(preprocess))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    n = c1 = c5 = 0
    with torch.inference_mode():
        for x, y in tqdm(loader, desc="ImageNet evaluation"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x).topk(5, dim=1).indices
            n += y.numel()
            c1 += (pred[:, 0] == y).sum().item()
            c5 += (pred == y[:, None]).any(dim=1).sum().item()
    return {"images": n, "top1": 100.0 * c1 / n, "top5": 100.0 * c5 / n}


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


def run(args):
    import torch
    from common import load_model
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    summary = []
    for spec in TABLE6_ROWS:
        model = load_model(
            "vit_base_patch16_224",
            spec["method_arg"],
            spec["r"],
            spec["beta"],
            checkpoint=args.checkpoint,
            pretrained=args.pretrained,
            device=device,
            stabilization=spec["stabilization"],
        )
        eval_result = evaluate(model, args.data_path, args.preprocess, args.batch_size, args.workers, device)
        throughputs = benchmark(model, args.batch_size, args.warmup, args.runs, args.repeats, device)
        result = {
            "model": "ViT-B/16",
            "model_name": "vit_base_patch16_224",
            "variant": spec["variant"],
            "r": spec["r"],
            "beta": spec["beta"],
            "stabilization": spec["stabilization"],
            **eval_result,
            "throughput_mean": statistics.mean(throughputs),
            "throughput_median": statistics.median(throughputs),
            "throughput_all": throughputs,
        }
        name = spec["variant"].lower().replace(" ", "_").replace(",", "").replace("=", "").replace(".", "p").replace("-", "_")
        (args.raw_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
        summary.append(result)
        print(json.dumps(result, indent=2))

    summary_path = args.raw_dir / "table6_stabilization_ablation_rerun_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "model_name", "variant", "r", "beta", "stabilization",
            "images", "top1", "top5", "throughput_mean", "throughput_median"
        ])
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row[k] for k in writer.fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["assemble", "run"], default="assemble")
    parser.add_argument("--output", type=Path, default=Path("results/final/table6_stabilization_ablation.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw/table6_stabilization_ablation"))
    parser.add_argument("--data-path", type=str, help="Path to ImageNet-1K validation folder for run mode.")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--preprocess", choices=["inception", "imagenet"], default="inception")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "assemble":
        write_final(args.output)
        print(f"Wrote {args.output}")
    else:
        if not args.data_path:
            raise SystemExit("--data-path is required in run mode")
        run(args)


if __name__ == "__main__":
    main()
