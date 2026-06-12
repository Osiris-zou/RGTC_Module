"""Assemble manuscript Table 5 beta-sensitivity summary."""
from pathlib import Path
import argparse
import csv

ROWS = [
    ["ViT-B/16",8,83.70,0.035,83.78,0.035,83.78,0.08],
    ["ViT-B/16",12,82.69,0.050,82.87,0.050,82.87,0.18],
    ["ViT-B/16",16,80.17,0.015,80.27,0.015,80.27,0.10],
    ["ViT-L/16",4,85.22,0.020,85.23,0.020,85.23,0.01],
    ["ViT-L/16",8,83.21,0.055,83.29,0.055,83.29,0.08],
    ["DeiT-S/16",8,79.32,0.035,79.34,0.300,79.42,0.10],
    ["DeiT-S/16",12,78.89,0.050,78.91,0.035,78.98,0.09],
    ["DeiT-S/16",16,77.79,0.015,77.75,0.035,77.90,0.11],
    ["DeiT-B/16",8,81.10,0.035,81.17,0.100,81.19,0.09],
    ["DeiT-B/16",12,80.14,0.050,80.14,0.010,80.16,0.02],
    ["DeiT-B/16",16,77.51,0.015,77.49,0.200,77.61,0.10],
]
FIELDS = ["model","r","tome_top1_percent","fixed_beta","fixed_beta_ours_top1_percent","best_beta","best_ours_top1_percent","best_delta_top1_pp"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/final/table5_beta_sensitivity.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(ROWS)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
