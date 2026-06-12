"""Assemble manuscript Table 3 final token-count summary."""
from pathlib import Path
import argparse
import csv

ROWS = [
    ["ViT-B/16", 12, 149, 101, 53, 11, 4, 2, "r=8/12/16 practical; r=20/25 extreme"],
    ["DeiT-S/16", 12, 149, 101, 53, 11, 4, 2, "same token schedule as ViT-B/16"],
    ["DeiT-B/16", 12, 149, 101, 53, 11, 4, 2, "same token schedule as ViT-B/16"],
    ["ViT-L/16", 24, 101, 7, 2, "", "", "", "r=8/12 already highly aggressive"],
]
FIELDS = ["model", "depth", "r4", "r8", "r12", "r16", "r20", "r25", "interpretation"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/final/table3_final_token_count.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(ROWS)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
