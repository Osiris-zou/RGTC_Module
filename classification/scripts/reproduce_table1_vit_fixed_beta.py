"""Assemble manuscript Table 1 for ViT-B/16 and ViT-L/16.

This script writes the paper-authoritative Table 1 CSV. Accuracy and GFLOPs are
from the submitted manuscript. Throughput values are the manuscript-reported
mean model-only throughput values under the stated runtime protocol.
"""
from pathlib import Path
import argparse
import csv

ROWS = [
    ["ViT-B/16","Full patched",0,"",84.41,97.26,0.00,17.58,"0.00%",1153.37,"1.00x",197],
    ["ViT-B/16","ToMe",4,"",84.25,97.21,0.16,15.34,"12.73%",1192.38,"1.03x",149],
    ["ViT-B/16","Reliability-Guided",4,"0.000",84.25,97.21,0.16,15.34,"12.73%",1178.09,"1.02x",149],
    ["ViT-B/16","ToMe",8,"",83.70,97.02,0.71,13.12,"25.37%",1430.93,"1.24x",101],
    ["ViT-B/16","Reliability-Guided",8,"0.035",83.78,97.05,0.63,13.12,"25.37%",1406.41,"1.22x",101],
    ["ViT-B/16","ToMe",12,"",82.69,96.57,1.72,10.92,"37.88%",1713.21,"1.49x",53],
    ["ViT-B/16","Reliability-Guided",12,"0.050",82.87,96.65,1.54,10.92,"37.88%",1680.40,"1.46x",53],
    ["ViT-B/16","ToMe",16,"",80.17,95.43,4.24,8.78,"50.06%",2077.79,"1.80x",11],
    ["ViT-B/16","Reliability-Guided",16,"0.015",80.27,95.39,4.14,8.78,"50.06%",2030.99,"1.76x",11],
    ["ViT-B/16","ToMe",20,"",63.57,83.78,20.84,7.14,"59.39%",2487.78,"2.16x",4],
    ["ViT-B/16","Reliability-Guided",20,"0.040",63.78,83.95,20.63,7.14,"59.39%",2424.59,"2.10x",4],
    ["ViT-B/16","ToMe",25,"",21.15,35.96,63.26,5.80,"67.01%",2932.20,"2.54x",2],
    ["ViT-B/16","Reliability-Guided",25,"0.020",21.28,36.01,63.13,5.80,"67.01%",2846.69,"2.47x",2],
    ["ViT-L/16","Full patched",0,"",85.68,97.74,0.00,61.60,"0.00%",247.74,"1.00x",197],
    ["ViT-L/16","ToMe",4,"",85.22,97.49,0.46,46.15,"25.08%",318.51,"1.29x",101],
    ["ViT-L/16","Reliability-Guided",4,"0.020",85.23,97.51,0.45,46.15,"25.08%",315.14,"1.27x",101],
    ["ViT-L/16","ToMe",8,"",83.21,96.60,2.47,30.99,"49.69%",474.66,"1.92x",7],
    ["ViT-L/16","Reliability-Guided",8,"0.055",83.29,96.71,2.39,30.99,"49.69%",467.98,"1.89x",7],
    ["ViT-L/16","ToMe",12,"",20.41,31.93,65.27,20.89,"66.09%",682.49,"2.75x",2],
    ["ViT-L/16","Reliability-Guided",12,"0.050",20.62,31.94,65.06,20.89,"66.09%",670.88,"2.71x",2],
]
FIELDS = ["model","method","r","beta","top1_percent","top5_percent","accuracy_drop","gflops","flops_reduction","throughput_img_s","speedup","final_tokens"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/final/table1_vit_fixed_beta.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(ROWS)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
