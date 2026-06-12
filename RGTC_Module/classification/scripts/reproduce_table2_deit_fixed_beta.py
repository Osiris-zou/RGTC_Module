"""Assemble manuscript Table 2 for DeiT-S/16 and DeiT-B/16."""
from pathlib import Path
import argparse
import csv

ROWS = [
    ["DeiT-S/16","Full patched",0,"",79.71,94.97,0.00,4.61,"0.00%",3146.45,"1.00x",197,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-S/16","ToMe",4,"",79.59,94.89,0.12,4.02,"12.80%",3145.09,"≈1.00x",149,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-S/16","Reliability-Guided",4,"0.000",79.59,94.89,0.12,4.02,"12.80%",3028.47,"0.96x",149,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-S/16","ToMe",8,"",79.32,94.74,0.39,3.43,"25.60%",3495.93,"1.11x",101,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-S/16","Reliability-Guided",8,"0.035",79.34,94.78,0.37,3.43,"25.60%",3097.76,"0.98x",101,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-S/16","ToMe",12,"",78.89,94.48,0.82,2.85,"38.18%",3386.76,"1.08x",53,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-S/16","Reliability-Guided",12,"0.050",78.91,94.53,0.80,2.85,"38.18%",2983.53,"0.95x",53,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-S/16","ToMe",16,"",77.79,93.88,1.92,2.30,"50.11%",3326.15,"1.06x",11,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-S/16","Reliability-Guided",16,"0.015",77.75,93.78,1.96,2.30,"50.11%",2939.86,"0.93x",11,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-S/16","ToMe",20,"",74.10,91.37,5.61,1.88,"59.22%",3276.43,"1.04x",4,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-S/16","Reliability-Guided",20,"0.040",73.97,91.36,5.74,1.88,"59.22%",2894.08,"0.92x",4,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-S/16","ToMe",25,"",61.26,81.57,18.45,1.53,"66.81%",3227.47,"1.03x",2,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-S/16","Reliability-Guided",25,"0.020",61.05,81.63,18.66,1.53,"66.81%",2887.90,"0.92x",2,"accuracy from main table; throughput from robust DeiT-S diagnostic"],
    ["DeiT-B/16","Full patched",0,"",81.74,95.59,0.00,17.58,"0.00%",1167.02,"1.00x",197,"Table 2 manuscript value"],
    ["DeiT-B/16","ToMe",4,"",81.55,95.44,0.19,15.34,"12.73%",1208.06,"1.04x",149,"Table 2 manuscript value"],
    ["DeiT-B/16","Reliability-Guided",4,"0.000",81.55,95.44,0.19,15.34,"12.73%",1188.17,"1.02x",149,"Table 2 manuscript value"],
    ["DeiT-B/16","ToMe",8,"",81.10,95.14,0.64,13.12,"25.37%",1442.48,"1.24x",101,"Table 2 manuscript value"],
    ["DeiT-B/16","Reliability-Guided",8,"0.035",81.17,95.18,0.57,13.12,"25.37%",1413.55,"1.21x",101,"Table 2 manuscript value"],
    ["DeiT-B/16","ToMe",12,"",80.14,94.59,1.60,10.93,"37.88%",1732.12,"1.48x",53,"Table 2 manuscript value"],
    ["DeiT-B/16","Reliability-Guided",12,"0.050",80.14,94.55,1.60,10.93,"37.88%",1692.60,"1.45x",53,"Table 2 manuscript value"],
    ["DeiT-B/16","ToMe",16,"",77.51,93.07,4.23,8.78,"50.06%",2098.87,"1.80x",11,"Table 2 manuscript value"],
    ["DeiT-B/16","Reliability-Guided",16,"0.015",77.49,93.01,4.25,8.78,"50.06%",2046.72,"1.75x",11,"Table 2 manuscript value"],
    ["DeiT-B/16","ToMe",20,"",66.22,84.72,15.52,7.14,"59.39%",2505.07,"2.15x",4,"Table 2 manuscript value"],
    ["DeiT-B/16","Reliability-Guided",20,"0.040",66.34,84.71,15.40,7.14,"59.39%",2432.34,"2.08x",4,"Table 2 manuscript value"],
    ["DeiT-B/16","ToMe",25,"",27.48,44.14,54.26,5.80,"67.01%",2946.83,"2.53x",2,"Table 2 manuscript value"],
    ["DeiT-B/16","Reliability-Guided",25,"0.020",27.23,44.03,54.51,5.80,"67.01%",2844.46,"2.44x",2,"Table 2 manuscript value"],
]
FIELDS = ["model","method","r","beta","top1_percent","top5_percent","accuracy_drop","gflops","flops_reduction","throughput_img_s","speedup","final_tokens","source_note"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/final/table2_deit_fixed_beta.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(ROWS)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
