import csv
from pathlib import Path
from collections import defaultdict

PAIR_CSV = Path(os.environ.get("RGTC_PATH", ""))

records = defaultdict(dict)

with open(PAIR_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        method = row["method"]
        image_name = row["image_name"]

        records[image_name][method] = {
            "lpips": float(row["lpips_vs_full"]),
            "msssim": float(row["ms_ssim_vs_full"]),
            "full_path": row["full_path"],
            "method_path": row["method_path"],
        }

total = 0
reliability_guided_lpips_win = 0
reliability_guided_msssim_win = 0
both_win = 0

diff_rows = []

for image_name, item in records.items():
    if "tome" not in item or "reliability_guided" not in item:
        continue

    total += 1

    tome_lpips = item["tome"]["lpips"]
    reliability_guided_lpips = item["reliability_guided"]["lpips"]

    tome_msssim = item["tome"]["msssim"]
    reliability_guided_msssim = item["reliability_guided"]["msssim"]

    lpips_diff = tome_lpips - reliability_guided_lpips
    msssim_diff = reliability_guided_msssim - tome_msssim

    lpips_win = lpips_diff > 0
    msssim_win = msssim_diff > 0

    if lpips_win:
        reliability_guided_lpips_win += 1
    if msssim_win:
        reliability_guided_msssim_win += 1
    if lpips_win and msssim_win:
        both_win += 1

    diff_rows.append({
        "image_name": image_name,
        "lpips_diff_tome_minus_reliability_guided": lpips_diff,
        "msssim_diff_reliability_guided_minus_tome": msssim_diff,
        "tome_lpips": tome_lpips,
        "reliability_guided_lpips": reliability_guided_lpips,
        "tome_msssim": tome_msssim,
        "reliability_guided_msssim": reliability_guided_msssim,
    })

print("========== Pairwise Win Statistics ==========")
print(f"Total pairs: {total}")
print(f"Reliability-Guided lower LPIPS: {reliability_guided_lpips_win} / {total} = {reliability_guided_lpips_win / total * 100:.2f}%")
print(f"Reliability-Guided higher MS-SSIM: {reliability_guided_msssim_win} / {total} = {reliability_guided_msssim_win / total * 100:.2f}%")
print(f"Reliability-Guided wins both: {both_win} / {total} = {both_win / total * 100:.2f}%")

diff_rows_sorted = sorted(
    diff_rows,
    key=lambda x: x["lpips_diff_tome_minus_reliability_guided"],
    reverse=True
)

out_csv = PAIR_CSV.parent / "pairwise_win_analysis.csv"

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = list(diff_rows_sorted[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(diff_rows_sorted)

print(f"[DONE] Pairwise analysis saved to: {out_csv}")
print("\nTop 10 cases where Reliability-Guided has lower LPIPS than ToMe:")
for row in diff_rows_sorted[:10]:
    print(
        row["image_name"],
        "LPIPS diff:",
        f"{row['lpips_diff_tome_minus_reliability_guided']:.6f}",
        "MS-SSIM diff:",
        f"{row['msssim_diff_reliability_guided_minus_tome']:.6f}",
    )