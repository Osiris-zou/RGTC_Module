# ADE20K semantic segmentation module

This module reproduces the Segmenter-B/16 transfer experiment in Appendix A.

## Installation

```bash
conda env create -f environment.yml
conda activate rgtc-segmentation
pip install -e third_party/segmenter
```

## Evaluation

```bash
python scripts/evaluate_ade20k.py --checkpoint /path/checkpoint.pth --ade20k-root /path/ADEChallengeData2016 --device cuda --r-list 16 32 48 --beta-map 16:0.015 32:0.035 48:0.050 --out-csv results/final/table_a1_ade20k_results.csv --out-json results/final/table_a1_ade20k_results.json
```

The official Segmenter source is kept under `third_party/segmenter/` with its original MIT license. Dataset and checkpoint files are not redistributed.
