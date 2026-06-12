# Classification module: ViT and DeiT on ImageNet-1K

This module reproduces the classification experiments, token-count analysis, beta sensitivity analysis, paired bootstrap confidence intervals, FLOPs estimates, and model-only throughput benchmarks.

## Installation

```bash
conda env create -f environment.yml
conda activate rgtc-classification
pip install -e .
```

## Data and checkpoints

See `data/README.md` and `checkpoints/README.md`. ImageNet-1K and third-party checkpoints are not redistributed.

## Core usage

```python
import timm
from rgtc_classification import patch_timm

model = timm.create_model("vit_base_patch16_224", pretrained=True)
patch_timm(model, method="reliability_guided", beta=0.05)
model.r = 12
```

## Main commands

```bash
python scripts/evaluate_imagenet.py --data-path /path/to/imagenet/val --model vit_base_patch16_224 --method reliability_guided --r 12 --beta 0.05 --preprocess inception
python scripts/benchmark_throughput.py --model vit_base_patch16_224 --method reliability_guided --r 12 --beta 0.05
python scripts/estimate_flops.py --model vit_base_patch16_224 --method reliability_guided --r 12 --beta 0.05
python scripts/save_correctness.py --data-path /path/to/imagenet/val --model vit_base_patch16_224 --method reliability_guided --r 12 --beta 0.05 --output results/per_sample/vit_b_ours_r12_correct.npy
python scripts/paired_bootstrap.py --baseline results/per_sample/vit_b_tome_r12_correct.npy --proposed results/per_sample/vit_b_ours_r12_correct.npy --samples 10000 --seed 0
```

The final result files used by the paper are in `results/final/`. Runtime diagnostics are separated in `results/diagnostics/` to avoid mixing protocols.

## Manuscript table assembly

The paper-authoritative CSV files are stored in `results/final/`. They can be regenerated in assemble mode with:

```bash
python scripts/reproduce_table1_vit_fixed_beta.py
python scripts/reproduce_table1_vit_throughput.py --mode assemble
python scripts/reproduce_table2_deit_fixed_beta.py
python scripts/reproduce_table3_final_token_count.py
python scripts/reproduce_table5_beta_sensitivity.py
python scripts/reproduce_table6_stabilization_ablation.py --mode assemble
```

`results/sweeps/`, `results/diagnostics/`, and `results/legacy/` are retained for transparency, but should not be treated as manuscript table sources.
