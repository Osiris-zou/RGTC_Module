# Stable Diffusion v1.5 Module

This module reproduces the Stable Diffusion experiments reported in the manuscript:

**Reliability-Guided Token Compression for Efficient Transformer-Based Visual Inference**  
Submitted to **The Visual Computer**.

The module compares three settings:

- **Full**: Stable Diffusion v1.5 without token merging;
- **ToMe-SD**: ToMeSD-style similarity-based token merging;
- **Reliability-Guided-SD**: the same merge-unmerge pipeline with reliability-guided edge ranking.

The proposed calibration keeps the ToMeSD-style random 2D partition, merge-unmerge locations, merging ratio, and aggregation behavior unchanged, and only replaces the source-edge ranking score with:

```text
calibrated_score = top1_similarity + beta * (top1_similarity - top2_similarity)
```

In the reported Stable Diffusion experiment, the merging ratio is fixed to `0.5`, and the reliability-guided coefficient is fixed to `beta = 0.005` after a small 20-prompt selection sweep.

## Directory structure

```text
stable_diffusion/
├── checkpoints/              # Checkpoint/tokenizer preparation notes
├── configs/                  # Stable Diffusion experiment configuration
├── data/prompts/             # ImageNet-1K class prompts and beta-selection prompts
├── results/final/            # Final manuscript-level result tables
├── results/per_sample/       # Per-sample paired metric files
├── scripts/                  # Generation, metric, and paired-analysis scripts
├── src/rgtc_sd/              # Stable Diffusion and token-merging implementation
├── environment.yml
├── pyproject.toml
├── THIRD_PARTY.md
└── README.md
```

## Key implementation files

```text
src/rgtc_sd/merging.py       # Reliability-guided matching for SD token merging
src/rgtc_sd/pipeline.py      # Stable Diffusion generation pipeline
src/rgtc_sd/diffusion.py     # U-Net blocks and merge-unmerge insertion points
src/rgtc_sd/tomesd/merge.py  # ToMeSD-style utility implementation
```

## Installation

```bash
conda env create -f environment.yml
conda activate rgtc-sd
pip install -e .
```

Required third-party packages include PyTorch, NumPy, Pillow, tqdm, lpips, pytorch-msssim, and pytorch-fid. The exact dependency list is provided in `environment.yml`.

## Checkpoints and tokenizer files

The following files are required for full image generation but are not redistributed in this repository:

```text
v1-5-pruned-emaonly.ckpt
vocab.json
merges.txt
```

Place them according to `checkpoints/README.md`, or provide their paths when running the generation script.

## Prompt files

The prompt files used in the manuscript are included:

```text
data/prompts/imagenet1k_prompts.txt
data/prompts/beta_selection_first20.txt
```

The main generation prompt template is:

```text
a photo of a [class name]
```

For the reported experiment, two images are generated for each of the 1,000 ImageNet-1K class prompts using seeds `0` and `1`, giving 2,000 images per method.

## Image generation

Example command:

```bash
python scripts/generate_imagenet.py \
  --checkpoint /path/to/v1-5-pruned-emaonly.ckpt \
  --tokenizer-dir /path/to/tokenizer \
  --prompt-file data/prompts/imagenet1k_prompts.txt \
  --output-dir generated \
  --method reliability_guided \
  --ratio 0.5 \
  --beta 0.005 \
  --seeds 0 1
```

Windows PowerShell example:

```powershell
python scripts\generate_imagenet.py `
  --checkpoint C:\path\to\v1-5-pruned-emaonly.ckpt `
  --tokenizer-dir C:\path\to\tokenizer `
  --prompt-file data\prompts\imagenet1k_prompts.txt `
  --output-dir generated `
  --method reliability_guided `
  --ratio 0.5 `
  --beta 0.005 `
  --seeds 0 1
```

The final manuscript results use:

```text
Resolution: 512 x 512
Sampler: DDPM
Inference steps: 50
Classifier-free guidance scale: 7.5
Seeds: 0 and 1
Merging ratio: 0.5
Reliability-Guided beta: 0.005
```

## Metric evaluation

The final aggregate metrics used in the manuscript are stored in:

```text
results/final/metrics_summary.csv
```

The paired per-sample metric file is stored in:

```text
results/per_sample/pair_metrics_detail.csv
```

### Recomputing LPIPS, MS-SSIM, FID, time, and memory summary

The script `scripts/evaluate_metrics.py` follows the original evaluation protocol. It currently reads its input paths from the parameter block at the top of the script. Before running the script, set the following variables in `scripts/evaluate_metrics.py`:

```python
RESULT_ROOT = "path/to/generated_or_result_root"
GEN_LOG_CSV = "path/to/generation_log.csv"
IMAGENET_REF_DIR = "path/to/fid_reference_5000"
COMPUTE_FID = True
DEVICE = "cuda"
```

Then run:

```bash
python scripts/evaluate_metrics.py
```

The script writes:

```text
metrics_summary.csv
pair_metrics_detail.csv
```

For reproducibility, the manuscript-level outputs are already included under `results/final/` and `results/per_sample/`.

## Paired stability analysis

The paired stability analysis for the manuscript table is reproduced with:

```bash
python scripts/paired_stability.py \
  --pair-csv results/per_sample/pair_metrics_detail.csv \
  --out-csv results/final/paired_stability.csv \
  --out-md results/final/paired_stability.md \
  --out-detail-csv results/per_sample/pairwise_improvement_detail.csv
```

Windows PowerShell:

```powershell
python scripts\paired_stability.py `
  --pair-csv results\per_sample\pair_metrics_detail.csv `
  --out-csv results\final\paired_stability.csv `
  --out-md results\final\paired_stability.md `
  --out-detail-csv results\per_sample\pairwise_improvement_detail.csv
```

This fixes the command-line argument names to match the actual script interface. The script reports paired LPIPS, MS-SSIM, and quality-distance statistics between ToMe-SD and Reliability-Guided-SD using the Full output as the reference.

## Beta selection

The reported `beta = 0.005` is selected using the first 20 ImageNet prompts and seed `0`:

```bash
python scripts/select_beta.py
```

The fixed subset is provided in:

```text
data/prompts/beta_selection_first20.txt
```

## Final result files

```text
results/final/metrics_summary.csv       # Table 7 aggregate metrics
results/final/paired_stability.csv      # Table 8 paired stability metrics
results/final/generation_log.csv        # Generation time and memory log
results/per_sample/pair_metrics_detail.csv
results/per_sample/pairwise_improvement_detail.csv
results/per_sample/pairwise_win_analysis.csv
```

## Note on manuscript rounding

The manuscript table reports displayed paired differences after rounding the metric means to the shown decimal precision, whereas `paired_stability.csv` retains full-precision raw differences computed from per-sample values. Therefore, the last decimal digit of very small differences may differ slightly. This rounding convention does not affect the interpretation because all reported confidence intervals overlap zero.

## Third-party code and licenses

The Stable Diffusion core is adapted from `hkproj/pytorch-stable-diffusion`, and the token-merging structure follows a ToMeSD-style merge-unmerge pipeline. Third-party attribution and license notes are retained in:

```text
THIRD_PARTY.md
licenses/HKPROJ_STABLE_DIFFUSION_LICENSE.txt
```

Users must obtain the Stable Diffusion checkpoint and tokenizer files from their authorized sources and comply with the corresponding licenses.
