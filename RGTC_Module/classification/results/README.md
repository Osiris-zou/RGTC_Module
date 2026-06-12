# Classification result data

The `results/final/` directory is the authoritative source for the manuscript tables. Each CSV file corresponds directly to one table in the revised manuscript:

- `table1_vit_fixed_beta.csv`: Table 1, ViT-B/16 and ViT-L/16 fixed-beta classification results.
- `table1_vit_throughput.csv`: Table 1 throughput-only source data for ViT-B/16 and ViT-L/16.
- `table2_deit_fixed_beta.csv`: Table 2, DeiT-S/16 and DeiT-B/16 fixed-beta transfer results.
- `table3_final_token_count.csv`: Table 3, final token counts under fixed per-layer merging rates.
- `table4_paired_bootstrap.csv`: Table 4, paired bootstrap stability analysis.
- `table5_beta_sensitivity.csv`: Table 5, beta-sensitivity summary.
- `table6_stabilization_ablation.csv`: Table 6, stabilization ablation.

Intermediate beta sweeps are stored in `results/sweeps/`. Earlier main-result logs that do not exactly correspond to the manuscript tables are stored in `results/legacy/`. Runtime diagnostics are stored in `results/diagnostics/`; in particular, `deit_small_robust_bs64.csv` is the separate robust throughput diagnostic used for the DeiT-S/16 throughput column.

Small differences may appear when recomputing accuracy-drop or FLOPs-reduction values from full-precision raw logs because the manuscript tables report rounded values. Throughput measurements may vary across runs and hardware states; the final CSV files preserve the submitted manuscript values, while scripts provide the protocol for independent verification.
