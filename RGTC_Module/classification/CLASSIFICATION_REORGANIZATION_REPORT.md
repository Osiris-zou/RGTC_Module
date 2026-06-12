# Classification module reorganization report

This classification module has been reorganized to use the revised manuscript as the authoritative source for final tables.

## Main changes

1. `results/final/` now contains only manuscript-table source files:
   - `table1_vit_fixed_beta.csv`
   - `table1_vit_throughput.csv`
   - `table2_deit_fixed_beta.csv`
   - `table3_final_token_count.csv`
   - `table4_paired_bootstrap.csv`
   - `table5_beta_sensitivity.csv`
   - `table6_stabilization_ablation.csv`

2. Intermediate beta sweeps have been moved to `results/sweeps/`:
   - `vit_b_beta_sweep.csv`
   - `vit_l_beta_sweep.csv`
   - `deit_small_beta_sweep.csv`
   - `deit_base_beta_sweep.csv`

3. Earlier main-result logs that do not exactly correspond to the submitted manuscript tables have been moved to `results/legacy/`:
   - `vit_l_main_results.csv`
   - `deit_small_main_results.csv`
   - `deit_base_main_results.csv`
   - `vit_b_flops_results.csv`

4. Table 4 has been completed by adding the DeiT-S/16 and DeiT-B/16 bootstrap rows used in the manuscript.

5. Table 1 ViT throughput values and the assemble script have been synchronized with the current manuscript values.

6. New assemble scripts have been added:
   - `scripts/reproduce_table1_vit_fixed_beta.py`
   - `scripts/reproduce_table2_deit_fixed_beta.py`
   - `scripts/reproduce_table3_final_token_count.py`
   - `scripts/reproduce_table5_beta_sensitivity.py`

7. IDE files, Python cache directories, and compiled Python files have been removed.

## Important manuscript note

The repository now uses `results/diagnostics/deit_small_robust_bs64.csv` as the separate robust runtime diagnostic for DeiT-S/16. The manuscript runtime-protocol paragraph should therefore state that the DeiT-S/16 diagnostic uses batch size 64, not batch size 32.

## Validation

The table-assembly scripts were executed successfully in assemble mode. Python syntax compilation was also checked for all scripts and package source files.
