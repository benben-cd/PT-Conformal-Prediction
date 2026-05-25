# Questioning the Coverage-Length Metric in Conformal Prediction

Experimental code for:

> **Questioning the Coverage-Length Metric in Conformal Prediction: When Shorter Intervals Are Not Better**
>
> Yizhou Min, Yizhou Lu, Lanqi Li, Zhen Zhang, and Jiaye Teng
>
> International Conference on Machine Learning (ICML), 2026

The current manuscript PDF is included as
[`icml2026_PCP_camera_ready.pdf`](./icml2026_PCP_camera_ready.pdf).

## Overview

This repository contains the experimental code used to study a limitation of
the commonly reported coverage-length metric in conformal prediction. The
paper introduces **Prejudicial Trick (PT)**, a constructed transformation that
can make prediction sets shorter while preserving marginal coverage, but at
the cost of substantial instability across repeated calibrations.

The experiments cover:

- synthetic regression examples illustrating the misleading improvement in
  interval length and a no-misspecification failure case;
- real-data regression comparisons of VCP and PT-VCP;
- interval stability comparisons among VCP, PT-VCP, and Localized-CP;
- ImageNet classification comparisons of RAPS and PT-RAPS;
- CQR and PT-CQR experiments;
- p-value criteria, group coverage, larger-bias, relaxed-PT, and ablation
  studies.

In several scripts, the historical variable name `PCP` refers to the
PT-transformed counterpart of the base conformal method.

## Paper-to-Code Map

| Manuscript item | Experiment | Code and supplied artifacts |
| --- | --- | --- |
| Table 1 | Synthetic motivating example | `ordinary_regression_task/simulation_subgaussian.py`; `ordinary_regression_task/simulation_ablation/` |
| Figure 5 | Synthetic case without misspecification | `ordinary_regression_task/simulation_guassian.py`; `ordinary_regression_task/simulation_ablation/` |
| Figure 1 | BIKE marginal and group coverage | `ordinary_regression_task/ordinary_regression_cvg.py`, `ordinary_regression_task/group_coverage.py`; `ordinary_regression_task/group_coverage_result/`, `plt/cvg/` |
| Table 2 | VCP versus PT-VCP on ten regression datasets | Core fitting and evaluation components are in `ordinary_regression_task/`; a consolidated Table 2 exporter/result is not included in this snapshot |
| Table 3 | Interval Stability for VCP, PT-VCP, and Localized-CP | `ordinary_regression_task/interval_stability.py`, `sim_local_cp/ordinary_regression_sigma_interval_stability.py`; `ordinary_regression_task/interval_stability_results/table3_interval_stability_vcp_ptvcp.csv`, `sim_local_cp/sigma_interval_stability_summary.csv` |
| Table 4 | RAPS versus PT-RAPS on ImageNet | `classification_task/cvg_len_cls.py`; `classification_task/cvg_len_result/` |
| Table 5 | CQR versus PT-CQR | Model and evaluation components in `cqr/`; a consolidated Table 5 exporter/result is not included in this snapshot |
| Table 6 | Relaxed PT-VCP with perturbation rate `epsilon = 0.01` | `ordinary_regression_task/PT_perturbation.py`; `ordinary_regression_task/pt_result/` |
| Table 7 | Regression experiment with larger bias | `ordinary_regression_task/larger_bias.py`; `ordinary_regression_task/larger_bias/table7_larger_bias_alpha0.1_p0.96.csv` |
| Table 8 | Group coverage results | `ordinary_regression_task/group_coverage.py`; checked-in results currently cover BIKE only |
| Tables 9-10 | Classification p-value criteria | `classification_task/p_value_criteria.py`, `classification_task/conformal.py`; `classification_task/p_value_result/` |
| Table 11 | Interval Stability for CQR and PT-CQR | `cqr/Stability_cqr.py`; `cqr/interval_stability_results.csv` |
| Table 12 | Interval Stability for RAPS and PT-RAPS | `classification_task/interval_stability_cls.py`; `classification_task/std_results/std_results_cls.csv` |
| Figures 6-15 | Ablation studies over `p` and misspecification level | `ordinary_regression_task/ordinary_regression_ablation.py`, `cqr/cqr_ablation.py`; `plt/ablation_study/` |

## Repository Layout

```text
.
|-- classification_task/        # RAPS/PT-RAPS ImageNet experiments
|   |-- cvg_len_cls.py           # coverage and set-size evaluation (Table 4)
|   |-- interval_stability_cls.py# classification stability (Table 12)
|   |-- p_value_criteria.py      # p-value criteria (Tables 9-10)
|   |-- cvg_len_result/
|   |-- p_value_result/
|   `-- std_results/
|-- cqr/                        # CQR/PT-CQR experiments
|   |-- cqr_FitModel.py
|   |-- cqr_ablation.py
|   |-- Stability_cqr.py         # Table 11
|   |-- cqr_ablation_result/
|   `-- interval_stability_results.csv
|-- ordinary_regression_task/   # VCP/PT-VCP regression experiments
|   |-- ordinary_regression_FitModel.py
|   |-- ordinary_regression_ablation.py
|   |-- ordinary_regression_cvg.py
|   |-- interval_stability.py   # VCP/PT-VCP part of Table 3
|   |-- group_coverage.py       # Figure 1 and Table 8
|   |-- PT_perturbation.py      # Table 6
|   |-- larger_bias.py          # Table 7
|   |-- simulation_subgaussian.py
|   |-- simulation_guassian.py
|   `-- *_result/               # supplied result tables
|-- sim_local_cp/               # Localized-CP stability comparison
|   `-- ordinary_regression_sigma_interval_stability.py
|-- plt/                        # notebooks/scripts and exported figures
|-- requirements.txt
`-- icml2026_PCP_camera_ready.pdf
```

## Environment

The experiments were written for Python and PyTorch. A convenient setup is:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The main dependencies are `numpy`, `pandas`, `scipy`, `scikit-learn`,
`torch`, `torchvision`, `Pillow`, `tqdm`, `matplotlib`, `plotly`, and
`jupyter`.

## Data Preparation

Large datasets and trained checkpoints are intentionally not committed to the
repository.

### Regression Data

The regression experiments use MEPS-19, MEPS-20, MEPS-21, BIKE,
BLOG-DATA, BIO, FACEBOOK-1, FACEBOOK-2, CONCRETE, and STAR.

Raw input files should be placed under the corresponding module directory
before running the preprocessing scripts:

```text
ordinary_regression_task/datasets/
ordinary_regression_task/get_meps_data/
cqr/datasets/
cqr/get_meps_data/
sim_local_cp/datasets/
sim_local_cp/get_meps_data/
```

For each module that you plan to run, prepare the MEPS datasets with:

```bash
cd ordinary_regression_task/get_meps_data
python get_meps_data.py
python get_meps_data_19.py
python get_meps_data_20.py
python get_meps_data_21.py
```

Repeat from `cqr/get_meps_data/` and `sim_local_cp/get_meps_data/` when
running those independent experiment modules.

Expected external regression files include:

```text
datasets/bike_train.csv
datasets/blogfeedback/blogData_train.csv
datasets/CASP.csv
datasets/facebook/Features_Variant_1.csv
datasets/facebook/Features_Variant_2.csv
datasets/concrete_data.csv
datasets/star.csv
```

### ImageNet Classification Data

For the classification experiments, arrange ImageNet validation images as:

```text
classification_task/imagenet_val/
|-- n01440764/
|-- n01443537/
`-- ...
```

The classification scripts use torchvision pretrained model weights and
require access to those weights if they are not already cached locally.

## Reproducing Experiments

Commands below assume execution from the named experiment directory.
Randomness is controlled by seed loops in the scripts.

### Synthetic Experiments: Table 1 and Figure 5

```bash
cd ordinary_regression_task
python simulation_subgaussian.py
python simulation_guassian.py
```

The supplied summaries have been organized in
`ordinary_regression_task/simulation_ablation/`. The filename
`simulation_guassian.py` retains the spelling used by the original
experiment code.

### Marginal and Group Coverage: Figure 1 and Table 8

```bash
cd ordinary_regression_task
mkdir -p cvg_result
python ordinary_regression_cvg.py
python group_coverage.py
```

The current checked-in configuration of `group_coverage.py` runs BIKE, which
supports Figure 1 and the BIKE entries of Table 8. The manuscript also reports
STAR and MEPS group-coverage results; those require enabling the additional
datasets in the script and regenerating their result files.

### Regression Length Experiments: Tables 2, 6, and 7

Train point prediction models:

```bash
cd ordinary_regression_task
python ordinary_regression_FitModel.py
```

Run the relaxed-PT and larger-bias supplementary experiments:

```bash
python PT_perturbation.py
python larger_bias.py
```

`PT_perturbation.py` corresponds to Table 6 with `epsilon = 0.01`.
`larger_bias.py` corresponds to Table 7 and writes the supplied
`larger_bias/table7_larger_bias_alpha0.1_p0.96.csv` summary.

The folder contains the core VCP/PT-VCP training and evaluation code used for
the real-data study, but this snapshot does not contain a single command or
precomputed summary table that directly emits the final Table 2 layout.

### Interval Stability: Tables 3, 11, and 12

For ordinary regression VCP/PT-VCP stability:

```bash
cd ordinary_regression_task
python interval_stability.py
```

This command writes the consolidated VCP/PT-VCP summary to
`interval_stability_results/table3_interval_stability_vcp_ptvcp.csv`.

For the Localized-CP comparison in Table 3:

```bash
cd sim_local_cp
python ordinary_regression_FitModel.py
python ordinary_regression_sigma_interval_stability.py
```

For CQR/PT-CQR stability:

```bash
cd cqr
python cqr_FitModel.py
python Stability_cqr.py
```

For RAPS/PT-RAPS stability:

```bash
cd classification_task
python interval_stability_cls.py
```

The supplied classification summary contains all nine architectures reported
in Table 12. In the current script, only `ResNet18` is enabled by default;
enable the remaining model names to regenerate the complete table.

### Classification: Tables 4, 9, and 10

```bash
cd classification_task
python cvg_len_cls.py
python p_value_criteria.py
```

`cvg_len_cls.py` evaluates one selected architecture at a time through its
`model_name` setting. The supplied Table 4 result files are organized under
`cvg_len_result/`.

`p_value_criteria.py` generates p-value criterion summaries. New runs are
written to `p_value_result_full/` by the current script, while the supplied
paper-facing results are organized under `p_value_result/`.

### CQR and Ablation Studies: Table 5 and Figures 6-15

```bash
cd cqr
python cqr_FitModel.py
python cqr_ablation.py

cd ../ordinary_regression_task
python ordinary_regression_ablation.py
```

The checked-in CQR code provides model fitting, ablation, and stability
experiments. It does not currently include one consolidated exporter for the
Table 5 presentation in the manuscript.

Plotting notebooks and exported PDFs for the ablation figures are provided
under `plt/ablation_study/`.

## Acknowledgements

The CQR experiments include code adapted from the public CQR implementation:

- Romano, Patterson, and Candes. *Conformalized Quantile Regression*.
- https://github.com/yromano/cqr

Please also follow the original licenses and terms of all datasets and
pretrained torchvision models used in the experiments.

## Citation

Please cite the ICML 2026 paper when using this code. A BibTeX entry can be
added here once the final proceedings metadata is available.
