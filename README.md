# Questioning the Coverage-Length Metric in Conformal Prediction

Official experimental code for:

> **Questioning the Coverage-Length Metric in Conformal Prediction: When Shorter Intervals Are Not Better**  
> Yizhou Min, Yizhou Lu, Lanqi Li, Zhen Zhang, Jiaye Teng  
> Proceedings of the International Conference on Machine Learning (ICML), 2026

The camera-ready manuscript snapshot used to organize this release is available
at [`icml2026_PCP_camera_ready.pdf`](icml2026_PCP_camera_ready.pdf).

## Overview

This repository studies probabilistically transformed conformal prediction
(PT-VCP / PT-RAPS in the experiment files) and illustrates why interval or set
size alone can reward uninformative randomization. It contains the experiments
reported for:

- ImageNet classification with RAPS and PT-RAPS.
- Ordinary regression on ten benchmark datasets.
- Conformalized Quantile Regression (CQR).
- A localized-CP-like regression comparison based on repeatedly fitted
  conditional scale models.
- Synthetic, group-coverage, stability, and ablation experiments.

In several older script identifiers, `PCP` denotes the probabilistic
transformation used for PT-VCP or PT-RAPS in the paper.

## Repository Layout

```text
.
|-- icml2026_PCP_camera_ready.pdf
|-- classification_task/
|   |-- cvg_len_cls.py                  # ImageNet coverage/set-size comparison
|   |-- interval_stability_cls.py       # Randomized set-size stability
|   |-- p_value_criteria.py             # Paper p-value criteria, all classifiers
|   |-- conformal.py                    # PT-RAPS criteria implementation
|   |-- conformal_procedure.py          # RAPS/PT-RAPS prediction sets
|   |-- cvg_len_result/                 # Stored coverage/set-size results
|   `-- p_value_result/                 # Stored p-value criteria results
|-- ordinary_regression_task/
|   |-- ordinary_regression_FitModel.py # Train mean regression networks
|   |-- ordinary_regression_cvg.py      # Coverage curves
|   |-- ordinary_regression_ablation.py # Alpha/p/bias ablations
|   |-- PT_perturbation.py              # PT-VCP results used in the paper
|   |-- larger_bias.py                  # Larger-bias comparison
|   |-- group_coverage.py               # Group coverage experiment
|   |-- interval_stability.py           # Randomization stability
|   |-- simulation_*.py                 # Synthetic examples
|   `-- datasets/                       # Loaders and local dataset location
|-- cqr/
|   |-- cqr_FitModel.py                 # Train CQR models
|   |-- cqr_ablation.py                 # CQR ablations
|   |-- Stability_cqr.py                # CQR stability
|   |-- cqrfile/                        # Quantile network implementation
|   `-- nonconformist/                  # CQR conformal utilities
|-- sim_local_cp/
|   |-- ordinary_regression_FitModel.py
|   `-- ordinary_regression_sigma_interval_stability.py
`-- plt/                                # Plotting notebooks/scripts and figures
```

## Environment

The classification experiments require a CUDA-enabled PyTorch setup because
the provided ImageNet evaluation code moves models and batches to CUDA.
Regression and CQR scripts select CUDA when available and otherwise use CPU.

One possible setup is:

```bash
conda create -n pt-vcp python=3.10 -y
conda activate pt-vcp
pip install -r requirements.txt
```

For CUDA, install the PyTorch and torchvision builds matching your CUDA
runtime if they differ from the default pip resolution.

## Data

Data files and trained checkpoints are intentionally excluded by `.gitignore`.
The local preparation folder may contain them, but they should not be committed
to the public repository.

### Regression and CQR

Each of `ordinary_regression_task/`, `cqr/`, and `sim_local_cp/` has a
`datasets/` directory with the loader and a short dataset note. Place the
required files under the `datasets/` directory of the experiment family that
you run:

| Script identifier | Dataset |
| --- | --- |
| `meps_19`, `meps_20`, `meps_21` | Processed MEPS panels; follow `get_meps_data/README.md` |
| `bike` | Bike Sharing (`bike_train.csv`) |
| `blog_data` | BlogFeedback (`blogfeedback/blogData_train.csv`) |
| `bio` | CASP (`CASP.csv`) |
| `facebook_1`, `facebook_2` | Facebook Comment Volume feature variants |
| `concrete` | Concrete Compressive Strength (`Concrete_Data.csv`) |
| `star` | STAR (`STAR.csv`) |

The regression scripts expect the processed MEPS filenames
`meps_19_reg_fix.csv`, `meps_20_reg_fix.csv`, and `meps_21_reg_fix.csv`;
rename or place the generated processed files accordingly.

### ImageNet

Obtain the ImageNet validation set under its applicable license and arrange it
in the `ImageFolder` layout:

```text
classification_task/imagenet_val/
|-- n01440764/
|   `-- *.JPEG
`-- ...
```

The experiments use all 50,000 validation images, with 10,000 sampled for
calibration and the remaining 40,000 for evaluation.

## Reproducing Experiments

All commands below are executed from the corresponding experiment directory,
because the scripts use relative dataset, model, and output paths.

### Ordinary Regression

The ordinary regression evaluation uses ten datasets, five random seeds,
`alpha = 0.1`, and the paper's experiment-specific `p` and bias settings.

```bash
cd ordinary_regression_task

# Train mean models; writes checkpoints to model/
python ordinary_regression_FitModel.py

# Main and supplementary evaluations
python PT_perturbation.py
python larger_bias.py
python ordinary_regression_ablation.py
python interval_stability.py
python group_coverage.py

# Synthetic illustrations
python simulation_guassian.py
python simulation_subgaussian.py
```

Key stored artifacts include:

| Output | Description |
| --- | --- |
| `ordinary_regression_task/pt_result/` | PT-VCP coverage and length outputs |
| `ordinary_regression_task/larger_bias/table2_larger_bias_alpha0.1_p0.96.csv` | Larger-bias comparison |
| `ordinary_regression_task/interval_stability_results/` | Randomization stability |
| `ordinary_regression_task/group_coverage_result/` | Group coverage outputs |
| `ordinary_regression_task/ablation_study_result/` | Ablation outputs |

### Localized-CP-Like Stability Comparison

This experiment trains separate conditional scale functions and measures
variation over repeated training of `sigma(x)`.

```bash
cd sim_local_cp
python ordinary_regression_FitModel.py
python ordinary_regression_sigma_interval_stability.py
```

New runs write the summary to
`sim_local_cp/interval_stability_results/sigma_interval_stability_summary.csv`;
the currently provided summary is also retained at
`sim_local_cp/sigma_interval_stability_summary.csv`.

### Conformalized Quantile Regression

```bash
cd cqr

# Train quantile models; writes checkpoints to cqr_model/
python cqr_FitModel.py

python cqr_ablation.py
python Stability_cqr.py
```

The ablation script writes to `cqr/cqr_ablation_result/`. A new stability run
writes `cqr/std_results.csv`; the provided organized stability summary is
`cqr/interval_stability_results.csv`.

### ImageNet Classification

The supplied classifiers are torchvision pretrained models; downloading
weights may require network access on the first run. The experiment code
requires a CUDA device.

```bash
cd classification_task

# RAPS versus PT-RAPS coverage and set size.
# Select the model by editing model_name in cvg_len_cls.py.
python cvg_len_cls.py

# P-value criteria for all nine classifiers.
python p_value_criteria.py

# Set-size stability experiment.
python interval_stability_cls.py
```

The p-value run uses nine torchvision architectures, five seeds,
`alpha = epsilon = 0.1`, `p = 0.95`, `pt_bias = 40`,
`pt_index_range = 300`, and smoothed p-values. Stored summaries are provided
under `classification_task/p_value_result/`.

## Provided Results

The repository includes small CSV/Markdown result summaries and plotting
material, while omitting input datasets and checkpoint files. In particular:

- `classification_task/p_value_result/all_models_p_value_criteria_summary.csv`
  collects the ImageNet p-value criteria for all evaluated classifiers.
- `ordinary_regression_task/larger_bias/` contains the generated larger-bias
  result table.
- `sim_local_cp/sigma_interval_stability_summary.csv` contains the provided
  localized-CP-like stability summary.
- `plt/` contains plotting notebooks and exported PDF figures.

## Acknowledgements

The ImageNet conformal-classification implementation builds on:

- Angelopoulos, Bates, Jordan, and Malik. *Uncertainty Sets for Image
  Classifiers using Conformal Prediction*. ICLR, 2021.

The CQR code builds on:

- Romano, Patterson, and Candes. *Conformalized Quantile Regression*.
  NeurIPS, 2019.

## Citation

Please cite the ICML 2026 paper when using this code. A finalized BibTeX
entry can be added after the proceedings metadata is published.
