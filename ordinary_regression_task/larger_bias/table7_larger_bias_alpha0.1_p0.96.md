# Table 7 (larger bias)

**Caption.** Comparison of performance between VCP and PT-VCP in regression tasks across different datasets at fixed `alpha = 0.1` and `p = 0.96`, using a larger bias setting to amplify the interval-length contrast. Values are reported as `mean +/- std` over 5 random seeds. Bold marks the smaller (better) value in the two length columns for each dataset.

| Dataset | Bias | VCP Coverage | VCP Length | PT-VCP Coverage | PT-VCP Length |
|---|---:|---:|---:|---:|---:|
| MEPS-19 | 80 | 0.900 +/- 0.006 | 162.32 +/- 0.51 | 0.898 +/- 0.006 | **156.99 +/- 0.65** |
| MEPS-20 | 80 | 0.900 +/- 0.005 | 161.96 +/- 0.26 | 0.901 +/- 0.003 | **156.62 +/- 0.65** |
| MEPS-21 | 80 | 0.904 +/- 0.005 | 162.27 +/- 0.25 | 0.899 +/- 0.003 | **156.88 +/- 0.75** |
| BIKE | 40 | 0.901 +/- 0.005 | 80.46 +/- 0.04 | 0.899 +/- 0.006 | **77.29 +/- 0.18** |
| BLOG-DATA | 80 | 0.899 +/- 0.005 | 161.64 +/- 0.73 | 0.899 +/- 0.004 | **156.10 +/- 0.92** |
| BIO | 40 | 0.901 +/- 0.004 | 81.13 +/- 0.05 | 0.900 +/- 0.003 | **78.11 +/- 0.14** |
| FACEBOOK-1 | 40 | 0.902 +/- 0.002 | 80.77 +/- 0.07 | 0.901 +/- 0.003 | **78.15 +/- 0.28** |
| FACEBOOK-2 | 40 | 0.900 +/- 0.001 | 80.94 +/- 0.15 | 0.899 +/- 0.001 | **78.33 +/- 0.32** |
| CONCRETE | 20 | 0.895 +/- 0.030 | 40.32 +/- 0.02 | 0.892 +/- 0.021 | **38.65 +/- 0.21** |
| STAR | 20 | 0.909 +/- 0.010 | 40.14 +/- 0.01 | 0.906 +/- 0.009 | **38.47 +/- 0.27** |
