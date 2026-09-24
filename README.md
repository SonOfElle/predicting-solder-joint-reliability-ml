# Predicting Solder Joint Reliability with Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
</p>

> MSc Data Science thesis, rebuilt as a clean, reproducible project.

Predicting the operational lifetime of solder joints in electronic assemblies from material, thermal, and geometric features. Built around a real research constraint: only **450 physical samples** were available, and failure data is expensive to collect. The original thesis used a **Generative Adversarial Network (GAN)** to synthesise 5,000 training samples, and benchmarked four supervised regression models against a published correlation-driven neural network baseline.

The rebuild audits each stage of that pipeline and finds that the thesis's central claim, that GAN augmentation rescued models which could not learn from the raw data, was confounded by a data-layout bug. Both the bug and the corrected result are documented below.

## The problem

Solder joints are the connective tissue of every electronic device: smartphones, medical implants, aerospace avionics. Their failure under thermal cycling, mechanical stress, or creep deformation is a leading cause of field failures, recalls, and warranty claims.

Predicting when a joint will fail matters. But gathering training data is slow and costly. Accelerated life testing takes weeks, and each sample requires destructive testing. Real-world datasets end up small, which is exactly where standard ML models struggle.

This project asks: **can GAN-synthesised data close the gap?** The rebuild asks a second question: **what was the original GAN result actually measuring?**

## Approach

| Stage | Detail |
|---|---|
| **Data** | 450 FEM-derived samples, compiled following the methodology in Samavatian et al. (2020), 24 feature candidates each |
| **Synthesis** | GAN trained to expand to 5,000 samples (TensorFlow / Keras) |
| **Models** | Random Forest, XGBoost, SVR, GPR, using scikit-learn and XGBoost |
| **Baseline** | CDNN from Samavatian et al. [3], preserved unmodified in `baseline_matlab/` |
| **Metrics** | RMSE, R², MSE across training, test, and 3-fold cross-validation |
| **Tuning** | GridSearchCV, capped at 2,000 search samples on the synthetic track for tractability |
| **Audit** | Four-stage narrative from buggy reproduction to fixed pipeline |

## The four stages

Every model runs through a defined sequence of pipeline choices. One axis changes per stage, so each row of `results/metrics.csv` isolates one cause.

| Stage | Layout | Scaling | Grid | What it isolates |
|---|---|---|---|---|
| 1. reproduce | original | original | original | The thesis pipeline as written |
| 2. layout fix | fixed | original | original | The C-order reshape bug |
| 3. scale fix | fixed | principled | original | The scaler-before-split leak |
| 4. regrid | fixed | principled | extended | Miscalibrated SVR and GPR grids |

Layout: `original` is the C-order reshape, `fixed` is per-sample rows. Scaling: `original` fits one MinMaxScaler on the full array before splitting, `principled` fits the feature scaler on `X_train` and the target scaler on `y_train`. Grid: `original` is the thesis grid, `extended` widens SVR and GPR ranges because the original grids were tuned against the scrambled layout.

Stages 1 through 3 apply to all four models. Stage 4 applies to SVR and GPR only. Random Forest and XGBoost grids are layout-agnostic because tree splits do not depend on feature order or column scale.

## Features (24 candidates)

Grouped by physical category, from the original per-sample feature matrix:

- **Thermal**: hot/cold dwelling temperature and time, heating/cooling ramp
- **Material**: solder/upper/lower layer density, CTE, and melting temperature
- **Mechanical**: solder/upper/lower layer Young's modulus and Poisson ratio
- **Geometric**: solder thickness, width, length

Target: **measured useful lifetime** of the joint, in hours.

## Results

### Real track, 450 samples

R² and test RMSE in hours, per model per stage. Higher R² is better, lower RMSE is better.

| Model | Stage 1 reproduce | Stage 2 layout fix | Stage 3 scale fix | Stage 4 regrid |
|---|---|---|---|---|
| Random Forest | +0.09 / 612h | **+0.75 / 321h** | +0.75 / 321h | n/a |
| XGBoost | +0.03 / 633h | **+0.73 / 335h** | +0.73 / 335h | n/a |
| SVR | +0.11 / 605h | +0.37 / 508h | +0.37 / 508h | +0.37 / 509h |
| GPR | +0.08 / 617h | +0.52 / 446h | +0.52 / 446h | +0.52 / 445h |

Read the stage 1 to stage 2 deltas. Random Forest goes from +0.09 to +0.75. XGBoost from +0.03 to +0.73. SVR and GPR improve less but the direction is the same. The thesis reported near-zero R² for all four models and attributed it to the small sample size. The actual cause was the C-order reshape bug in the data loader.

For reference, the thesis author's re-run of the CDNN baseline reached R² +0.82 on the same data. Random Forest at +0.75 and XGBoost at +0.73 sit in the same range, from general-purpose models with no baseline-specific tuning.

### Synthetic track, 5,000 GAN samples

| Model | Stage 1 reproduce | Stage 2 layout fix | Stage 3 scale fix | Stage 4 regrid |
|---|---|---|---|---|
| Random Forest | +0.48 / 784h | +0.48 / 887h | +0.48 / 887h | n/a |
| XGBoost | +0.48 / 787h | +0.47 / 889h | +0.47 / 889h | n/a |
| SVR | +0.53 / 747h | +0.52 / 849h | +0.52 / 849h | +0.53 / 845h |
| GPR | +0.54 / 739h | +0.52 / 846h | +0.52 / 846h | +0.53 / 844h |

Different pattern here. No stage dominates. All four models sit in a narrow band, R² +0.47 to +0.54, and the RMSE in hours actually increases slightly when the layout is fixed. That is the diagnostic: R² on synthetic test data measures how well the model fits the GAN output, not how well it predicts reality. The GAN output has internal structure the models fit regardless of pipeline choices.

One additional caveat. The GAN widened the target distribution. Real target spread is 3,986 hours; synthetic target spread is 9,226 hours. Scaled RMSE is not comparable across tracks without correcting for that. `results/metrics.csv` carries a `target_spread_hours` column so a reader can verify the conversion, and every table in this README reports hours alongside R².

### Cross-evaluation: train on synthetic, test on real

The experiment the original thesis did not run. Same four models, trained at their best synthetic-track stage, evaluated on real data.

| Model | Source R² (synthetic test) | Cross R² (real test) | Cross RMSE |
|---|---|---|---|
| Random Forest | +0.48 | **-0.66** | 767h |
| XGBoost | +0.47 | **-0.63** | 760h |
| SVR | +0.53 | **-0.85** | 808h |
| GPR | +0.53 | **-0.95** | 832h |

Cross R² below zero means the model is worse than predicting the mean. The synthetic R² of ~0.50 was in-distribution fitting, not learned real structure. Cross RMSE of 760 to 832 hours is also worse than the 605 to 633 hours of a model trained on the *buggy-layout real* data. The augmented models underperform even the scramble-bug baseline on real data.

That is the finding. The original negative R² was caused by the layout bug, not the small sample. The original GAN improvement was measured on synthetic test data and does not transfer to reality.

## What this demonstrates

- **End-to-end reproducibility**: every number in this README comes from a notebook that imports from `src/`. No logic in cells.
- **Diagnosis of a subtle bug**: the C-order reshape was invisible for two years because the models did converge, they just converged on noise. A four-stage audit trail isolates it.
- **Working with limited data**: 450 real samples expanded to 5,000 via GAN. The augmentation improved in-distribution fit but did not transfer to real test data, an outcome the original thesis did not measure.
- **Rigorous comparative evaluation**: multiple metrics, cross-validation, a published baseline for context, and a cross-track transfer experiment.
- **Honest reporting**: negative R² and negative cross-track R² are documented, not hidden. The audit trail is committed so a reader can verify every step.
- **Physics-informed ML**: feature engineering and interpretation grounded in solder joint mechanics.

## Repository structure

```
predicting-solder-joint-reliability-ml/
├── README.md
├── LICENSE
├── .gitignore
├── environment.yml
├── data/
│   ├── raw/                    # 450 real samples, committed
│   ├── synthetic/              # generated by make data, gitignored
│   └── README.md               # provenance, licensing, per-file notes
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_gan_synthesis.ipynb
│   ├── 03_model_training_real.ipynb
│   ├── 04_model_training_synthetic.ipynb
│   └── 05_results_comparison.ipynb
├── src/
│   └── solder_reliability/
│       ├── __init__.py         # find_repo_root
│       ├── data_prep.py        # load_raw with dual layout
│       ├── gan.py              # TensorFlow GAN
│       ├── models.py           # RF, XGBoost, SVR, GPR factories
│       ├── train.py            # four-stage orchestrator, cross-eval
│       ├── evaluate.py         # RMSE, R2, CV, denormalisation
│       └── viz.py              # plotting helpers
├── baseline_matlab/
│   ├── README.md
│   └── cdnn_baseline.m
├── reports/
│   └── figures/                # exported figures from all five notebooks
├── results/
│   ├── metrics.csv             # real and synthetic tracks, all stages
│   ├── cross_evaluation.csv    # train-on-synthetic, test-on-real
│   └── partials/               # per-track outputs from notebooks 3 and 4
├── tests/
│   ├── conftest.py
│   └── test_evaluate.py
├── Makefile
└── pyproject.toml
```

### On the notebooks / src split

`src/solder_reliability/` holds all reusable logic: data loading, the GAN, model factories, the four-stage training orchestrator, evaluation, and plotting helpers. The notebooks import from `src` and focus on narrative and visualisation. That means the code that actually runs is unit-testable and importable, not buried in cells, and the notebooks read as a walkthrough of the thesis rather than a pile of one-off scripts.

Every notebook starts by pinning its working directory to repo root via `find_repo_root`, so relative paths like `data/raw` and `reports/figures` resolve regardless of how the notebook is launched.

### On the MATLAB baseline

The CDNN baseline from Samavatian et al. [3] is MATLAB code that the thesis author replicated and re-ran against the augmented dataset. It lives in `baseline_matlab/` rather than being ported to Python, for two reasons:

1. It is not my model to reimplement, and preserving the original implementation keeps the comparison honest.
2. Porting it would introduce a second source of error: an incorrect port could silently change the baseline's numbers.

`baseline_matlab/README.md` documents how to run it. This rebuild does not include a Python port of the CDNN. The baseline is quoted in the Results section for context, not re-run.

### On data provenance

`data/README.md` documents where the 450 samples come from, their licence, how the synthetic 5,000 were generated, and any preprocessing applied. The original samples are derived from Samavatian et al. (Scientific Reports, CC-BY; licence confirmed before committing).

The synthetic data is regenerated by `make data` or by running `02_gan_synthesis.ipynb`. It is not committed. Reproduction from source produces a statistically similar but not bit-identical dataset, because GAN training is stochastic across hardware and TensorFlow versions. The notebook documents this caveat and reports the target-spread widening that results.

## Reproducing these results

```
git clone https://github.com/SonOfElle/predicting-solder-joint-reliability-ml
cd predicting-solder-joint-reliability-ml
make env
make data
make notebooks
make figures
make metrics
make test
```

What each target does:

- `make env`: creates the conda environment and installs pinned dependencies
- `make data`: trains the GANs, writes synthetic arrays to `data/synthetic/`
- `make notebooks`: executes all five notebooks end to end
- `make figures`: verifies figures are present in `reports/figures/`
- `make metrics`: verifies `results/metrics.csv` and `results/cross_evaluation.csv` are present
- `make test`: runs the pytest suite

Or `make all` to chain everything.

The executed notebooks are committed with their outputs, so the numbers in this README can be inspected without running anything. `results/metrics.csv` and `results/cross_evaluation.csv` are the machine-readable artefacts.

### A caveat on exact reproducibility

GAN training is stochastic. Running `make data` on your machine will produce a *plausible* 5,000-sample dataset, not the byte-identical one used for the numbers in this README. A fixed seed is set and documented, and library versions are pinned in `environment.yml`, but some drift is unavoidable across hardware and TensorFlow versions.

The four-stage deltas on the real track are not affected by this. Real data is deterministic, and the layout bug fix is deterministic. Only the synthetic-track numbers and the cross-evaluation numbers depend on the GAN output.

## Limitations and future work

Pulled from the thesis's own closing section, updated for the rebuild's findings:

- **Absolute data volume remains small.** 450 real samples, no augmentation scheme tested here compensates for the small real dataset in a way that transfers to real test data.
- **GAN synthesis does not transfer on this dataset.** An initial GAN trained on 450 samples in 24 dimensions produces mode-covering variation that fits well in-distribution but does not preserve the real feature-target relationship. A conditional GAN, a tabular GAN (e.g. CTGAN), or a much larger epoch budget might change this. Not attempted here.
- **No time-dependent features.** The current feature set is static per sample. Incorporating temperature cycling profiles over time would better capture creep and fatigue behaviour.
- **No ensembling across models.** Combining RF, XGBoost, SVR, and GPR outputs might beat any single model. Not attempted here.
- **Baseline hyperparameter asymmetry.** The CDNN's hyperparameters were tuned on the real dataset in the original research, not re-tuned for the synthetic set. The real-data comparison in particular should be read with that caveat in mind.

## Baseline citation

> Samavatian, V., Fotuhi-Firuzabad, M., Samavatian, M., Dehghanian, P., & Blaabjerg, F. (2020). *Correlation-driven machine learning for accelerated reliability assessment of solder joints in electronics.* Scientific Reports, 10(1), 14821. https://doi.org/10.1038/s41598-020-71926-7

The baseline is cited as a comparative reference. Its MATLAB implementation used for the CDNN replication is included in `baseline_matlab/` for completeness and to keep the comparison reproducible.

## Related

- [Predictive Maintenance Pipeline](https://github.com/SonOfElle/predictive-maintenance-pipeline): where a lifetime-prediction model gets operationalised as a Fabric data pipeline

## Reference

Full thesis: *Reliability prediction of soldering joints in electronic systems*, MSc Data Science, Silesian University of Technology, 2023.

## License

Apache 2.0 for the code. Data licensing is documented separately in `data/README.md`.
