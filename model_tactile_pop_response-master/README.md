# model_tactile_pop_response

Source code for the paper *"Real-time Simulation of Populations of Tactile Receptors and Afferents in the Skin"* (Ouyang et al. 2019), extended with an LNP spike-generator alternative to the original LIF model.

Requires Python ≥ 3.6 with `numpy`, `matplotlib`, `scipy`, `scikit-image`, `Pillow`. Run every script from the `model_tactile_pop_response-master/` directory so relative `Data/` and `saved_figs/` paths resolve correctly.

---

## Core library

| File | What it does |
|---|---|
| `Receptors.py`        | `tactile_receptors` class - skin mechanics + resistance network + LIF spike generator (the original Ouyang 2019 model). |
| `Lnp.py`              | `LnpReceptors` wrapper - drop-in Linear-Nonlinear-Poisson spike generator that reuses the spatial front-end of `tactile_receptors`. |
| `simset.py`           | Skin ROI loaders (fingertip / palm / back-finger), receptor grid layouts, helpers for setting up simulations. |
| `img_to_eqstimuli.py` | Image -> EPS / EEPS pipeline (greyscale -> edge detection -> resize to probe pitch -> zero-pad). |
| `utils.py`            | Small helpers: Wilcoxon tests, RMSE / R², curve fits, text-file readers. (Inherited from the original codebase.) |


## My LIF-vs-LNP comparisons

| Script | Output | Story |
|---|---|---|
| `compare_ramp_response.py`         | `saved_figs/compare_ramp_psth.png`           | 30-trial ramp-and-hold: LIF vs LNP per afferent + per-phase Hz |
| `compare_rf_size.py`               | `saved_figs/compare_rf_size.png`             | Receptive-field diameter, LIF vs LNP per type |
| `compare_spatial_resolution.py`    | `saved_figs/compare_spatial_resolution.png`  | Two-point discrimination |
| `compare_two_skins.py`             | `saved_figs/compare_two_skins.png`           | Fingertip vs palm skin response |
| `compare_vibration_rates.py`       | `saved_figs/compare_vibration_rates.png`     | Firing rate vs vibration frequency |
| `compare_lif_vs_lnp_isabel.py`     | `saved_figs/lif_vs_lnp_isabel_corr.png`      | "ISABEL" reconstruction correlation, fingertip |
| `compare_lif_vs_lnp_isabel_palm.py`| `saved_figs/lif_vs_lnp_isabel_palm_corr.png` | "ISABEL" reconstruction correlation, palm |
| `compare_lif_vs_lnp_letters.py`    | `saved_figs/lnp_vs_lif_letters.png`          | 3-column comparison: LIF / LNP / Saal 2017 letters |
| `quantify_rf_area.py`              | `saved_figs/quantify_rf_area.png`            | RF area (mm²) vs Saal 2017 benchmark |
| `speed_check_lif_vs_lnp.py`        | *prints to stdout*                           | Wall-clock LIF/LNP runtime comparison |
| `key_findings_summary.py`          | `saved_figs/key_findings_summary.png`        | A4 one-page summary figure (panels A-G + caption) |

## "ISABEL" letter-scan experiments

| Script | Output |
|---|---|
| `name_on_fingertip_LIF.py` | `saved_figs/name_isabel_population_LIF.png` |
| `name_on_fingertip_LNP.py` | `saved_figs/name_isabel_population.png` |
| `name_on_palm_LIF.py`      | `saved_figs/name_isabel_population_palm_LIF.png` |
| `name_on_palm_LNP.py`      | `saved_figs/name_isabel_population_palm.png` |

## Stimulus generators / helpers

| Script | Output |
|---|---|
| `make_letters_isa_image.py`     | `saved_figs/letters_ISA.jpg` (source image for downstream scans) |
| `image_to_stimuli_pipeline.py`  | `saved_figs/image_to_stimuli_pipeline.png` (image -> EEPS pipeline diagram) |

## Parameter / model-explanation plots

Standalone scripts under `plot_model_params/`:

| Script | Output |
|---|---|
| `plot_linear_kernels.py`  | `saved_figs/kernels.png`         |
| `plot_nonlinearities.py`  | `saved_figs/nonlinearities.png`  |
| `plot_hand_densities.py`  | `saved_figs/hand_densities.png`  |
| `plot_poisson_demo.py`    | `saved_figs/poisson_demo.png`    |

---

All observed-data files live under `Data/txtdata/`. Cached simulation outputs and receptor position buffers live under `Data/*.npy`.
