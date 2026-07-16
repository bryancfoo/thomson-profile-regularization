# Examples

Eight self-contained examples, each in its own subdirectory. They share the
synthetic data files in [`data/`](data/).

## One-time setup

```bash
python examples/data/make_data_epw.py        # → data_epw.h5
python examples/data/make_data_iaw.py        # → data_iaw.h5  +  throughput.csv
python examples/data/make_data_kappa.py      # → data_kappa.h5       (iaw_kappa)
python examples/data/make_data_epw_2temp.py  # → data_epw_2temp.h5   (epw_custom_dist)
```

## Layout

| Directory | Demonstrates | Data |
|---|---|---|
| [`epw_basic/`](epw_basic/) | minimal EPW fit (`n`, `Te`), pure LBFGS | `data_epw.h5` |
| [`forward_only/`](forward_only/) | forward model from `[profiles]`, no fitting | `data_epw.h5` |
| [`iaw_constraints/`](iaw_constraints/) | multi-ion IAW + `[constraints]` + `[[extra_params]]` (charge balance) + `irf_mode = "gaussian"` | `data_iaw.h5` |
| [`iaw_sample/`](iaw_sample/) | same as `iaw_constraints` plus posterior error bars (`[sampling]`) | `data_iaw.h5` |
| [`iaw_l_curve/`](iaw_l_curve/) | same as `iaw_constraints` plus `[penalty.*]` + `[l_curve]` Tikhonov sweep | `data_iaw.h5` |
| [`iaw_full/`](iaw_full/) | "kitchen sink": IRF + throughput + notch + background + `[probe_beam]` + every `[penalty.*]` | `data_iaw.h5` |
| [`iaw_kappa/`](iaw_kappa/) | `[species]` with the built-in `kappa` ion model (general quadrature path) | `data_kappa.h5` |
| [`epw_custom_dist/`](epw_custom_dist/) | user-supplied distribution callable (`my_dists.py:two_temp`, bi-Maxwellian) | `data_epw_2temp.h5` |

[`forward_model_examples.ipynb`](forward_model_examples.ipynb) walks the
forward-model Python API interactively.

## Running an example

```bash
cd examples/iaw_sample
thomson-fit fit.toml             # writes fit_result.h5 (incl. /summary and /samples when sampling ran)
python plot.py                   # writes params_vs_time.png and spectra.png
```

The `plot.py` script auto-detects whether posterior summary statistics are
present in the fit result and renders 16/84-percentile error bands on each
parameter when they are. It also overlays synthetic-truth profiles when the
data file contains them.
