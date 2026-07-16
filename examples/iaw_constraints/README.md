# `iaw_constraints` — multi-ion IAW with constraints

D + C ion-acoustic-wave fit using `[constraints]` to enforce thermal
equilibrium (`Ti1 = Ti0`) and charge-balance with a floor
(`ifract1 = max(ifract1_floor, 1 - ifract0)`). The floor is an
`[[extra_params]]` dummy that the optimizer is free to move.
Optimizer: SGLD warmup → LBFGS.

Also demonstrates `irf_mode = "gaussian"`: instead of convolving with the
IRF array stored in the data file (see `iaw_full` for that path), the deck
gives only the IRF's standard deviation in pixels (`irf_sigma_px = 1.63`,
matching the FWHM ≈ 25 pm Gaussian baked into the synthetic data) and the
parser builds a clean unit-area Gaussian kernel — useful when a measured
IRF is too noisy to use directly.

```bash
python examples/data/make_data_iaw.py   # one-time
thomson-fit fit.toml                    # writes fit_result.h5
python plot.py                          # writes params_vs_time.png, spectra.png
```

Truth profiles: D fraction 0.7, C fraction 0.3, `Te = 500 eV`,
`Ti_D` ramps 200→500 eV. After the fit, `ifract1` should track
`1 - ifract0` (subject to the floor) exactly because the constraint is
applied to every sample.
