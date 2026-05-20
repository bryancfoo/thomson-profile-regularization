# `epw_basic` — minimal EPW fit

Single-ion electron-plasma-wave fit. Free parameters: `n` and `Te0` per
time step (20 total over `Nt = 10`). Optimizer: LBFGS. The simplest
working example.

```bash
python examples/data/make_data_epw.py   # one-time
thomson-fit fit.toml                    # writes fit_result.h5
python plot.py                          # writes params_vs_time.png, spectra.png
```

Truth `ne(t)` ramps 3→7 ×10¹⁹ cm⁻³, `Te(t)` ramps 200→400 eV. Both should
be recovered to within a few percent.
