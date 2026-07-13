# iaw_kappa — fitting a kappa-distributed ion species

Demonstrates the `[species]` section with the built-in `kappa` model: a
single-H-ion IAW streak whose ion distribution becomes increasingly
suprathermal (κ ramping 4 → 2.2) while Ti ramps 200 → 350 eV. The kappa index
is fit per time step as the parameter `kappai0`.

```bash
python examples/data/make_data_kappa.py    # one-time
thomson-fit examples/iaw_kappa/fit.toml
```

Expected: loss ≈ 0.99 (reduced χ² of the synthetic noise), Te/Ti recovered to
~1–2%, κ tracked through its ramp to ±0.15. Compare `/params/kappai0` in
`fit_result.h5` against `kappa_true` in the data file.

Note the deck pins `ue0`/`ui0`/`efract0`/`ifract0` — free parameters with
unbounded defaults give NaN gradients (same as the original package).
