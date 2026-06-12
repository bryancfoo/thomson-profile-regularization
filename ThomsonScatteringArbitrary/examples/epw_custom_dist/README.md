# epw_custom_dist — fitting a user-supplied distribution function

Demonstrates the custom-callable path: the electron species model is the
plain JAX function `two_temp(x, fhot, rhot)` defined in `my_dists.py`
(bi-Maxwellian: cold bulk + hot fraction at `rhot`× the bulk temperature).
The deck references it as `"my_dists.py:two_temp"`; the shape parameters are
introspected from the signature and become the fit parameters `fhote0` /
`rhote0`.

The synthetic EPW streak has a hot-electron fraction growing 0.05 → 0.20 at
fixed `rhot = 4`; the fit recovers `Te0` and `fhote0` per time step
(`rhote0` pinned — degenerate with `fhot` at this noise level).

```bash
python ThomsonScatteringArbitrary/examples/data/make_data_epw_2temp.py   # one-time
python -m ThomsonScatteringArbitrary.thomson_fit ThomsonScatteringArbitrary/examples/epw_custom_dist/fit.toml
```

Expected: loss ≈ 0.98, Te to ~1%, fhot within ±0.01 of truth. See
`DECK_API.md` ("Custom-callable contract") for the function requirements
(normalization, velocity convention, differentiability).
