# `forward_only` — forward model without fitting

Re-computes the synthetic EPW streak from `[profiles]` in the deck.
Useful for confirming that geometry / wavelength window / probe settings
match what the fitter sees.

```bash
python examples/data/make_data_epw.py   # one-time (provides reference data)
thomson-forward forward.toml            # writes forward_streak.png
```

The output PNG overlays the computed streak with the data stored in
`../data/data_epw.h5`. Differences would indicate a deck / data mismatch.
