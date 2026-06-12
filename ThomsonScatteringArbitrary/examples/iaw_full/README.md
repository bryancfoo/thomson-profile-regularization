# `iaw_full` — IAW kitchen sink

Exercises every optional `[measurement]` feature: per-time IRF, CSV
throughput, notched wavelength band, polynomial background,
`[probe_beam]` SRS/SBS gain correction, and per-prefix `[penalty.*]`
Tikhonov terms. Optimizer: Adam.

```bash
python examples/data/make_data_iaw.py   # one-time
thomson-fit fit.toml                    # writes fit_result.h5
python plot.py                          # writes params_vs_time.png, spectra.png
```

The deck imports the throughput from `../data/throughput.csv` (file
loading) rather than the HDF5 (`data_iaw.h5:throughput`, dataset
loading). Both routes are supported; this example shows the file path.
