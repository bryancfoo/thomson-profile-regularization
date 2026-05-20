#!/usr/bin/env python
"""Plot the fit result in this directory.

Run from the example directory:

    python plot.py

Produces:
- params_vs_time.png — each fitted prefix vs time, with a 16/84 percentile
                       posterior band when [sampling] was enabled
- spectra.png        — data, MAP fit, and residuals at three time slices

Auto-detects whether posterior summary stats are present in fit_result.h5
(written when sampling ran). Truth profiles overlaid when available in the
synthetic data file.
"""
import argparse
import tomllib
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# Map fit-prefix to truth-dataset name in the synthetic data file.
TRUTH_KEYS = {
    "n":       "ne_true_cm3",
    "Te0":     "Te_true_eV",
    "Ti0":     "Ti_true_eV",
    "ifract0": "ifrac_D_true",
    "ifract1": "ifrac_C_true",
}

# Canonical display order for parameter panels.
ORDER = ["n", "Te0", "Te1", "Ti0", "Ti1",
         "ifract0", "ifract1", "ifract2", "ifract3",
         "ue0", "ue1", "ui0", "ui1",
         "pe0", "pi0", "efract0"]


def load_result(path):
    out = {"params": {}, "summary": None}
    with h5py.File(path, "r") as fh:
        out["best_fit"] = fh["best_fit"][...]
        out["time"] = fh["time"][...] if "time" in fh else \
            np.arange(out["best_fit"].shape[1])
        for k in fh["params"].keys():
            out["params"][k] = fh["params"][k][...]
        out["attrs"] = dict(fh.attrs)
        if "summary" in fh:
            s = {}
            for stat in ("mean", "std", "p16", "p50", "p84"):
                s[stat] = {k: fh[f"summary/{stat}/{k}"][...]
                           for k in fh[f"summary/{stat}"].keys()}
            s["rhat"] = {k: fh[f"summary/rhat/{k}"][...]
                         for k in fh["summary/rhat"].keys()}
            s["ess"]  = {k: fh[f"summary/ess/{k}"][...]
                         for k in fh["summary/ess"].keys()}
            s["attrs"] = dict(fh["summary"].attrs)
            out["summary"] = s
    return out


def load_data(deck_path):
    """Load the data HDF5 referenced by the deck, including any truth datasets."""
    with open(deck_path, "rb") as f:
        deck = tomllib.load(f)
    data_path = (Path(deck_path).parent / deck["data"]["path"]).resolve()
    out = {"path": data_path}
    if not data_path.exists():
        return out
    with h5py.File(data_path, "r") as fh:
        out["pkl"] = fh[deck["data"]["pkl_dataset"]][...]
        out["var"] = fh[deck["data"]["var_dataset"]][...]
        if "wavelengths" in fh:
            out["wavelengths"] = fh["wavelengths"][...]
        for k in fh.keys():
            if k.endswith("_true_cm3") or k.endswith("_true_eV") \
                    or k.startswith("ifrac_"):
                out[k] = fh[k][...]
    return out


def _sorted_params(result):
    time = result["time"]
    candidates = []
    for k, v in result["params"].items():
        v = np.asarray(v)
        if v.ndim == 1 and v.size == time.size and float(np.ptp(v)) > 0:
            candidates.append(k)
    candidates.sort(key=lambda k: (ORDER.index(k) if k in ORDER else 99, k))
    return candidates


def plot_params(result, data, out_png):
    time = result["time"]
    summary = result["summary"]
    keys = _sorted_params(result)
    if not keys:
        print(f"  (no time-varying params to plot in {out_png.name})")
        return

    cols = min(3, len(keys))
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows),
                             squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[i // cols][i % cols]
        # Posterior band first (under the MAP point)
        title_extra = ""
        if summary is not None and k in summary["p16"]:
            ax.fill_between(time, summary["p16"][k], summary["p84"][k],
                            alpha=0.25, color="C0", label="68% credible")
            ax.plot(time, summary["mean"][k], "--", color="C0", alpha=0.7,
                    label="post. mean")
            with np.errstate(invalid="ignore"):
                rhat_arr = np.asarray(summary["rhat"][k])
                if np.isfinite(rhat_arr).any():
                    title_extra = f"  (max R̂={np.nanmax(rhat_arr):.2f})"
        ax.plot(time, result["params"][k], "o-", color="C0", ms=4, label="MAP")
        if k in TRUTH_KEYS and TRUTH_KEYS[k] in data:
            ax.plot(time, data[TRUTH_KEYS[k]], "k:", lw=1.5, label="truth")
        ax.set_xlabel("time")
        ax.set_ylabel(k)
        ax.set_title(k + title_extra, fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    for j in range(len(keys), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png.name}")


def plot_spectra(result, data, out_png):
    if "pkl" not in data:
        print(f"  (no data array available, skipping {out_png.name})")
        return
    pkl = data["pkl"]
    var = data["var"]
    best = result["best_fit"]
    Nt = pkl.shape[1]
    ts = sorted(set([0, Nt // 2, Nt - 1]))

    wave = data.get("wavelengths")
    if wave is not None:
        if float(wave.max()) < 1e-3:
            wave = wave * 1e9   # m → nm
        xlabel = "wavelength (nm)"
    else:
        wave = np.arange(pkl.shape[0])
        xlabel = "pixel"

    fig, axes = plt.subplots(2, len(ts), figsize=(4.2 * len(ts), 5.5),
                             sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, t in enumerate(ts):
        ax = axes[0][col]
        sig = np.sqrt(np.maximum(var[:, t], 0))
        ax.errorbar(wave, pkl[:, t], yerr=sig, fmt="o", ms=2, lw=0.5,
                    alpha=0.4, label="data")
        ax.plot(wave, best[:, t], "r-", lw=1.2, label="MAP fit")
        ax.set_ylabel("scattered power")
        ax.legend(fontsize=8)
        ax.set_title(f"t = {result['time'][t]:.3g}", fontsize=9)
        ax.grid(alpha=0.3)
        ax2 = axes[1][col]
        with np.errstate(invalid="ignore", divide="ignore"):
            resid = (pkl[:, t] - best[:, t]) / np.where(sig > 0, sig, 1)
        ax2.plot(wave, resid, ".", ms=2, color="k", alpha=0.6)
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("(d−f)/σ")
        ax2.set_ylim(-5, 5)
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", default=str(HERE / "fit_result.h5"),
                    help="HDF5 written by thomson-fit (default: fit_result.h5)")
    ap.add_argument("--deck",   default=str(HERE / "fit.toml"),
                    help="Deck used to produce the fit (default: fit.toml)")
    args = ap.parse_args()

    result = load_result(args.result)
    data   = load_data(args.deck) if Path(args.deck).exists() else {}

    print(f"Loaded {args.result}")
    print(f"  loss={result['attrs'].get('loss', float('nan')):.4g}  "
          f"nit={result['attrs'].get('nit', '?')}  "
          f"success={result['attrs'].get('success', '?')}")
    if result["summary"] is not None:
        s = result["summary"]["attrs"]
        print(f"  SGLD: {s['n_chains']} × {s['n_samples']} "
              f"(burn {s['burn_in']}, thin {s['thin']}) | "
              f"max R̂={s['max_rhat']:.2f} | min ESS={s['min_ess']:.0f}")

    plot_params(result, data, HERE / "params_vs_time.png")
    plot_spectra(result, data, HERE / "spectra.png")


if __name__ == "__main__":
    main()
