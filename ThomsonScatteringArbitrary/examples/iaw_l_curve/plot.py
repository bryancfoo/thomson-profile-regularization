#!/usr/bin/env python
"""Plot the L-curve fit result in this directory.

Run from the example directory:

    python plot.py

Produces:
- params_vs_time.png    — MAP profiles at the *optimal* lambda_scale,
                          with truth overlaid.
- spectra.png           — data vs MAP fit at three time slices.
- l_curve_profiles.png  — the L-curve itself plus Te0 / Ti0 profile
                          families colored by lambda_scale, with the
                          corner pick highlighted. This is the figure
                          that shows the regularization trade-off most
                          directly.

The `l_curve.png` written by `thomson-fit` itself is the bare L-curve.
This script adds the per-λ profile overlay on top.
"""
import argparse
import tomllib
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

HERE = Path(__file__).resolve().parent

TRUTH_KEYS = {
    "n":       "ne_true_cm3",
    "Te0":     "Te_true_eV",
    "Ti0":     "Ti_D_true_eV",
    "ifract0": "ifrac_D_true",
    "ifract1": "ifrac_C_true",
}

ORDER = ["n", "Te0", "Te1", "Ti0", "Ti1",
         "ifract0", "ifract1", "ifract2", "ifract3",
         "ue0", "ue1", "ui0", "ui1",
         "pe0", "pi0", "efract0"]


def load_result(path):
    out = {"params": {}, "l_curve": None}
    with h5py.File(path, "r") as fh:
        out["best_fit"] = fh["best_fit"][...]
        out["time"] = fh["time"][...] if "time" in fh else \
            np.arange(out["best_fit"].shape[1])
        for k in fh["params"].keys():
            out["params"][k] = fh["params"][k][...]
        out["attrs"] = dict(fh.attrs)
        if "l_curve" in fh:
            lc = {
                "lambda_scale":  fh["l_curve/lambda_scale"][...],
                "residual_norm": fh["l_curve/residual_norm"][...],
                "penalty_norm":  fh["l_curve/penalty_norm"][...],
                "curvature":     fh["l_curve/curvature"][...],
                "loss":          fh["l_curve/loss"][...],
                "best_fits":     fh["l_curve/best_fits"][...],
                "params":        {k: fh[f"l_curve/params/{k}"][...]
                                  for k in fh["l_curve/params"].keys()},
                "attrs":         dict(fh["l_curve"].attrs),
            }
            if "unreg" in fh["l_curve"]:
                lc["unreg_params"] = {
                    k: fh[f"l_curve/unreg/params/{k}"][...]
                    for k in fh["l_curve/unreg/params"].keys()
                }
            out["l_curve"] = lc
    return out


def load_data(deck_path):
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
        ax.plot(time, result["params"][k], "o-", color="C0", ms=4,
                label="optimal-λ MAP")
        if k in TRUTH_KEYS and TRUTH_KEYS[k] in data:
            ax.plot(time, data[TRUTH_KEYS[k]], "k:", lw=1.5, label="truth")
        ax.set_xlabel("time")
        ax.set_ylabel(k)
        ax.set_title(k, fontsize=9)
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
    pkl = data["pkl"]; var = data["var"]; best = result["best_fit"]
    Nt = pkl.shape[1]
    ts = sorted(set([0, Nt // 2, Nt - 1]))
    wave = data.get("wavelengths")
    if wave is not None:
        if float(wave.max()) < 1e-3:
            wave = wave * 1e9
        xlabel = "wavelength (nm)"
    else:
        wave = np.arange(pkl.shape[0]); xlabel = "pixel"
    fig, axes = plt.subplots(2, len(ts), figsize=(4.2 * len(ts), 5.5),
                             sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, t in enumerate(ts):
        ax = axes[0][col]
        sig = np.sqrt(np.maximum(var[:, t], 0))
        ax.errorbar(wave, pkl[:, t], yerr=sig, fmt="o", ms=2, lw=0.5,
                    alpha=0.4, label="data")
        ax.plot(wave, best[:, t], "r-", lw=1.2, label="MAP fit (optimal λ)")
        ax.set_ylabel("scattered power"); ax.legend(fontsize=8)
        ax.set_title(f"t = {result['time'][t]:.3g}", fontsize=9)
        ax.grid(alpha=0.3)
        ax2 = axes[1][col]
        with np.errstate(invalid="ignore", divide="ignore"):
            resid = (pkl[:, t] - best[:, t]) / np.where(sig > 0, sig, 1)
        ax2.plot(wave, resid, ".", ms=2, color="k", alpha=0.6)
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.set_xlabel(xlabel); ax2.set_ylabel("(d−f)/σ")
        ax2.set_ylim(-5, 5); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png.name}")


def plot_l_curve_with_profiles(result, data, out_png):
    """Composite figure: L-curve on the left, Te0 / Ti0 profile families
    on the right, all colored by lambda_scale. Optimal-λ point and curve
    are emphasized."""
    lc = result["l_curve"]
    if lc is None:
        print("  (no /l_curve group in result file; skipping)")
        return
    ls = lc["lambda_scale"]
    r  = lc["residual_norm"]
    p  = lc["penalty_norm"]
    opt = int(lc["attrs"]["optimal_index"])
    time = result["time"]

    norm = mcolors.LogNorm(vmin=max(ls.min(), 1e-30), vmax=max(ls.max(), 1e-29))
    cmap = plt.get_cmap("viridis")
    colors = cmap(norm(ls))

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.30)
    ax_lc = fig.add_subplot(gs[:, 0])
    ax_te = fig.add_subplot(gs[0, 1:])
    ax_ti = fig.add_subplot(gs[1, 1:], sharex=ax_te)

    # L-curve panel
    ax_lc.plot(r, p, "-", color="gray", lw=1, alpha=0.6, zorder=1)
    for i in range(len(ls)):
        ax_lc.plot(r[i], p[i], "o", color=colors[i], ms=8, zorder=2)
    ax_lc.plot(r[opt], p[opt], "o", color="none", mec="red",
               mew=2.0, ms=18, zorder=3,
               label=f"corner (λ={ls[opt]:.3g})")
    ax_lc.set_xscale("log"); ax_lc.set_yscale("log")
    ax_lc.set_xlabel("residual norm  (data χ²/N_pix)")
    ax_lc.set_ylabel("penalty norm  R(x)  (base-λ weighting)")
    ax_lc.set_title("L-curve", fontsize=11)
    ax_lc.legend(fontsize=8)
    ax_lc.grid(True, which="both", ls=":", alpha=0.5)

    # Profile families
    for ax, key, ylabel, truth_key in [
        (ax_te, "Te0", "Te₀ [eV]", TRUTH_KEYS.get("Te0")),
        (ax_ti, "Ti0", "Ti₀ [eV]", TRUTH_KEYS.get("Ti0")),
    ]:
        if key not in lc["params"]:
            ax.axis("off"); continue
        all_profiles = lc["params"][key]   # (N_lambda, Nt)
        for i in range(len(ls)):
            ax.plot(time, all_profiles[i], "-", color=colors[i],
                    lw=1.0, alpha=0.85)
        ax.plot(time, all_profiles[opt], "-", color="red", lw=2.5,
                label=f"optimal (λ={ls[opt]:.3g})")
        if truth_key is not None and truth_key in data:
            ax.plot(time, data[truth_key], "k:", lw=1.5, label="truth")
        ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    ax_ti.set_xlabel("time [ns]")
    ax_te.tick_params(labelbottom=False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_te, ax_ti], pad=0.02)
    cbar.set_label("lambda_scale", rotation=270, labelpad=15)

    fig.suptitle("Tikhonov L-curve sweep — profiles by λ", fontsize=12)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", default=str(HERE / "fit_result.h5"))
    ap.add_argument("--deck",   default=str(HERE / "fit.toml"))
    args = ap.parse_args()

    result = load_result(args.result)
    data   = load_data(args.deck) if Path(args.deck).exists() else {}

    print(f"Loaded {args.result}")
    print(f"  loss={result['attrs'].get('loss', float('nan')):.4g}  "
          f"nit={result['attrs'].get('nit', '?')}  "
          f"success={result['attrs'].get('success', '?')}")
    if result["l_curve"] is not None:
        a = result["l_curve"]["attrs"]
        print(f"  L-curve: optimal_index={a['optimal_index']}  "
              f"optimal_lambda_scale={a['optimal_lambda_scale']:.4g}  "
              f"warm_start={bool(a['warm_start'])}")

    plot_params(result, data, HERE / "params_vs_time.png")
    plot_spectra(result, data, HERE / "spectra.png")
    plot_l_curve_with_profiles(result, data, HERE / "l_curve_profiles.png")


if __name__ == "__main__":
    main()
