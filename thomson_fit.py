"""Run a Thomson scattering fit from a TOML input deck.

Usage
-----
    thomson-fit path/to/deck.toml                 # after `pip install -e .`
    thomson-fit path/to/deck.toml --sample        # MAP + posterior sampling
    python thomson_fit.py path/to/deck.toml
    python thomson_fit.py                         # interactive prompt

The deck schema follows the standard sections consumed by
``ThomsonScattering.deck.build_settings_from_deck`` (``[data]``,
``[measurement]``, ``[probe_beam]``, ``[params]``, ``[penalty]``, ``[fit]``,
``[output]``, ``[[extra_params]]``, ``[constraints]``, ``[sampling]``) plus
a few extensions resolved here before the standard parser runs:

  [measurement.throughput_xlsx]   Read xlsx → smooth → interp → throughput[].
  [measurement.irf_hdf4]          Read HDF4 → crop → mean → instr_func_arr[].
  [plotting]                      Save initial-guess, streak, and profile pngs.

The ``--sample`` flag (or ``[sampling].enabled = true`` in the deck) triggers
multi-chain preconditioned SGLD posterior sampling after the MAP fit. The
posterior summary and (by default) the raw chains both land in the primary
HDF5 — see ``DECK_API.md`` for the layout.

The deck file's directory is the base for all relative paths it references.
See ``DECK_API.md`` at the repo root for the full deck schema.
"""

import argparse
import os
import sys
from pathlib import Path


def _bootstrap_cpu_devices():
    """Resolve --n-devices (or THOMSON_CPU_DEVICES) into the XLA device-count
    flag *before* JAX is imported below. Used for intra-fit time-axis sharding
    of a single streak fit. (For L-curve sweeps use --n-workers instead; don't
    combine the two — they would oversubscribe the cores.)
    """
    # Global kill-switch: --serial (or THOMSON_NO_PARALLEL) disables both the
    # process pool and device sharding. Honor it here, before JAX initializes,
    # so no extra devices are created.
    serial = "--serial" in sys.argv or os.environ.get(
        "THOMSON_NO_PARALLEL", "").strip().lower() not in ("", "0", "false", "no")
    if serial:
        os.environ["THOMSON_NO_PARALLEL"] = "1"
        return

    n = os.environ.get("THOMSON_CPU_DEVICES")
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--n-devices" and i + 1 < len(argv):
            n = argv[i + 1]
        elif a.startswith("--n-devices="):
            n = a.split("=", 1)[1]
    if not n:
        return
    os.environ["THOMSON_CPU_DEVICES"] = str(n)
    try:
        ni = int(n)
    except (TypeError, ValueError):
        return
    if ni <= 1:
        return
    flag = "--xla_force_host_platform_device_count"
    cur = os.environ.get("XLA_FLAGS", "")
    if flag not in cur:
        os.environ["XLA_FLAGS"] = (cur + f" {flag}={ni}").strip()


_bootstrap_cpu_devices()

# Force float64 BEFORE any other code touches jax.numpy. Some downstream
# operations (e.g. Hessian-preconditioned sampling) need the extra precision.
import jax as _jax
_jax.config.update("jax_enable_x64", True)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from ThomsonScattering.deck import (
    load_deck,
    build_settings_from_deck,
    save_fit_results,
)
from ThomsonScattering.fitting import run_fit_grad, compute_initial_fit, _build_grad_problem


# ─── deck preprocessing extensions ─────────────────────────────────────────────

def _resolve_throughput_xlsx(deck, base_dir, wavelengths_m):
    """If [measurement.throughput_xlsx] is present, read xlsx → throughput array."""
    cfg = deck.get("measurement", {}).pop("throughput_xlsx", None)
    if cfg is None:
        return
    df = pd.read_excel(base_dir / cfg["path"])
    lam = np.asarray(df[cfg.get("lam_col", "Lambda")])
    val = np.asarray(df[cfg.get("value_col", "Sensitivity No Grating")])
    mask = ~np.isnan(val)
    lam, val = lam[mask], val[mask]
    sigma = cfg.get("gaussian_sigma", 0)
    if sigma:
        val = gaussian_filter(val, sigma)
    lam_unit = cfg.get("lam_unit", "nm")
    if lam_unit == "nm":
        lam_m = lam * 1e-9
    elif lam_unit == "m":
        lam_m = lam
    else:
        raise ValueError(
            f"[measurement.throughput_xlsx] lam_unit must be 'nm' or 'm', got {lam_unit!r}"
        )
    deck["measurement"]["throughput"] = np.interp(wavelengths_m, lam_m, val)


def _resolve_irf_hdf4(deck, base_dir, Nk, Nt):
    """If [measurement.irf_hdf4] is present, build a per-slice (Nk, Nt) IRF array."""
    cfg = deck.get("measurement", {}).pop("irf_hdf4", None)
    if cfg is None:
        return
    from pyhdf.SD import SD, SDC

    hdf = SD(str(base_dir / cfg["path"]), SDC.READ)
    raw = np.asarray(hdf.select(cfg.get("dataset", "Streak_array")))
    hdf.end()
    instr_data = raw[0] / raw[1] - 1.0

    sigma = cfg.get("gaussian_sigma_2d")
    if sigma:
        instr_data = gaussian_filter(instr_data, sigma=tuple(sigma))

    center = int(cfg.get("center_index", 512))
    half = Nk // 2
    instr_cropped = instr_data[:, center - half : center - half + Nk]
    if cfg.get("flip_wavelength", False):
        instr_cropped = instr_cropped[:, ::-1]

    slice_mode = cfg.get("slice_mode", "uniform")
    instr_func_arr = np.zeros((Nk, Nt))
    if slice_mode == "uniform":
        col = np.mean(instr_cropped, axis=0)
        col = col / col.sum()
        for t in range(Nt):
            instr_func_arr[:, t] = col
    elif slice_mode == "per_slice":
        N_avg = cfg.get("N_avg")
        if N_avg is None:
            raise ValueError(
                "[measurement.irf_hdf4] slice_mode='per_slice' requires N_avg"
            )
        for t in range(Nt):
            col = np.mean(instr_cropped[t * N_avg : (t + 1) * N_avg, :], axis=0)
            instr_func_arr[:, t] = col / col.sum()
    else:
        raise ValueError(
            f"[measurement.irf_hdf4] slice_mode must be 'uniform'|'per_slice', got {slice_mode!r}"
        )

    deck["measurement"]["instr_func_arr"] = instr_func_arr


# ─── plotting ──────────────────────────────────────────────────────────────────

def _plot_initial_guess(measurement, params_settings, extra_params, Pkl_data,
                        lam_nm, time_axis, png_path, shot_num):
    Nt = Pkl_data.shape[1]
    init_fit = np.asarray(
        compute_initial_fit(measurement, params_settings, extra_params, Nt)
    )
    mid = Nt // 2
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lam_nm, Pkl_data[:, mid], label="Data")
    ax.plot(lam_nm, init_fit[:, mid], linestyle="--", label="Initial guess")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Pkl (normalized)")
    ax.set_title(f"Shot {shot_num} — initial guess (t = {time_axis[mid]:.2f} ns)")
    ax.legend()
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {png_path}...")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_streak(Pkl_data, best_fit, lam_nm, time_axis, png_path, shot_num):
    fig, (axd, axf) = plt.subplots(ncols=2, figsize=(12, 5), sharey=True)
    vmin = float(np.nanmin(Pkl_data))
    vmax = float(np.nanmax(Pkl_data))
    im_d = axd.pcolormesh(time_axis, lam_nm, Pkl_data, vmin=vmin, vmax=vmax)
    axd.set_xlabel("Time [ns]")
    axd.set_ylabel("Wavelength [nm]")
    axd.set_title("Data")
    fig.colorbar(im_d, ax=axd)
    im_f = axf.pcolormesh(time_axis, lam_nm, best_fit, vmin=vmin, vmax=vmax)
    axf.set_xlabel("Time [ns]")
    axf.set_title("Best fit")
    fig.colorbar(im_f, ax=axf)
    fig.suptitle(f"Shot {shot_num} fit")
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {png_path}...")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_profiles_iaw(profiles, time_axis, png_path, shot_num):
    fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(18, 8))
    (ax_n, ax_Te, ax_Ti, ax_ui), (ax_ue, ax_pe, ax_ifract, ax_blank) = axes

    def _p(ax, key, title, ylabel):
        if key in profiles:
            ax.plot(time_axis, profiles[key])
        ax.set_xlabel("Time [ns]"); ax.set_ylabel(ylabel); ax.set_title(title)

    _p(ax_n,  "n",   "Electron density",      r"$n_e$ [cm$^{-3}$]")
    _p(ax_Te, "Te0", "Electron temperature",  r"$T_e$ [eV]")
    _p(ax_ue, "ue0", "Electron drift",        r"$u_e$ [m/s]")
    _p(ax_pe, "pe0", "Super-Gaussian order",  r"$p_e$")

    for key, label in [("Ti0", r"$T_{i,0}$ (D/C ctr)"), ("Ti1", r"$T_{i,1}$ (shifted)"), ("Ti2", r"$T_{i,2}$ (shifted)")]:
        if key in profiles:
            ax_Ti.plot(time_axis, profiles[key], label=label)
    ax_Ti.set_xlabel("Time [ns]"); ax_Ti.set_ylabel(r"$T_i$ [eV]")
    ax_Ti.set_title("Ion temperatures"); ax_Ti.legend(fontsize=7)

    for key, label in [("ui0", "center"), ("ui1", "positive"), ("ui2", "negative")]:
        if key in profiles:
            ax_ui.plot(time_axis, profiles[key], label=label)
    ax_ui.set_xlabel("Time [ns]"); ax_ui.set_ylabel(r"$u_i$ [m/s]")
    ax_ui.set_title("Ion velocities"); ax_ui.legend(fontsize=7)

    ifract_labels = ["0 (D ctr)", "1 (D+)", "2 (D−)", "3 (C ctr)", "4 (C+)", "5 (C−)"]
    for k, lbl in enumerate(ifract_labels):
        key = f"ifract{k}"
        if key in profiles:
            ax_ifract.plot(time_axis, profiles[key], label=lbl)
    ax_ifract.set_xlabel("Time [ns]"); ax_ifract.set_ylabel("ifract")
    ax_ifract.set_title("Ion charge fractions"); ax_ifract.legend(fontsize=6)

    ax_blank.set_visible(False)
    fig.suptitle(f"Shot {shot_num} IAW fitted profiles")
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {png_path}...")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_profiles_epw(profiles, time_axis, png_path, shot_num):
    fig, axes = plt.subplots(ncols=3, figsize=(13, 4))
    layout = [
        ("n",  "Electron density",     r"$n_e$ [cm$^{-3}$]"),
        ("Te", "Electron temperature", r"$T_e$ [eV]"),
        ("pe", "Super-Gaussian order", r"$p_e$"),
    ]
    for ax, (key, title, ylabel) in zip(axes, layout):
        if key not in profiles:
            ax.set_visible(False)
            continue
        ax.plot(time_axis, profiles[key])
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    fig.suptitle(f"Shot {shot_num} fitted profiles")
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {png_path}...")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_YLABEL_MAP = {
    "n":      r"$n_e$ [cm$^{-3}$]",
    "Te":     r"$T_e$ [eV]",
    "Ti":     r"$T_i$ [eV]",
    "ue":     r"$u_e$ [m/s]",
    "ui":     r"$u_i$ [m/s]",
    "pe":     r"$p_e$",
    "pi":     r"$p_i$",
    "efract": "efract",
    "ifract": "ifract",
    "bg":     "background coef",
}


def _ylabel_for_key(key):
    base = key.rstrip("0123456789")
    return _YLABEL_MAP.get(base, key)


def _normalize_profile_groups(profile_vars):
    """Normalize ``profile_vars`` into a list of groups, one group per subplot.

    Accepts either the legacy flat form ``["n", "Te0", "Ti0"]`` (each key on its
    own subplot) or the grouped form ``[["Ti0", "Ti1"], ["ui0", "ui1"]]`` (each
    inner list overlaid on a shared subplot). The two forms may be mixed: a bare
    string is treated as a singleton group.
    """
    groups = []
    for entry in profile_vars:
        if isinstance(entry, str):
            groups.append([entry])
        else:
            keys = [k for k in entry if k]
            if keys:
                groups.append(list(keys))
    return groups


def _group_title(group):
    """Title for a subplot: the shared base name if every key shares one
    (e.g. ``["Ti0", "Ti1"]`` -> ``"Ti"``), otherwise the keys joined."""
    bases = {k.rstrip("0123456789") for k in group}
    if len(bases) == 1:
        return next(iter(bases))
    return ", ".join(group)


def _plot_profiles_generic(profiles, time_axis, png_path, shot_num, profile_vars):
    """Plot grouped profile keys, one subplot per group, auto-laid-out up to 3
    per row. Keys within a group are overlaid on the same axes with a legend."""
    import math as _math
    groups = _normalize_profile_groups(profile_vars)
    n_groups = len(groups)
    if n_groups == 0:
        return
    ncols = min(3, n_groups)
    nrows = _math.ceil(n_groups / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    for idx, group in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        plotted = [k for k in group if k in profiles]
        for key in plotted:
            ax.plot(time_axis, profiles[key], label=key)
        missing = [k for k in group if k not in profiles]
        if not plotted:
            ax.text(0.5, 0.5, f"{', '.join(missing)} not in results",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
        elif len(plotted) > 1:
            ax.legend(fontsize="small")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel(_ylabel_for_key(group[0]))
        ax.set_title(_group_title(group))
    for idx in range(n_groups, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(f"Shot {shot_num} fitted profiles")
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {png_path}...")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── L-curve phase ─────────────────────────────────────────────────────────────

def _run_l_curve_phase(Pkl_data, Pkl_var, meas, pen, pars, extras, constraints,
                       fit_kw, l_curve_settings):
    """Run a Tikhonov L-curve sweep. Returns ``(result, best_fit, lc)`` where
    ``result``/``best_fit`` correspond to the max-curvature ("optimal") λ.
    """
    from ThomsonScattering.l_curve import compute_L_curve

    if not pen:
        raise ValueError(
            "[l_curve] requires at least one [penalty.<prefix>] section in the "
            "deck — the sweep multiplies those base lambda_weights by lambda_scale."
        )

    lambda_scale = l_curve_settings.get("lambda_scale")
    if lambda_scale is None:
        raise ValueError(
            "[l_curve] lambda_scale not resolved by the deck parser — "
            "this is a bug in build_settings_from_deck."
        )
    warm_start = bool(l_curve_settings.get("warm_start", True))
    n_workers = l_curve_settings.get("n_workers")

    print(f"\nRunning L-curve sweep over {len(lambda_scale)} lambda_scale "
          f"values (warm_start={warm_start})...")
    lc = compute_L_curve(
        Pkl_data, Pkl_var, meas,
        penalty_settings=pen,
        lambda_scale=lambda_scale,
        params_settings=pars,
        constraints=constraints,
        extra_params=extras,
        fit_settings=fit_kw,
        warm_start=warm_start,
        progress=True,
        n_workers=n_workers,
    )
    return lc.optimal_result, lc.optimal_best_fit, lc


def _plot_l_curve(lc, png_path, shot_num):
    """Log-log plot of penalty_norm vs. residual_norm with the corner marked."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(lc.residual_norm, lc.penalty_norm, "o-", color="C0",
              label="L-curve")
    i = int(lc.optimal_index)
    ax.loglog(lc.residual_norm[i], lc.penalty_norm[i], "o", color="C3",
              markersize=12, fillstyle="none", markeredgewidth=2,
              label=f"corner (λ_scale={lc.lambda_scale[i]:.3g})")
    for j, s in enumerate(lc.lambda_scale):
        ax.annotate(f"{s:.2g}",
                    (lc.residual_norm[j], lc.penalty_norm[j]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points",
                    color="gray")
    ax.set_xlabel("Residual norm  (data chi² / N_pix)")
    ax.set_ylabel("Penalty norm  R(x)  (base λ weighting)")
    ax.set_title(f"Shot {shot_num} — Tikhonov L-curve")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {png_path}...")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── sampling phase ────────────────────────────────────────────────────────────

def _run_sampling_phase(Pkl_data, Pkl_var, meas, pen, pars, extras, constraints,
                        map_result, sampling_settings):
    """Run multi-chain SGLD given a finished MAP fit. Returns sampling result."""
    from ThomsonScattering.sampling import run_sgld_posterior

    # shard_time=False: the per-chain vmap in the sampler does not compose with
    # the shard_map inside the objective, so the sampler always uses the
    # unsharded objective (it parallelizes over chains instead).
    problem = _build_grad_problem(
        Pkl_data, Pkl_var, meas,
        penalty_settings=pen, params_settings=pars,
        constraints=constraints, extra_params=extras,
        shard_time=False,
    )
    x_phys = np.asarray(map_result.x)
    u_map = np.array([
        problem.to_internal_np(x_phys[i], problem.lower_np[i], problem.upper_np[i])
        for i in range(len(x_phys))
    ])

    # Map deck fields → run_sgld_posterior kwargs. Defaults match the
    # function signature except for the temperature sentinel.
    s = dict(sampling_settings)
    s.pop("enabled", None)
    s.pop("save_samples", None)
    s.pop("save_cross_corr", None)
    # Temperature default sentinel: "auto" → None inside the sampler
    if "temperature" not in s:
        s["temperature"] = None
    elif s["temperature"] == "auto":
        s["temperature"] = None

    print("\nRunning SGLD posterior sampling...")
    samp = run_sgld_posterior(problem, u_map, progress=True, **s)
    print(
        f"\nSGLD: {samp.n_chains} chains × {samp.n_samples} samples "
        f"(burn {samp.burn_in}, thin {samp.thin}) | "
        f"step={samp.step_size_final:.2e} | T={samp.temperature:.2e}"
    )
    print(
        f"  max R-hat = {samp.max_rhat:.3f} ({samp.max_rhat_key}) | "
        f"min ESS = {samp.min_ess:.0f} ({samp.min_ess_key}) | "
        f"wall {samp.wall_time:.1f}s"
    )
    if samp.max_rhat > 1.1:
        print(
            f"  WARNING: max R-hat = {samp.max_rhat:.2f} > 1.1; "
            "chains may not have mixed. Consider larger burn_in or "
            "n_samples, or a different step_size."
        )
    if samp.min_ess < 50:
        print(
            f"  WARNING: min ESS = {samp.min_ess:.0f} < 50; "
            "effective sample size is small for at least one coordinate."
        )
    return samp


# ─── main ──────────────────────────────────────────────────────────────────────

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="thomson-fit",
        description="Run a Thomson-scattering fit (and optionally posterior "
                    "sampling) from a TOML input deck.",
    )
    parser.add_argument("deck", nargs="?",
                        help="Path to the input deck (.toml). If omitted, "
                             "you'll be prompted interactively.")
    parser.add_argument("--sample", action="store_true",
                        help="Run multi-chain SGLD posterior sampling after "
                             "the MAP fit. Overrides [sampling].enabled in "
                             "the deck if set.")
    parser.add_argument("--l-curve", dest="l_curve", action="store_true",
                        help="Run a Tikhonov L-curve sweep instead of a "
                             "single MAP fit. Requires an [l_curve] section "
                             "in the deck (for lambda_scale). The optimal-"
                             "lambda fit becomes the saved best fit.")
    parser.add_argument("--n-workers", dest="n_workers", type=int, default=None,
                        help="Number of parallel worker PROCESSES for the L-curve "
                             "sweep (independent fits run at once; each uses ~3-4 "
                             "cores — this is not a core count). Default 1 "
                             "(sequential). N>1 runs N fits at a time; 0 or "
                             "negative auto-sizes to cores//4. Overrides "
                             "[l_curve].n_workers in the deck.")
    parser.add_argument("--n-devices", dest="n_devices", type=int, default=None,
                        help="Expose the host CPU as N XLA devices and shard a "
                             "single fit's forward+grad over the time axis across "
                             "them (intra-fit parallelism). Read before JAX init "
                             "(see _bootstrap_cpu_devices). Use for one big streak "
                             "fit; do NOT combine with --n-workers > 1.")
    parser.add_argument("--serial", action="store_true",
                        help="Kill-switch: disable ALL parallelism (process pool "
                             "and device sharding) regardless of other flags or "
                             "deck settings. Equivalent to THOMSON_NO_PARALLEL=1. "
                             "Use on small/shared machines to keep core usage low.")
    return parser.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)
    if args.deck:
        raw = args.deck
    else:
        raw = input("Path to input deck (.toml): ").strip()
    deck_path = Path(raw).expanduser().resolve()

    if not deck_path.exists():
        raise FileNotFoundError(f"Deck file not found: {deck_path}")

    deck_text = deck_path.read_text(encoding="utf-8")
    deck = load_deck(deck_path)
    base_dir = deck["_base_dir"]

    # Resolve deck extensions before build_settings_from_deck consumes the rest.
    # We need wavelengths (Nk-array) for throughput interp, and Nt for the IRF.
    data_sec = deck["data"]
    with h5py.File(base_dir / data_sec["path"], "r") as hf:
        Nt = hf[data_sec["pkl_dataset"]].shape[1]

    meas_sec = deck.setdefault("measurement", {})
    wavelengths_m = meas_sec["wavelengths"]
    if isinstance(wavelengths_m, str):
        if ".h5:" in wavelengths_m:
            file_part, dset = wavelengths_m.split(".h5:", 1)
            with h5py.File(base_dir / (file_part + ".h5"), "r") as hf:
                wavelengths_m = np.asarray(hf[dset])
        else:
            raise ValueError(
                f"[measurement].wavelengths string must be 'file.h5:dataset', got {wavelengths_m!r}"
            )
    else:
        wavelengths_m = np.asarray(wavelengths_m)
    Nk = len(wavelengths_m)

    _resolve_throughput_xlsx(deck, base_dir, wavelengths_m)
    _resolve_irf_hdf4(deck, base_dir, Nk, Nt)

    (
        Pkl_data, Pkl_var,
        meas, pen, pars, fit_kw,
        extras, constraints, out_path,
        sampling_settings,
        l_curve_settings,
    ) = build_settings_from_deck(deck)

    print(f"\nRunning fit  Nt={Pkl_data.shape[1]}  Nk={Pkl_data.shape[0]}")

    # Plot initial-guess diagnostic before launching the optimizer.
    plot_cfg = deck.get("plotting", {})
    shot_num = plot_cfg.get("shot_num", "?")
    lam_nm = np.asarray(meas["wavelengths"]) * 1e9
    time_axis = None
    for ps in (pen or {}).values():
        ax = ps.get("profile_axis")
        if ax is not None:
            time_axis = np.asarray(ax)
            break
    if time_axis is None:
        time_axis = np.arange(Pkl_data.shape[1])

    init_png = plot_cfg.get("init_png")
    if init_png and pars is not None:
        _plot_initial_guess(meas, pars, extras, np.asarray(Pkl_data),
                            lam_nm, time_axis, base_dir / init_png, shot_num)
        print(f"Wrote {init_png}")

    if args.serial:
        os.environ["THOMSON_NO_PARALLEL"] = "1"
    if args.n_workers is not None:
        l_curve_settings["n_workers"] = args.n_workers

    do_l_curve = args.l_curve or bool(l_curve_settings.get("enabled", False))
    l_curve_result = None
    if do_l_curve:
        result, best_fit, l_curve_result = _run_l_curve_phase(
            Pkl_data, Pkl_var, meas, pen, pars, extras, constraints, fit_kw,
            l_curve_settings,
        )
        opt_i  = l_curve_result.optimal_index
        opt_ls = float(l_curve_result.lambda_scale[opt_i])
        print(f"\nL-curve: optimal index={opt_i}  lambda_scale={opt_ls:.4g}  "
              f"residual={l_curve_result.residual_norm[opt_i]:.4g}  "
              f"penalty={l_curve_result.penalty_norm[opt_i]:.4g}")
        print(f"loss={result.fun:.6g}  nit={result.nit}  success={result.success}")
    else:
        result, best_fit = run_fit_grad(
            Pkl_data, Pkl_var, meas,
            penalty_settings=pen,
            params_settings=pars,
            fit_settings=fit_kw,
            extra_params=extras,
            constraints=constraints,
            progress=True,
        )
        print(f"\nloss={result.fun:.6g}  nit={result.nit}  success={result.success}")

    best_fit_np = np.asarray(best_fit)

    # ── Optional: posterior sampling ───────────────────────────────────────
    sampling_result = None
    do_sample = args.sample or bool(sampling_settings.get("enabled", False))
    if do_sample and do_l_curve:
        print("\nWARNING: both [l_curve] and [sampling] are enabled. "
              "Sampling around a single MAP point is ill-defined when the "
              "regularization strength is itself being swept; skipping the "
              "sampling phase. Re-run without --l-curve (or set "
              "[l_curve].enabled = false) to sample at the deck's base lambda.")
    elif do_sample:
        sampling_result = _run_sampling_phase(
            Pkl_data, Pkl_var, meas, pen, pars, extras, constraints,
            result, sampling_settings,
        )

    save_cross_corr = bool(sampling_settings.get("save_cross_corr", True))
    save_samples    = bool(sampling_settings.get("save_samples", True))
    save_fit_results(
        out_path, result, best_fit_np,
        deck_text=deck_text, time_axis=time_axis,
        sampling_result=sampling_result,
        save_cross_corr=save_cross_corr,
        save_samples=save_samples,
        l_curve_result=l_curve_result,
    )
    print(f"Results saved to: {out_path}")

    if l_curve_result is not None:
        l_curve_png = l_curve_settings.get("plot_path")
        if l_curve_png:
            _plot_l_curve(l_curve_result, base_dir / l_curve_png, shot_num)
            print(f"Wrote {l_curve_png}")

    # Read the per-prefix profiles back for plotting and optional legacy layout.
    with h5py.File(out_path, "r") as hf:
        profiles = {k: np.asarray(hf["params"][k]) for k in hf["params"]}

    streak_png = plot_cfg.get("streak_png")
    if streak_png:
        _plot_streak(np.asarray(Pkl_data), best_fit_np, lam_nm, time_axis,
                     base_dir / streak_png, shot_num)
        print(f"Wrote {streak_png}")

    profiles_png = plot_cfg.get("profiles_png")
    profile_vars = plot_cfg.get("profile_vars")
    if profiles_png:
        if profile_vars is not None:
            _plot_profiles_generic(profiles, time_axis, base_dir / profiles_png,
                                   shot_num, profile_vars)
            print(f"Wrote {profiles_png}")
        else:
            layout = plot_cfg.get("profile_layout", "epw")
            if layout == "epw":
                plot_profiles = {k: profiles[k] for k in ("n", "Te", "pe") if k in profiles}
                _plot_profiles_epw(plot_profiles, time_axis, base_dir / profiles_png, shot_num)
                print(f"Wrote {profiles_png}")
            elif layout == "iaw":
                _plot_profiles_iaw(profiles, time_axis, base_dir / profiles_png, shot_num)
                print(f"Wrote {profiles_png}")
            else:
                print(f"Skipping profiles plot: layout={layout!r} not implemented.")


if __name__ == "__main__":
    main()
