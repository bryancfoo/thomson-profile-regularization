"""Run a Thomson scattering fit from a TOML input deck.

Usage
-----
    thomson-fit path/to/deck.toml        # after `pip install -e .`
    python fit.py path/to/deck.toml
    python fit.py                        # falls back to interactive prompt

The deck schema follows the standard sections consumed by
``ThomsonScattering.utility.build_settings_from_deck`` (``[data]``,
``[measurement]``, ``[probe_beam]``, ``[params]``, ``[penalty]``, ``[fit]``,
``[output]``, ``[[extra_params]]``, ``[constraints]``) plus a few extensions
resolved here before the standard parser runs:

  [measurement.throughput_xlsx]   Read xlsx → smooth → interp → throughput[].
  [measurement.irf_hdf4]          Read HDF4 → crop → mean → instr_func_arr[].
  [output].legacy_layout = true   Also write flat top-level datasets
                                  (e.g. epw_lam, epw_time, n_fit, ...).
  [plotting]                      Save initial-guess, streak, and profile pngs.

The deck file's directory is the base for all relative paths it references.
See ``example_deck.toml`` for the full deck schema with inline documentation.
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from ThomsonScattering.utility import (
    load_deck,
    build_settings_from_deck,
    save_fit_results,
)
from ThomsonScattering.fitting import run_fit, run_fit_grad, build_params, _compute_fit


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
    Nelectrons = measurement["Nelectrons"]
    Nions = len(measurement["ion_z"])
    Nt = Pkl_data.shape[1]
    p = build_params(Nelectrons, Nions, Nt, params_settings)
    if extra_params:
        for entry in extra_params:
            ed = dict(entry)
            name = ed.pop("name")
            for t in range(Nt):
                kwargs = {
                    k: float(v[t]) if (hasattr(v, "__len__") and not isinstance(v, str)) else v
                    for k, v in ed.items()
                }
                p.add(f"{name}_{t}", **kwargs)
    init_fit = np.asarray(_compute_fit(p, measurement))
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


# ─── legacy h5 layout ──────────────────────────────────────────────────────────

def _save_legacy_layout(out_path, prefix, flatten_spec, profiles,
                        Pkl_data, Pkl_var, lam_nm, time_axis):
    """Append flat top-level datasets to an h5 already written by save_fit_results."""
    rename = {}
    for entry in flatten_spec:
        if ":" in entry:
            src, dst = entry.split(":", 1)
        else:
            src = dst = entry
        rename[src] = dst
    with h5py.File(out_path, "a") as hf:
        hf.create_dataset(f"{prefix}lam",     data=lam_nm)
        hf.create_dataset(f"{prefix}time",    data=np.asarray(time_axis))
        hf.create_dataset(f"{prefix}Pkl_avg", data=np.asarray(Pkl_data))
        hf.create_dataset(f"{prefix}Pkl_var", data=np.asarray(Pkl_var))
        for src, dst in rename.items():
            if src not in profiles:
                raise KeyError(
                    f"[output].legacy_flatten references param prefix {src!r} "
                    f"but only have {sorted(profiles.keys())}"
                )
            hf.create_dataset(f"{dst}_fit", data=np.asarray(profiles[src]))


# ─── main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        raw = argv[0]
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
        extras, constraints, out_path, backend,
    ) = build_settings_from_deck(deck)

    print(f"\nRunning fit  backend={backend!r}  Nt={Pkl_data.shape[1]}  Nk={Pkl_data.shape[0]}")

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

    if backend == "lmfit":
        result, best_fit = run_fit(
            Pkl_data, Pkl_var, meas,
            penalty_settings=pen,
            params_settings=pars,
            fit_settings=fit_kw,
            extra_params=extras,
            constraints=constraints,
            progress=True,
        )
        loss    = float(result.residual) if not hasattr(result.residual, "__len__") else float("nan")
        neval   = getattr(result, "nfev", "?")
        success = result.success
        msg     = getattr(result, "message", "")
        print(f"\nloss={loss:.6g}  nfev={neval}  success={success}  message={msg!r}")
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

    save_fit_results(out_path, result, best_fit_np, backend, deck_text=deck_text, time_axis=time_axis)
    print(f"Results saved to: {out_path}")

    # Read the per-prefix profiles back for plotting and optional legacy layout.
    with h5py.File(out_path, "r") as hf:
        profiles = {k: np.asarray(hf["params"][k]) for k in hf["params"]}

    output_cfg = deck.get("output", {})
    if output_cfg.get("legacy_layout", False):
        prefix  = output_cfg.get("legacy_prefix", "")
        flatten = output_cfg.get("legacy_flatten", [])
        _save_legacy_layout(
            out_path, prefix, flatten, profiles,
            np.asarray(Pkl_data), np.asarray(Pkl_var), lam_nm, time_axis,
        )
        print(f"Appended legacy layout to {out_path}")

    streak_png = plot_cfg.get("streak_png")
    if streak_png:
        _plot_streak(np.asarray(Pkl_data), best_fit_np, lam_nm, time_axis,
                     base_dir / streak_png, shot_num)
        print(f"Wrote {streak_png}")

    profiles_png = plot_cfg.get("profiles_png")
    layout = plot_cfg.get("profile_layout", "epw")
    if profiles_png:
        flatten = output_cfg.get("legacy_flatten", [])
        rename = {}
        for entry in flatten:
            if ":" in entry:
                src, dst = entry.split(":", 1)
            else:
                src = dst = entry
            rename[dst] = src
        plot_profiles = {}
        for legacy_name in ("n", "Te", "pe"):
            src = rename.get(legacy_name, legacy_name)
            if src in profiles:
                plot_profiles[legacy_name] = profiles[src]
        if layout == "epw":
            _plot_profiles_epw(plot_profiles, time_axis, base_dir / profiles_png, shot_num)
            print(f"Wrote {profiles_png}")
        elif layout == "iaw":
            _plot_profiles_iaw(profiles, time_axis, base_dir / profiles_png, shot_num)
            print(f"Wrote {profiles_png}")
        else:
            print(f"Skipping profiles plot: layout={layout!r} not implemented.")


if __name__ == "__main__":
    main()
