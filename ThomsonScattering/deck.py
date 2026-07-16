"""TOML input-deck loading, expansion, and result writing.

Port of the original ThomsonScattering.deck plus the ``[species]`` section
for choosing per-species distribution models:

    [species]
    electron = ["super_gaussian"]
    ion = ["maxwellian", { model = "kappa", x_max = 25.0 }]
    # or a custom callable (path relative to the deck):
    # ion = ["my_dists.py:hot_tail"]

Each entry is either a registry name (``maxwellian``, ``super_gaussian``,
``kappa``, ...), a ``"file.py:function"`` reference to a JAX callable
``g(x, *shape_params)`` (the normalized 1D reduced distribution on
x = (v − u)/vth, vth = sqrt(2T/m)), or an inline table with a ``model`` key
plus quadrature options (``x_max``, ``n_points``). When the section is
absent every species defaults to ``super_gaussian`` — identical behavior and
parameters (``pe``/``pi``) to the original package.

Shape parameters of each model become fit parameters named
``<name><e|i><species>`` (e.g. ``pe0``, ``kappai1``) and are configured via
``[params.*]`` / ``[penalty.*]`` / ``[constraints]`` exactly like the moment
parameters.

Two public entry points:
- ``load_deck(path)`` reads a TOML file and tags it with ``_base_dir`` /
  ``_deck_stem`` so downstream code can resolve relative paths.
- ``build_settings_from_deck(deck)`` turns the parsed dict into the argument
  bundle that ``run_fit_grad`` consumes.

``save_fit_results`` writes the standard HDF5 output produced by both CLI
entry points after a successful fit.
"""
import pathlib as _pathlib
import re as _re

import numpy as _np


def load_deck(deck_path):
    """Parse a TOML input deck and return the config dict.

    Adds private keys ``_base_dir`` (pathlib.Path to deck's parent) and
    ``_deck_stem`` (filename without suffix) for use by build_settings_from_deck.
    """
    try:
        import tomllib as _tomllib
    except ImportError:
        try:
            import tomli as _tomllib
        except ImportError:
            raise ImportError(
                "TOML parsing requires 'tomllib' (Python >=3.11) or "
                "'tomli' (pip install tomli) on Python <3.11."
            )
    deck_path = _pathlib.Path(deck_path).expanduser().resolve()
    with open(deck_path, "rb") as fh:
        deck = _tomllib.load(fh)
    deck["_base_dir"] = deck_path.parent
    deck["_deck_stem"] = deck_path.stem
    return deck


def _load_h5_dataset(path, dataset):
    """Return a numpy array from ``path`` (HDF5 file) at ``dataset``."""
    import h5py
    with h5py.File(path, "r") as fh:
        return fh[dataset][()]


def _load_array(value, base_dir):
    """Resolve a deck field to a numpy array.

    Accepted forms for ``value``:
    - list / tuple              → np.asarray(value)
    - ``"file.npy"``            → np.load
    - ``"file.csv|txt|dat"``    → np.loadtxt (auto comma/space)
    - ``"file.h5:dataset"``     → HDF5 dataset load
    Paths are resolved relative to ``base_dir``.
    """
    if isinstance(value, (list, tuple)):
        return _np.asarray(value)
    if not isinstance(value, str):
        raise TypeError(
            f"Expected a file path string or list, got {type(value).__name__!r}"
        )
    base_dir = _pathlib.Path(base_dir)

    if ".h5:" in value:
        file_part, dataset = value.split(".h5:", 1)
        return _load_h5_dataset(base_dir / (file_part + ".h5"), dataset)

    file_path = base_dir / value
    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        return _np.load(file_path)
    elif suffix in (".csv", ".txt", ".dat"):
        try:
            return _np.loadtxt(file_path, delimiter=",")
        except ValueError:
            return _np.loadtxt(file_path)
    elif suffix == ".h5":
        raise ValueError(
            f"HDF5 reference {value!r} is missing ':dataset' suffix. "
            "Use the form 'file.h5:dataset_name'."
        )
    else:
        raise ValueError(
            f"Cannot load array from {value!r}: unsupported extension {suffix!r}. "
            "Supported: .npy, .csv, .txt, .dat, or .h5:dataset_name"
        )


def _gaussian_irf(sigma_px, Nk):
    """Build a (L,) unit-area Gaussian IRF from a pixel-space std dev.

    The kernel is odd-length and exactly centered, so the ``mode="same"``
    convolution in the forward model imparts no wavelength shift. Its length
    covers ±4σ (negligible tails beyond), capped at the spectral window so
    the convolved output keeps length Nk. A 1-D IRF is applied uniformly at
    every time step — this mode has no time dependence by construction.
    """
    if not sigma_px > 0:
        raise ValueError(
            f"[measurement] irf_sigma_px must be a positive number of pixels, "
            f"got {sigma_px!r}"
        )
    L = 2 * int(_np.ceil(4.0 * sigma_px)) + 1
    if L > Nk:
        L = Nk if Nk % 2 == 1 else Nk - 1
    x = _np.arange(L) - L // 2
    kernel = _np.exp(-0.5 * (x / sigma_px) ** 2)
    return kernel / kernel.sum()


def _require(d, keys, section):
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(
            f"Missing required keys in {section}: {missing}. "
            "Check your input deck."
        )


def _spec_model_name(spec):
    """Model name from a species spec (string or inline table)."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        try:
            return spec["model"]
        except KeyError:
            raise ValueError(
                f"[species] table entry {spec!r} is missing the 'model' key."
            ) from None
    raise TypeError(
        f"[species] entries must be strings or tables, got {type(spec).__name__!r}"
    )


def parse_species_section(deck, measurement_settings):
    """Resolve the ``[species]`` section into measurement_settings model specs.

    Stores plain (picklable) specs under ``e_models`` / ``i_models`` plus
    ``_model_base_dir`` for resolving relative custom-callable paths; actual
    Distribution objects are built inside each fit process by
    :func:`~.distributions.resolve_models`.

    Returns ``(e_specs, i_specs)`` (either may be None when defaulted).
    """
    base_dir = deck.get("_base_dir", _pathlib.Path("."))
    species_sec = deck.get("species", {}) or {}
    e_specs = species_sec.get("electron")
    i_specs = species_sec.get("ion")

    Nelectrons = measurement_settings.get("Nelectrons")
    Nions = len(measurement_settings.get("ion_z", []))

    if e_specs is not None:
        for sp in e_specs:
            _spec_model_name(sp)  # validates form
        if Nelectrons is not None and len(e_specs) != Nelectrons:
            raise ValueError(
                f"[species] electron has {len(e_specs)} entries but "
                f"[measurement] Nelectrons = {Nelectrons}."
            )
        measurement_settings["e_models"] = list(e_specs)
    if i_specs is not None:
        for sp in i_specs:
            _spec_model_name(sp)
        if Nions and len(i_specs) != Nions:
            raise ValueError(
                f"[species] ion has {len(i_specs)} entries but "
                f"[measurement] ion_z has {Nions} entries."
            )
        measurement_settings["i_models"] = list(i_specs)
    if e_specs is not None or i_specs is not None:
        measurement_settings["_model_base_dir"] = str(base_dir)
    return e_specs, i_specs


def _super_gaussian_species(specs, n_species):
    """Indices of species using the analytic super_gaussian model."""
    if specs is None:
        return set(range(n_species))
    return {
        idx for idx, sp in enumerate(specs)
        if _spec_model_name(sp) == "super_gaussian"
    }


def build_settings_from_deck(deck):
    """Convert a parsed deck dict (from load_deck) into arguments for run_fit_grad.

    Returns
    -------
    Pkl_data            : np.ndarray (Nk, Nt)
    Pkl_var             : np.ndarray (Nk, Nt)
    measurement_settings : dict
        Includes ``e_models`` / ``i_models`` specs when a ``[species]``
        section is present (absent → all species default to super_gaussian).
    penalty_settings    : dict or None
    params_settings     : dict or None
    fit_settings        : dict
    extra_params        : list of dict or None
    constraints_settings : dict[str, str] or None
        Mapping of parameter prefix → expression string from the deck's
        ``[constraints]`` table.
    output_path         : pathlib.Path
    sampling_settings   : dict
        From the deck's ``[sampling]`` section. Always present;
        ``enabled = False`` when the section is missing.
    l_curve_settings    : dict
        From the deck's ``[l_curve]`` section. Always present;
        ``enabled = False`` when the section is missing. When enabled, the
        ``lambda_scale`` field is resolved to a 1-D numpy array (either from
        a literal list or expanded from ``lambda_scale_log``).
    """
    base_dir = deck.get("_base_dir", _pathlib.Path("."))

    # ── 1. Load data arrays (needed first to get Nt) ────────────────────────
    data_sec = deck.get("data", {})
    _require(data_sec, ["path", "pkl_dataset", "var_dataset"], section="[data]")
    h5_path = base_dir / data_sec["path"]
    Pkl_data = _load_h5_dataset(h5_path, data_sec["pkl_dataset"])
    Pkl_var  = _load_h5_dataset(h5_path, data_sec["var_dataset"])
    Nt = Pkl_data.shape[1]

    # Optional: authoritative time axis for this fit — required when any param
    # uses source_time_axis for cross-time-axis interpolation.
    fit_time_axis = None
    if "time_axis" in data_sec:
        fit_time_axis = _load_array(data_sec["time_axis"], base_dir)
        if fit_time_axis.ndim != 1 or len(fit_time_axis) != Nt:
            raise ValueError(
                f"[data] time_axis has shape {fit_time_axis.shape}; "
                f"expected ({Nt},) to match Pkl_data time dimension."
            )

    # ── 2. Build measurement_settings ───────────────────────────────────────
    meas_raw = deck.get("measurement", {})
    _require(meas_raw, [
        "Nelectrons", "ion_z", "ion_a", "probe_wavelength",
        "probe_vec", "scatter_vec", "ue_dir", "ui_dir", "wavelengths",
    ], section="[measurement]")

    _arr_fields = {
        "wavelengths", "instr_func_arr", "throughput", "aperture_weights",
        "scatter_vec", "probe_vec", "ue_dir", "ui_dir", "ion_z", "ion_a",
    }

    measurement_settings = {}
    for key, val in meas_raw.items():
        if key in _arr_fields:
            measurement_settings[key] = (
                _load_array(val, base_dir) if isinstance(val, str)
                else _np.asarray(val)
            )
        elif key == "notch" and val is not None:
            measurement_settings[key] = tuple(val)
        else:
            measurement_settings[key] = val

    # ── IRF mode ────────────────────────────────────────────────────────────
    # "array" (default): instr_func_arr is used as given — loaded from a file/
    # list here, or injected upstream by the CLI's [measurement.irf_hdf4]
    # extension. "gaussian": ignore any measured IRF and build a clean
    # unit-area Gaussian from irf_sigma_px (std dev in pixels), applied
    # identically at every time step — for when the measured IRF is too noisy
    # (negative lobes, artifacts) to convolve with directly.
    irf_mode = measurement_settings.pop("irf_mode", "array")
    irf_sigma_px = measurement_settings.pop("irf_sigma_px", None)
    if irf_mode == "gaussian":
        if irf_sigma_px is None:
            raise ValueError(
                "[measurement] irf_mode = 'gaussian' requires irf_sigma_px "
                "(Gaussian standard deviation in pixels)."
            )
        if "instr_func_arr" in measurement_settings:
            raise ValueError(
                "[measurement] irf_mode = 'gaussian' conflicts with an "
                "explicit IRF array. Remove instr_func_arr (or the "
                "[measurement.irf_hdf4] section), or set irf_mode = 'array'."
            )
        measurement_settings["instr_func_arr"] = _gaussian_irf(
            float(irf_sigma_px), len(measurement_settings["wavelengths"])
        )
    elif irf_mode == "array":
        if irf_sigma_px is not None:
            raise ValueError(
                "[measurement] irf_sigma_px only applies when irf_mode = "
                "'gaussian'; remove it or switch irf_mode."
            )
        _irf = measurement_settings.get("instr_func_arr")
        if _irf is not None and _irf.ndim not in (1, 2):
            raise ValueError(
                f"[measurement] instr_func_arr has shape {_irf.shape}; "
                f"expected (L,) for a time-independent IRF or (L, Nt) with "
                f"one column per time step."
            )
    else:
        raise ValueError(
            f"[measurement] irf_mode must be 'gaussian' or 'array', got {irf_mode!r}"
        )

    # Per-species distribution models from the [species] section.
    e_specs, i_specs = parse_species_section(deck, measurement_settings)

    # Optional probe-beam parameters for the SRS/SBS gain correction
    # (Turnbull et al., PRL 136, 135101 (2026)). Absent section ⇒ correction
    # disabled.
    pb = deck.get("probe_beam")
    if pb is not None:
        mode = pb.get("gain_mode", "exact")
        if mode not in ("exact", "small_gain", "off"):
            raise ValueError(
                f"[probe_beam] gain_mode must be 'exact'|'small_gain'|'off', got {mode!r}"
            )
        fp = float(pb.get("pol_p_fraction", 1.0))
        if not (0.0 <= fp <= 1.0):
            raise ValueError(
                f"[probe_beam] pol_p_fraction must be in [0, 1], got {fp}"
            )
        measurement_settings["probe_intensity"] = float(pb["intensity_W_per_cm2"])
        measurement_settings["probe_diameter"] = float(pb["diameter_um"]) * 1e-6
        measurement_settings["pol_p_fraction"] = fp
        measurement_settings["gain_mode"] = mode

    # ── 3. Build params_settings (array value → per-time expansion) ──────────
    params_raw = deck.get("params", {})
    params_settings = {}

    _per_time_re = _re.compile(r"^(.+)_(\d+)$")

    # Pass 1: prefix-level entries; expand array-valued 'value' into per-time keys
    for key, kw in params_raw.items():
        if _per_time_re.match(key):
            continue  # handled in pass 2
        val = kw.get("value")
        if isinstance(val, str):
            arr = _load_array(val, base_dir)
            src_time_key = "source_time_axis"
            _strip = {"value", src_time_key, "rel_min", "rel_max"}
            other = {k: v for k, v in kw.items() if k not in _strip}
            rel_min = kw.get("rel_min")
            rel_max = kw.get("rel_max")
            if src_time_key in kw:
                if fit_time_axis is None:
                    raise ValueError(
                        f"[params.{key}] specifies source_time_axis but [data] has no "
                        "'time_axis' field. Add 'time_axis = \"file.h5:time\"' to [data]."
                    )
                src_time = _load_array(kw[src_time_key], base_dir)
                arr = _np.interp(fit_time_axis, src_time, arr)
            elif arr.ndim != 1 or len(arr) != Nt:
                raise ValueError(
                    f"[params.{key}] value array has shape {arr.shape}; "
                    f"expected ({Nt},) to match Pkl_data time dimension."
                )
            for t in range(Nt):
                v = float(arr[t])
                entry = {"value": v}
                for ok, ov in other.items():
                    if hasattr(ov, "__len__") and not isinstance(ov, str):
                        entry[ok] = ov[t]
                    else:
                        entry[ok] = ov
                if rel_min is not None:
                    entry["min"] = v * (1 + rel_min)
                if rel_max is not None:
                    entry["max"] = v * (1 + rel_max)
                params_settings[f"{key}_{t}"] = entry
        else:
            ps = dict(kw)
            v = ps.get("value")
            if isinstance(v, (int, float)):
                if "rel_min" in ps:
                    ps["min"] = float(v) * (1 + ps.pop("rel_min"))
                if "rel_max" in ps:
                    ps["max"] = float(v) * (1 + ps.pop("rel_max"))
            params_settings[key] = ps

    # Pass 2: explicit per-time overrides win over expansion (most specific)
    for key, kw in params_raw.items():
        if _per_time_re.match(key):
            params_settings[key] = dict(kw)

    # Super-Gaussian exponents pe / pi are sampled through the tabulated
    # _Zprime, which covers p ∈ [2.0, 5.0]. Below 2.0 the forward model
    # silently extrapolates and produces NaN. Refuse bounds that allow that,
    # so a user typo doesn't quietly poison a long fit. The check only
    # applies to species actually using the analytic 'super_gaussian' model —
    # a custom model whose shape param happens to be named "p" sets its own
    # valid range.
    Nelectrons = measurement_settings["Nelectrons"]
    Nions = len(measurement_settings["ion_z"])
    _sg_e = _super_gaussian_species(e_specs, Nelectrons)
    _sg_i = _super_gaussian_species(i_specs, Nions)
    _p_key_re = _re.compile(r"^p([ei])(\d+)?(?:_\d+)?$")
    for key, kw in params_settings.items():
        m = _p_key_re.match(key)
        if not m or "min" not in kw:
            continue
        kind, sp_idx = m.group(1), m.group(2)
        sg_set = _sg_e if kind == "e" else _sg_i
        applies = bool(sg_set) if sp_idx is None else int(sp_idx) in sg_set
        if applies and float(kw["min"]) < 2.0:
            raise ValueError(
                f"[params.{key}] min = {kw['min']!r} is below 2.0. The "
                f"super-Gaussian exponent table covers p ∈ [2.0, 5.0]; "
                f"values below 2.0 cause silent NaN in the forward model."
            )

    params_settings = params_settings or None

    # ── 4. Build penalty_settings (load profile_axis arrays) ────────────────
    penalty_raw = deck.get("penalty", {})
    penalty_settings = {}
    for prefix, psettings in penalty_raw.items():
        ps = dict(psettings)
        if "profile_axis" in ps:
            ps["profile_axis"] = _load_array(ps["profile_axis"], base_dir)
        penalty_settings[prefix] = ps
    penalty_settings = penalty_settings or None

    # ── 5. Constraints (string expressions evaluated against jnp arrays) ────
    constraints_raw = deck.get("constraints", {})
    constraints_settings = {
        str(k): str(v) for k, v in constraints_raw.items()
    } if constraints_raw else None

    # ── 6. Fit settings, extra_params, output path ───────────────────────────
    extra_params = deck.get("extra_params", None)
    if extra_params:
        for entry in extra_params:
            # `expr` was an lmfit-backend-only constraint field; drop silently
            # so old decks load cleanly. Express constraints via [constraints].
            entry.pop("expr", None)
            for key, val in list(entry.items()):
                if isinstance(val, str) and key != "name":
                    entry[key] = _load_array(val, base_dir)

    fit_settings = dict(deck.get("fit", {}))
    # Legacy lmfit-backend keys — silently dropped so old decks load cleanly.
    fit_settings.pop("backend", None)
    fit_settings.pop("method", None)

    output_raw = deck.get("output", {})
    out_rel = output_raw.get("path", None)
    if out_rel is None:
        stem = deck.get("_deck_stem", "result")
        output_path = base_dir / f"{stem}_result.h5"
    else:
        output_path = base_dir / out_rel

    # ── [sampling] section ─────────────────────────────────────────────────
    # Optional posterior-sampling configuration. ``enabled = false`` (default)
    # means the CLI/run_fit_grad path skips the sampler entirely. The
    # ``--sample`` CLI flag can also force sampling even when this section
    # is missing.
    samp_raw = deck.get("sampling", None)
    sampling_settings = dict(samp_raw) if samp_raw is not None else {}
    sampling_settings.setdefault("enabled", False)

    # ── [l_curve] section ──────────────────────────────────────────────────
    # Optional L-curve sweep — replaces the single MAP fit when enabled.
    # ``lambda_scale`` is either an explicit list or, if absent, expanded
    # from ``lambda_scale_log = {min = -3, max = 2, n = 21}``.
    lc_raw = deck.get("l_curve", None)
    l_curve_settings = dict(lc_raw) if lc_raw is not None else {}
    l_curve_settings.setdefault("enabled", False)
    # Resolve lambda_scale whenever it's specified — independent of `enabled` —
    # so the `--l-curve` CLI flag works even when the deck didn't set
    # enabled = true. Only error about a missing lambda_scale when the section is
    # actually switched on (so a disabled/empty [l_curve] stays harmless).
    if "lambda_scale" in l_curve_settings:
        l_curve_settings["lambda_scale"] = _np.asarray(
            l_curve_settings["lambda_scale"], dtype=float,
        )
    elif "lambda_scale_log" in l_curve_settings:
        spec = l_curve_settings.pop("lambda_scale_log")
        try:
            lo = float(spec["min"]); hi = float(spec["max"])
            n  = int(spec["n"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "[l_curve].lambda_scale_log must be a table with keys "
                "{min, max, n}, e.g. {min = -3, max = 2, n = 21}."
            ) from exc
        l_curve_settings["lambda_scale"] = _np.logspace(lo, hi, n)
    elif l_curve_settings.get("enabled"):
        raise ValueError(
            "[l_curve] requires either 'lambda_scale' (list) or "
            "'lambda_scale_log' ({min, max, n})."
        )

    return (
        Pkl_data, Pkl_var,
        measurement_settings,
        penalty_settings,
        params_settings,
        fit_settings,
        extra_params,
        constraints_settings,
        output_path,
        sampling_settings,
        l_curve_settings,
    )


def save_fit_results(output_path, result, best_fit, deck_text=None,
                     time_axis=None, sampling_result=None,
                     save_cross_corr=True, save_samples=True,
                     l_curve_result=None):
    """Save fit results to an HDF5 file.

    Datasets written:
    - ``/best_fit``         : (Nk, Nt) forward model at best-fit params
    - ``/params/<prefix>``  : (Nt,) array per parameter prefix
    - ``/time``             : (Nt,) time array (if provided)

    When ``l_curve_result`` is provided (from
    :func:`ThomsonScattering.l_curve.compute_L_curve`), a
    ``/l_curve`` group is added with the full sweep:
    - ``/l_curve/lambda_scale``       — (N,)
    - ``/l_curve/residual_norm``      — (N,)
    - ``/l_curve/penalty_norm``       — (N,)
    - ``/l_curve/curvature``          — (N,)
    - ``/l_curve/loss``               — (N,)
    - ``/l_curve/best_fits``          — (N, Nk, Nt)
    - ``/l_curve/params/<prefix>``    — (N, Nt)
    - ``/l_curve/unreg/best_fit``     — (Nk, Nt)   when warm-started
    - ``/l_curve/unreg/params/<prefix>`` — (Nt,)   when warm-started
    plus attrs ``optimal_index``, ``optimal_lambda_scale``, ``warm_start``.

    When ``sampling_result`` is provided, posterior outputs are written into
    the same file:
    - ``/summary/mean|std|p16|p50|p84/<prefix>``   — (Nt,)
    - ``/summary/correlations/<prefix>``           — (Nt, Nt) intra-prefix
    - ``/summary/rhat|ess/<prefix>``               — (Nt,)
    - ``/summary/cross_correlations``              — (P·Nt, P·Nt), optional
    - ``/samples/<prefix>``                        — (n_chains, n_samples, Nt)
    - ``/u_samples``                               — (n_chains, n_samples, D)
    - ``/log_probs``                               — (n_chains, n_samples)
    - ``/step_size_history``                       — (burn_in,)
    - ``/varying_keys``                            — (D,) string
    - ``/u_chain_init``                            — (D,)

    Set ``save_samples=False`` to keep only the ``/summary/`` group (much
    smaller file; raw chains discarded). ``save_cross_corr=False`` further
    drops the full cross-correlation matrix.

    File-level attributes:
    - ``loss``, ``success``, ``nit``
    - ``deck_toml`` (if provided)
    """
    import h5py

    output_path = _pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as fh:
        fh.create_dataset("best_fit", data=_np.asarray(best_fit))
        if time_axis is not None:
            fh.create_dataset("time", data=_np.asarray(time_axis))
        params_grp = fh.create_group("params")
        for prefix, arr in result.params_dict.items():
            params_grp.create_dataset(prefix, data=_np.asarray(arr))
        fh.attrs["loss"]    = float(result.fun)
        fh.attrs["nit"]     = int(result.nit)
        fh.attrs["success"] = bool(result.success)
        if deck_text is not None:
            fh.attrs["deck_toml"] = deck_text

        if sampling_result is not None:
            _write_sampling_summary(fh, sampling_result,
                                    save_cross_corr=save_cross_corr,
                                    save_samples=save_samples)

        if l_curve_result is not None:
            _write_l_curve(fh, l_curve_result)


def _write_l_curve(fh, lc):
    """Write the ``/l_curve`` group from a `compute_L_curve` result."""
    g = fh.create_group("l_curve")
    g.create_dataset("lambda_scale",  data=_np.asarray(lc.lambda_scale))
    g.create_dataset("residual_norm", data=_np.asarray(lc.residual_norm))
    g.create_dataset("penalty_norm",  data=_np.asarray(lc.penalty_norm))
    g.create_dataset("curvature",     data=_np.asarray(lc.curvature))
    g.create_dataset("loss",          data=_np.asarray(lc.loss))
    g.create_dataset("best_fits",     data=_np.asarray(lc.best_fits))
    params_g = g.create_group("params")
    for prefix, arr in lc.params.items():
        params_g.create_dataset(prefix, data=_np.asarray(arr))
    g.attrs["optimal_index"]        = int(lc.optimal_index)
    g.attrs["optimal_lambda_scale"] = float(lc.lambda_scale[lc.optimal_index])
    g.attrs["warm_start"]           = lc.unreg_result is not None

    if lc.unreg_result is not None:
        u = g.create_group("unreg")
        u.create_dataset("best_fit", data=_np.asarray(lc.unreg_best_fit))
        up = u.create_group("params")
        for prefix, arr in lc.unreg_result.params_dict.items():
            up.create_dataset(prefix, data=_np.asarray(arr))
        u.attrs["loss"]    = float(lc.unreg_result.fun)
        u.attrs["nit"]     = int(lc.unreg_result.nit)
        u.attrs["success"] = bool(lc.unreg_result.success)


def _write_sampling_summary(fh, samp, *, save_cross_corr=True,
                            save_samples=True):
    """Write posterior summary + (optionally) raw samples into ``fh``.

    Writes the ``/summary/...`` group, ``/varying_keys``, and (when available)
    the MAP-Hessian ``/hessian_u`` + ``/hessian_ref`` and the ``/laplace``
    group always. When ``save_samples`` is True and the result carries chains
    (``method == "mcmc"``), additionally writes the full posterior chains
    under ``/samples/<prefix>`` plus ``/u_samples``, ``/log_probs``,
    ``/step_size_history``, and ``/u_chain_init``. Laplace-only results
    (``method == "laplace"``) have no chains; their summary std/percentiles
    come from the delta-method covariance.
    """
    import h5py
    has_samples = getattr(samp, "samples_phys", None) is not None
    summary = fh.create_group("summary")
    for stat in ("mean", "std", "p16", "p50", "p84"):
        g = summary.create_group(stat)
        for prefix, s in samp.summary.items():
            g.create_dataset(prefix, data=_np.asarray(s[stat]))
    corr_g = summary.create_group("correlations")
    for prefix, s in samp.summary.items():
        corr_g.create_dataset(prefix, data=_np.asarray(s["corr_intra"]))
    if getattr(samp, "rhat", None) is not None:
        rhat_g = summary.create_group("rhat")
        ess_g = summary.create_group("ess")
        for prefix in samp.summary:
            rhat_g.create_dataset(prefix, data=_np.asarray(samp.rhat[prefix]))
            ess_g.create_dataset(prefix, data=_np.asarray(samp.ess[prefix]))
    if save_cross_corr:
        if has_samples:
            prefixes = list(samp.summary.keys())
            cols = []
            for p in prefixes:
                v = samp.samples_phys[p]
                cols.append(v.reshape(-1, v.shape[-1]))  # (nc*ns, Nt)
            flat = _np.concatenate(cols, axis=1)         # (nc*ns, sum Nt)
            with _np.errstate(divide="ignore", invalid="ignore"):
                cc = _np.corrcoef(flat.T)
            cc = _np.where(_np.isfinite(cc), cc, 0.0)
            _np.fill_diagonal(cc, 1.0)
            idx_labels = []
            for p in prefixes:
                v = samp.samples_phys[p]
                for t in range(v.shape[-1]):
                    idx_labels.append(f"{p}[t={t}]")
        elif getattr(samp, "cov_phys", None) is not None:
            # Laplace-only: derive the cross-correlation from Σ_phys.
            sig = _np.sqrt(_np.clip(_np.diag(samp.cov_phys), 0.0, None))
            with _np.errstate(divide="ignore", invalid="ignore"):
                cc = samp.cov_phys / _np.outer(sig, sig)
            cc = _np.where(_np.isfinite(cc), cc, 0.0)
            _np.fill_diagonal(cc, 1.0)
            idx_labels = list(samp.cov_phys_labels)
        else:
            cc = None
        if cc is not None:
            summary.create_dataset("cross_correlations", data=cc)
            # Index strings so a reader can decode the row/col order.
            summary.create_dataset(
                "cross_correlations_labels",
                data=_np.array(idx_labels, dtype=h5py.string_dtype()),
            )

    summary.attrs["method"]      = str(getattr(samp, "method", "mcmc"))
    summary.attrs["kernel"]      = str(getattr(samp, "kernel", "sgld"))
    summary.attrs["temperature"] = float(samp.temperature)
    summary.attrs["wall_time_s"] = float(samp.wall_time)
    if has_samples:
        summary.attrs["n_chains"]       = int(samp.n_chains)
        summary.attrs["n_samples"]      = int(samp.n_samples)
        summary.attrs["burn_in"]        = int(samp.burn_in)
        summary.attrs["thin"]           = int(samp.thin)
        summary.attrs["step_size_final"] = float(samp.step_size_final)
        summary.attrs["precond"]        = str(samp.precond)
        summary.attrs["max_rhat"]       = float(samp.max_rhat)
        summary.attrs["max_rhat_key"]   = str(samp.max_rhat_key)
        summary.attrs["min_ess"]        = float(samp.min_ess)
        summary.attrs["min_ess_key"]    = str(samp.min_ess_key)
        if getattr(samp, "kernel", "sgld") in ("hmc", "mala"):
            summary.attrs["n_leapfrog"]   = int(samp.n_leapfrog)
            summary.attrs["traj_length"]  = float(getattr(samp, "traj_length", 0.0))
            summary.attrs["avg_leapfrog"] = float(getattr(samp, "avg_leapfrog", 0.0))
            summary.attrs["accept_rate"]  = _np.asarray(samp.accept_rate)
            summary.attrs["n_divergent"]  = int(samp.n_divergent)

    # Always save the param ordering (it labels both the Hessian axes and the
    # u-space arrays) and the full Hessian of the log-posterior at the MAP.
    # The Hessian is a single (D, D) matrix; -inv(hessian_u) is the Laplace
    # covariance in u-space. Its row/col order matches /varying_keys.
    fh.create_dataset(
        "varying_keys",
        data=_np.array(samp.varying_keys, dtype=h5py.string_dtype()),
    )
    if getattr(samp, "hessian_u", None) is not None:
        fh.create_dataset("hessian_u",   data=_np.asarray(samp.hessian_u))
        fh.create_dataset("hessian_ref", data=_np.asarray(samp.hessian_ref))

    # Laplace physical-parameter covariance at the MAP: the full (singular) P×P
    # matrix plus per-prefix 1σ error bars (sqrt of its diagonal), so they sit
    # alongside summary/std. Row/col order of cov_phys matches cov_phys_labels.
    # Non-identified (flat) directions are exported as relative physical
    # loadings so a reader can see which parameter combinations the data
    # cannot pin down.
    if getattr(samp, "cov_phys", None) is not None:
        lap = fh.create_group("laplace")
        lap.create_dataset("cov_phys", data=_np.asarray(samp.cov_phys))
        lap.create_dataset(
            "cov_phys_labels",
            data=_np.array(samp.cov_phys_labels, dtype=h5py.string_dtype()),
        )
        sig_g = lap.create_group("sigma")
        for prefix, arr in samp.laplace_sigma.items():
            sig_g.create_dataset(prefix, data=_np.asarray(arr))
        lap.attrs["n_nonidentified"] = int(getattr(samp, "n_nonidentified", 0))
        loadings = getattr(samp, "nonid_loadings", None)
        if loadings is not None and len(loadings):
            lap.create_dataset("nonidentified_loadings",
                               data=_np.asarray(loadings))
            lap.create_dataset(
                "nonidentified_descriptions",
                data=_np.array(getattr(samp, "nonid_descriptions", []),
                               dtype=h5py.string_dtype()),
            )

    if save_samples and has_samples:
        samples_grp = fh.create_group("samples")
        for prefix, arr in samp.samples_phys.items():
            samples_grp.create_dataset(prefix, data=_np.asarray(arr))
        fh.create_dataset("u_samples",         data=_np.asarray(samp.u_samples))
        fh.create_dataset("log_probs",         data=_np.asarray(samp.log_probs))
        fh.create_dataset("step_size_history", data=_np.asarray(samp.step_size_history))
        fh.create_dataset("u_chain_init",      data=_np.asarray(samp.u_chain_init))
