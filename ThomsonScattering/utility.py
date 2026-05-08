import pathlib as _pathlib
import re as _re

import numpy as _np
import jax.numpy as jnp


#Some functions for moving data around
#Or reshaping arrays into the shape needed for the forward model
#Forward model wants the shape [Nions, Nt, Nk]

def reshape_moments(Q, Nions, Nt):
    #If the input is scalar:
    if jnp.ndim(Q)==0:
        return jnp.ones((Nions, Nt))[:, :, jnp.newaxis] * Q

    # Check to see if it's only 1D, eg one ion species
    if jnp.ndim(Q) == 1:
        # If it is, reshape to (, Nt, )
        return Q[jnp.newaxis, :, jnp.newaxis]
    elif jnp.ndim(Q) == 2:
        #This is if it's in the shape (Nions, Nt) already
        #Recast as (Nions, Nt,)
        return Q[:, :, jnp.newaxis]

# This matches the shapes of a, b 1D arrays to be [Na, Nb]
# It's here because interpax can't broadcast -.-

#Pulls params out of a output.params object
def extract_params_as_array(params, var, Nindices):
    return jnp.asarray([params[f"{var}_{i}"].value for i in range(Nindices)])

def extract_all_params_as_dict(params):
    """Extract all lmfit parameters into a dict of arrays, grouped by prefix.

    Parameters named like `Te0_3` are split into prefix `Te0` and index `3`.
    Each prefix maps to a 1-D array of its values ordered by index.

    Returns
    -------
    dict mapping prefix (str) -> jnp.array of values
    """
    from collections import defaultdict
    import re

    # Group key names by their prefix (everything before the last _<digits>)
    prefix_indices = defaultdict(list)
    for key in params:
        m = re.match(r"^(.+)_(\d+)$", key)
        if m:
            prefix_indices[m.group(1)].append(int(m.group(2)))

    result = {}
    for prefix, indices in prefix_indices.items():
        Nindices = max(indices) + 1
        result[prefix] = extract_params_as_array(params, prefix, Nindices)
    return result


#Adds a profile of params to an existing lmfit Parameters object
def add_params_from_array(params, var, value_array, settings):
    Nvalues = len(value_array)
    if isinstance(settings, dict):
        settings = [settings] * Nvalues
    for i in range(Nvalues):
        params.add(var + f"_{i}", value=value_array[i], **settings[i])
    #params are mutable (I think) so no return is needed?


# ─── Input deck helpers ─────────────────────────────────────────────────────

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

    # HDF5 with embedded dataset spec: "some/path.h5:dataset"
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


def _require(d, keys, section):
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(
            f"Missing required keys in {section}: {missing}. "
            "Check your input deck."
        )


def build_settings_from_deck(deck):
    """Convert a parsed deck dict (from load_deck) into the arguments for run_fit.

    Parameters
    ----------
    deck : dict
        Output of load_deck (includes private ``_base_dir`` and ``_deck_stem`` keys).

    Returns
    -------
    Pkl_data            : np.ndarray (Nk, Nt)
    Pkl_var             : np.ndarray (Nk, Nt)
    measurement_settings : dict
    penalty_settings    : dict or None
    params_settings     : dict or None
    fit_settings        : dict  (``backend`` key already removed)
    extra_params        : list of dict or None
    constraints_settings : dict[str, str] or None
        Mapping of parameter prefix → expression string from the deck's
        ``[constraints]`` table. Same string drives both backends.
    output_path         : pathlib.Path
    backend             : str  ``"lmfit"`` or ``"grad"``
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

    # Fields that must become numpy arrays (inline list or file ref)
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

    # Optional probe-beam parameters for the SRS/SBS gain correction
    # (Turnbull et al., PRL 136, 135101 (2026)). Absent section ⇒ correction
    # disabled, output identical to gain-free model.
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
                entry = {"value": v, **other}
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

    # ── 5. Constraints (string expressions; both backends compile their own) ─
    constraints_raw = deck.get("constraints", {})
    constraints_settings = {
        str(k): str(v) for k, v in constraints_raw.items()
    } if constraints_raw else None

    # ── 6. Fit settings, extra_params, output path ───────────────────────────
    extra_params = deck.get("extra_params", None)

    fit_raw = dict(deck.get("fit", {}))
    backend = fit_raw.pop("backend", "lmfit")
    if backend not in ("lmfit", "grad"):
        raise ValueError(
            f"[fit] backend must be 'lmfit' or 'grad', got {backend!r}"
        )
    fit_settings = fit_raw

    output_raw = deck.get("output", {})
    out_rel = output_raw.get("path", None)
    if out_rel is None:
        stem = deck.get("_deck_stem", "result")
        output_path = base_dir / f"{stem}_result.h5"
    else:
        output_path = base_dir / out_rel

    return (
        Pkl_data, Pkl_var,
        measurement_settings,
        penalty_settings,
        params_settings,
        fit_settings,
        extra_params,
        constraints_settings,
        output_path,
        backend,
    )


def save_fit_results(output_path, result, best_fit, backend, deck_text=None):
    """Save fit results to an HDF5 file.

    Datasets written:
    - ``/best_fit``         : (Nk, Nt) forward model at best-fit params
    - ``/params/<prefix>``  : (Nt,) array per parameter prefix
    File-level attributes:
    - ``loss``, ``success``, ``nfev`` (lmfit) or ``nit`` (grad), ``message``
    - ``deck_toml``         : raw deck text for provenance (if provided)

    Parameters
    ----------
    output_path : str or pathlib.Path
    result      : lmfit.MinimizerResult  (backend='lmfit')
                  or types.SimpleNamespace (backend='grad')
    best_fit    : array (Nk, Nt)
    backend     : ``"lmfit"`` or ``"grad"``
    deck_text   : str or None
    """
    import h5py
    from collections import defaultdict

    output_path = _pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as fh:
        fh.create_dataset("best_fit", data=_np.asarray(best_fit))
        params_grp = fh.create_group("params")

        if backend == "lmfit":
            param_dict = extract_all_params_as_dict(result.params)
            for name, arr in param_dict.items():
                params_grp.create_dataset(name, data=_np.asarray(arr))
            try:
                fh.attrs["loss"] = float(result.residual)
            except (TypeError, ValueError):
                fh.attrs["loss"] = float("nan")
            fh.attrs["nfev"]    = int(getattr(result, "nfev", 0))
            fh.attrs["success"] = bool(result.success)
            fh.attrs["message"] = str(getattr(result, "message", ""))
        else:  # grad
            # Group varying params by prefix into (Nt,) arrays
            prefix_vals = defaultdict(list)
            for name, val in zip(result.varying_keys, result.x):
                m = _re.match(r"^(.+)_(\d+)$", name)
                if m:
                    prefix_vals[m.group(1)].append((int(m.group(2)), float(val)))
                else:
                    prefix_vals[name].append((0, float(val)))
            for prefix, indexed in prefix_vals.items():
                indexed.sort()
                arr = _np.array([v for _, v in indexed])
                params_grp.create_dataset(prefix, data=arr)
            fh.attrs["loss"]    = float(result.fun)
            fh.attrs["nit"]     = int(result.nit)
            fh.attrs["success"] = bool(result.success)

        if deck_text is not None:
            fh.attrs["deck_toml"] = deck_text
