"""Run a Thomson scattering fit from a TOML input deck.

Usage
-----
    python fit_from_deck.py

The script prompts for the path to a ``.toml`` deck file, reads all
measurement geometry, penalty, parameter, and optimizer settings from it,
runs the fit, and writes results to an HDF5 file.

See ``example_deck.toml`` for the full deck schema with inline documentation.
"""

from pathlib import Path

from ThomsonScattering.utility import (
    load_deck,
    build_settings_from_deck,
    save_fit_results,
)
from ThomsonScattering.fitting import run_fit, run_fit_grad


def main():
    deck_path = Path(input("Path to input deck (.toml): ").strip()).expanduser().resolve()

    if not deck_path.exists():
        raise FileNotFoundError(f"Deck file not found: {deck_path}")

    deck_text = deck_path.read_text(encoding="utf-8")
    deck = load_deck(deck_path)

    (
        Pkl_data, Pkl_var,
        meas, pen, pars, fit_kw,
        extras, out_path, backend,
    ) = build_settings_from_deck(deck)

    print(f"\nRunning fit  backend={backend!r}  Nt={Pkl_data.shape[1]}  Nk={Pkl_data.shape[0]}")

    if backend == "lmfit":
        result, best_fit = run_fit(
            Pkl_data, Pkl_var, meas,
            penalty_settings=pen,
            params_settings=pars,
            fit_settings=fit_kw,
            extra_params=extras,
            progress=True,
        )
        loss    = float(result.residual) if not hasattr(result.residual, "__len__") else float("nan")
        neval   = getattr(result, "nfev", "?")
        success = result.success
        msg     = getattr(result, "message", "")
        print(f"\nloss={loss:.6g}  nfev={neval}  success={success}  message={msg!r}")
    else:
        if extras:
            import warnings
            warnings.warn(
                "extra_params are not supported with backend='grad' and will be ignored.",
                stacklevel=2,
            )
        result, best_fit = run_fit_grad(
            Pkl_data, Pkl_var, meas,
            penalty_settings=pen,
            params_settings=pars,
            fit_settings=fit_kw,
            progress=True,
        )
        print(f"\nloss={result.fun:.6g}  nit={result.nit}  success={result.success}")

    save_fit_results(out_path, result, best_fit, backend, deck_text=deck_text)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
