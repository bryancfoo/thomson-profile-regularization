"""Maxwellian fit-parity benchmark: ThomsonScatteringArbitrary vs ThomsonScattering.

Loads the original example decks (which use super-Gaussian/Maxwellian species
— the new package's default models) with BOTH packages and checks:

1. Objective parity:  V(u0) and dV/du(u0) agree to float64 round-off.
   (Strongest check — independent of optimizer trajectory chaos.)
2. Fit parity:        final loss, recovered parameter profiles, and best-fit
   spectra agree to fit tolerance after running the deck's optimizer.
3. Timing:            jit-compiled forward+grad eval and end-to-end fit
   wall-clock for both packages.

Run from the repo root:
    python -m ThomsonScatteringArbitrary.benchmarks.bench_maxwellian
"""
import pathlib
import sys
import time

import numpy as np

import ThomsonScattering as OLD
import ThomsonScatteringArbitrary as NEW
from ThomsonScattering.fitting import _build_grad_problem as old_problem
from ThomsonScatteringArbitrary.fitting import _build_grad_problem as new_problem

REPO = pathlib.Path(__file__).resolve().parents[2]
FAIL = []


def check(name, err, tol):
    status = "PASS" if err < tol else "FAIL"
    if err >= tol:
        FAIL.append(name)
    print(f"  [{status}] {name}: {err:.3e} (tol {tol:.0e})")


def max_rel(a, b, floor=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = np.maximum(np.abs(b), np.max(np.abs(b)) * 1e-9 + floor)
    return float(np.nanmax(np.abs(a - b) / scale))


def time_call(fn, *args, n=10):
    fn(*args)  # warmup / compile
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
    _block(out)
    return (time.perf_counter() - t0) / n


def _block(out):
    if isinstance(out, tuple):
        for o in out:
            _block(o)
    elif hasattr(out, "block_until_ready"):
        out.block_until_ready()


def run_deck(deck_rel):
    deck_path = REPO / deck_rel
    print(f"\n=== {deck_rel} ===")

    old_deck = OLD.load_deck(deck_path)
    (oP, oV, oM, oPen, oPar, oFit, oExt, oCon, _, _, _) = (
        OLD.build_settings_from_deck(old_deck))
    new_deck = NEW.load_deck(deck_path)
    (nP, nV, nM, nPen, nPar, nFit, nExt, nCon, _, _, _) = (
        NEW.build_settings_from_deck(new_deck))

    op = old_problem(oP, oV, oM, penalty_settings=oPen, params_settings=oPar,
                     constraints=oCon, extra_params=oExt)
    np_ = new_problem(nP, nV, nM, penalty_settings=nPen, params_settings=nPar,
                      constraints=nCon, extra_params=nExt)

    # 1. objective + gradient parity at the initial point
    assert op.varying_keys == np_.varying_keys, "varying-key sets differ!"
    assert np.allclose(op.u0, np_.u0), "initial unconstrained points differ!"
    ov, og = op.val_and_grad_fn(op.u0)
    nv, ng = np_.val_and_grad_fn(np_.u0)
    check("objective V(u0) parity", max_rel(float(nv), float(ov)), 1e-10)
    check("gradient dV(u0) parity", max_rel(ng, og), 1e-8)

    # 3a. forward+grad timing (jit-compiled steady state)
    t_old = time_call(op.val_and_grad_fn, op.u0)
    t_new = time_call(np_.val_and_grad_fn, np_.u0)
    print(f"  timing val_and_grad: old {t_old*1e3:7.2f} ms | "
          f"new {t_new*1e3:7.2f} ms | ratio {t_new/t_old:.2f}x")

    # 2. full fit parity
    t0 = time.perf_counter()
    o_res, o_fit = OLD.run_fit_grad(oP, oV, oM, penalty_settings=oPen,
                                    params_settings=oPar, constraints=oCon,
                                    extra_params=oExt, fit_settings=dict(oFit))
    t_fit_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    n_res, n_fit = NEW.run_fit_grad(nP, nV, nM, penalty_settings=nPen,
                                    params_settings=nPar, constraints=nCon,
                                    extra_params=nExt, fit_settings=dict(nFit))
    t_fit_new = time.perf_counter() - t0
    print(f"  timing full fit:     old {t_fit_old:7.2f} s  | "
          f"new {t_fit_new:7.2f} s  | ratio {t_fit_new/t_fit_old:.2f}x")
    print(f"  old: loss={o_res.fun:.8g} nit={o_res.nit} success={o_res.success}")
    print(f"  new: loss={n_res.fun:.8g} nit={n_res.nit} success={n_res.success}")

    check("final loss parity", max_rel(n_res.fun, o_res.fun), 1e-5)
    check("best-fit spectrum parity", max_rel(n_fit, o_fit), 1e-3)
    worst = 0.0
    worst_key = ""
    for key in o_res.params_dict:
        e = max_rel(n_res.params_dict[key], o_res.params_dict[key])
        if e > worst:
            worst, worst_key = e, key
    check(f"recovered params parity (worst: {worst_key})", worst, 1e-3)


def main():
    run_deck("examples/epw_basic/fit.toml")
    run_deck("examples/iaw_constraints/fit.toml")
    print()
    if FAIL:
        print(f"FAILURES: {FAIL}")
        sys.exit(1)
    print("All Maxwellian fit-parity checks passed.")


if __name__ == "__main__":
    main()
