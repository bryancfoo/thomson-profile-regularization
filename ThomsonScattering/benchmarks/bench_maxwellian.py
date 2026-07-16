"""Maxwellian fit-regression benchmark for the analytic (tabulated-Z') path.

Runs the super-Gaussian/Maxwellian example decks end-to-end and checks the
final loss against recorded reference values, plus reports jit-compiled
forward+grad and full-fit timings.

The references were recorded at the commit that promoted the
arbitrary-distribution rewrite to `ThomsonScattering` (old-vs-new parity was
validated there first: objective and gradient to float64 round-off, full fits
converging to identical losses in identical iteration counts, parameters to
~1e-15 — see the promotion commit message). This benchmark guards that
behavior against future regressions without needing the deleted original
package.

Run from the repo root:
    python -m ThomsonScattering.benchmarks.bench_maxwellian
"""
import pathlib
import sys
import time

import numpy as np

import ThomsonScattering as TS
from ThomsonScattering.fitting import _build_grad_problem

REPO = pathlib.Path(__file__).resolve().parents[2]
FAIL = []

# Reference final losses recorded from the validated parity run (2026-07-02,
# promotion of the rewrite; identical for old and new packages). The
# iaw_constraints reference was re-recorded after the deck switched to
# irf_mode = "gaussian" (2026-07-13 port from main; the array-mode deck's
# reference was 2.4358528 at the same nit).
REFERENCE = {
    "examples/epw_basic/fit.toml":       {"loss": 0.99565696,  "nit": 120},
    "examples/iaw_constraints/fit.toml": {"loss": 3.682913352, "nit": 316},
}


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
    out = fn(*args)  # warmup / compile
    _block(out)
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
    ref = REFERENCE[deck_rel]
    print(f"\n=== {deck_rel} ===")

    deck = TS.load_deck(deck_path)
    (P, V, M, Pen, Par, Fit, Ext, Con, _, _, _) = TS.build_settings_from_deck(deck)

    prob = _build_grad_problem(P, V, M, penalty_settings=Pen, params_settings=Par,
                               constraints=Con, extra_params=Ext)

    # jit-compiled steady-state forward+grad timing
    t_eval = time_call(prob.val_and_grad_fn, prob.u0)
    print(f"  timing val_and_grad: {t_eval*1e3:7.2f} ms")

    # full fit + loss regression
    t0 = time.perf_counter()
    res, best_fit = TS.run_fit_grad(P, V, M, penalty_settings=Pen,
                                    params_settings=Par, constraints=Con,
                                    extra_params=Ext, fit_settings=dict(Fit))
    t_fit = time.perf_counter() - t0
    print(f"  timing full fit:     {t_fit:7.2f} s")
    print(f"  loss={res.fun:.8g} nit={res.nit} success={res.success} "
          f"(reference loss={ref['loss']:.8g} nit={ref['nit']})")
    check("final loss vs reference", max_rel(res.fun, ref["loss"]), 1e-5)
    if res.nit != ref["nit"]:
        # informational: iteration count can legitimately drift with
        # jax/optax versions even when the optimum is unchanged.
        print(f"  [note] nit {res.nit} != reference {ref['nit']} "
              "(loss check is the regression gate)")
    if not np.all(np.isfinite(np.asarray(best_fit))):
        FAIL.append("best_fit finite")
        print("  [FAIL] best-fit spectrum contains non-finite values")


def main():
    for deck_rel in REFERENCE:
        run_deck(deck_rel)
    print()
    if FAIL:
        print(f"FAILURES: {FAIL}")
        sys.exit(1)
    print("All Maxwellian fit-regression checks passed.")


if __name__ == "__main__":
    main()
