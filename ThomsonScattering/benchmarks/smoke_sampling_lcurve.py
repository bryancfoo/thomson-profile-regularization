"""Smoke test: posterior sampling + L-curve sweep on the kappa (general-path) config.

Small budgets — checks the ports run end-to-end, produce finite outputs, and
that model SPECS (plain dicts) pickle across the L-curve process pool.

    python -m ThomsonScattering.benchmarks.smoke_sampling_lcurve
"""
import sys

import h5py
import numpy as np


from ThomsonScattering.fitting import _build_grad_problem
from ThomsonScattering.sampling import run_mcmc_posterior
from ThomsonScattering.l_curve import compute_L_curve

def main():
    # Everything lives inside main() behind the __main__ guard: the
    # L-curve pool uses spawn, which re-imports this module in each
    # worker — top-level work would re-execute recursively.
    FAIL = []

    with h5py.File("examples/data/data_kappa.h5", "r") as f:
        P = f["Pkl_data"][()]; V = f["Pkl_var"][()]
        wl = f["wavelengths"][()]; t_axis = f["time"][()]

    meas = dict(
        Nelectrons=1, ion_z=np.array([1.0]), ion_a=np.array([1.0]),
        probe_wavelength=2.6325e-7,
        probe_vec=np.array([0., 0., 1.]), scatter_vec=np.array([0.8660254, 0., 0.5]),
        ue_dir=np.array([1., 0., 0.]), ui_dir=np.array([1., 0., 0.]),
        wavelengths=wl, normalization_type="max",
        e_models=["maxwellian"],
        i_models=[{"model": "kappa", "x_max": 25.0, "n_points": 1001}],
    )
    pars = {
        "n": {"value": 6e19, "vary": False},
        "Te0": {"value": 480., "min": 100., "max": 1500.},
        "Ti0": {"value": 260., "min": 50., "max": 1000.},
        "kappai0": {"value": 4.0, "min": 1.7, "max": 30.0},
        "ue0": {"value": 0.0, "vary": False}, "ui0": {"value": 0.0, "vary": False},
        "efract0": {"value": 1.0, "vary": False}, "ifract0": {"value": 1.0, "vary": False},
    }

    # ── sampling smoke ───────────────────────────────────────────────────────────
    print("1. HMC posterior sampling (general path, tiny budget)")
    problem = _build_grad_problem(P, V, meas, params_settings=pars, shard_time=False)
    samp = run_mcmc_posterior(
        problem, problem.u0, n_samples=40, n_chains=2, burn_in=40, thin=1,
        precond="diag_hessian", progress=False,
    )
    finite = all(np.all(np.isfinite(s["mean"])) and np.all(np.isfinite(s["std"]))
                 for s in samp.summary.values())
    kap_mean = samp.summary["kappai0"]["mean"]
    kap_std = samp.summary["kappai0"]["std"]
    print(f"   kappai0 posterior mean: {np.round(kap_mean, 2)}")
    print(f"   kappai0 posterior std:  {np.round(kap_std, 3)}")
    print(f"   [{'PASS' if finite else 'FAIL'}] all summary stats finite")
    if not finite:
        FAIL.append("sampling finite")

    # ── L-curve smoke (2 workers → model-spec pickling across spawn) ────────────
    print("2. L-curve sweep, 3 points, n_workers=2 (spec pickling test)")
    pen = {"Ti0": {"profile_axis": t_axis, "lambda_weights": [0.0, 0.5, 0.2],
                   "thresholds": [0.0, 0.0, 0.0], "relative": True}}
    lc = compute_L_curve(
        P, V, meas, penalty_settings=pen, lambda_scale=np.array([0.1, 1.0, 10.0]),
        params_settings=pars, fit_settings={"optimizer": "lbfgs", "max_iter": 60},
        warm_start=False, progress=False, n_workers=2,
    )
    ok = (np.all(np.isfinite(lc.residual_norm)) and np.all(np.isfinite(lc.penalty_norm))
          and 0 <= lc.optimal_index < 3)
    print(f"   residual_norm: {np.round(lc.residual_norm, 4)}")
    print(f"   penalty_norm:  {np.round(lc.penalty_norm, 4)}")
    print(f"   optimal_index: {lc.optimal_index}")
    print(f"   [{'PASS' if ok else 'FAIL'}] L-curve finite + valid corner")
    if not ok:
        FAIL.append("l_curve")

    print()
    if FAIL:
        print(f"FAILURES: {FAIL}")
        sys.exit(1)
    print("Sampling + L-curve smoke tests passed.")


if __name__ == "__main__":
    main()
