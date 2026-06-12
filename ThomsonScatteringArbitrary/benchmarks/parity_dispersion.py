"""Early parity checks: general quadrature path vs the tabulated analytic path.

1. disp identity:    GeneralDistribution(projected super-Gaussian).disp(zeta)
                     vs 2*_Zprime(zeta, p)        for p in {2, 2.5, 3, 4, 5}
2. reduced identity: Maxwellian closed form vs the incomplete-gamma bracket
3. forward parity:   new scattered_power_wavelength (super_gaussian models)
                     vs old ThomsonScattering.scattered_power_wavelength,
                     on EPW-like and IAW-like parameter sets
4. gradient sanity:  finite gradients of the general path w.r.t. T and kappa

Run from the repo root:  python -m ThomsonScatteringArbitrary.benchmarks.parity_dispersion
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ThomsonScatteringArbitrary.dispersion import _Zprime
from ThomsonScatteringArbitrary.distributions import (
    Maxwellian, SuperGaussian, resolve_distribution, _supergauss_reduced,
)

FAIL = []


def check(name, err, tol):
    status = "PASS" if err < tol else "FAIL"
    if err >= tol:
        FAIL.append(name)
    print(f"  [{status}] {name}: max rel err = {err:.3e} (tol {tol:.0e})")


def rel_err(a, b, floor=None):
    a = np.asarray(a); b = np.asarray(b)
    scale = np.abs(b)
    if floor is None:
        floor = np.max(scale) * 1e-12 + 1e-300
    return float(np.max(np.abs(a - b) / np.maximum(scale, floor)))


# ── 1. disp identity ─────────────────────────────────────────────────────────
print("\n1. General quadrature disp vs 2*_Zprime (tabulated)")
zeta = jnp.linspace(-9.5, 9.5, 381).reshape(1, -1)  # (Nt=1, Nk)
for p in [2.0, 2.5, 3.0, 4.0, 5.0]:
    sg_num = resolve_distribution({"model": "super_gaussian_numeric",
                                   "x_max": 12.0, "n_points": 4001})
    shape = (jnp.array([p]),)
    got = np.asarray(sg_num.disp(zeta, shape))[0]
    want = np.asarray(2.0 * _Zprime(zeta, jnp.full((1, 1), p)))[0]
    # compare real and imaginary on the scale of |Zgen|
    scale = np.maximum(np.abs(want), 1e-2)
    err = float(np.max(np.abs(got - want) / scale))
    check(f"disp p={p}", err, 5e-4)

# far-field: |zeta| > 10 (Laurent branch of the table)
zeta_far = jnp.array([[12.0, 20.0, 50.0, 200.0, -15.0]])
sg_num = resolve_distribution({"model": "super_gaussian_numeric",
                               "x_max": 12.0, "n_points": 4001})
got = np.asarray(sg_num.disp(zeta_far, (jnp.array([2.0]),)))[0]
want = np.asarray(2.0 * _Zprime(zeta_far, jnp.full((1, 1), 2.0)))[0]
check("disp far-field p=2", rel_err(got.real, want.real), 1e-3)

# ── 2. reduced identity ──────────────────────────────────────────────────────
print("\n2. Maxwellian closed-form reduced vs incomplete-gamma bracket")
z = jnp.linspace(-6, 6, 1001)
got = np.asarray(Maxwellian().reduced(z, ()))
want = np.asarray(_supergauss_reduced(z, jnp.array(2.0)))
check("reduced maxwellian", rel_err(got, want, floor=1e-12), 1e-10)

# ── 3. forward parity vs old package ─────────────────────────────────────────
print("\n3. Forward-model parity: new (super_gaussian) vs old ThomsonScattering")
import ThomsonScattering.forward as old_fwd
import ThomsonScatteringArbitrary.forward as new_fwd

Nt = 5

# EPW-like setup (electron feature, ~526.5 nm probe, blue-shifted window)
epw = dict(
    wavelengths=jnp.linspace(450e-9, 520e-9, 200),
    probe_wavelength=526.5e-9,
    probe_vec=jnp.array([1.0, 0.0, 0.0]),
    scatter_vec=jnp.array([0.0, 1.0, 0.0]),
    ue_dir=jnp.array([1.0, 0.0, 0.0]),
    ui_dir=jnp.array([1.0, 0.0, 0.0]),
)
kB_over_e = 11604.518  # K per eV
n = jnp.full(Nt, 5e19) * 1e6                     # m^-3
Te = jnp.linspace(300.0, 500.0, Nt)[None, :] * kB_over_e   # K
Ti = jnp.full((1, Nt), 200.0) * kB_over_e
ue = jnp.zeros((1, Nt)); ui = jnp.zeros((1, Nt))
pe = jnp.full((1, Nt), 2.0); pi_arr = jnp.full((1, Nt), 2.0)
efract = jnp.ones((1, Nt)); ifract = jnp.ones((1, Nt))
ion_z = jnp.array([1.0]); ion_a = jnp.array([1.0])

for case_name, pe_val in [("EPW p=2", 2.0), ("EPW p=3.2", 3.2)]:
    pe_c = jnp.full((1, Nt), pe_val)
    old = old_fwd.scattered_power_wavelength(
        n, ue, ui, Te, Ti, pe_c, pi_arr, efract, ifract, ion_z, ion_a, **epw)
    new = new_fwd.scattered_power_wavelength(
        n, ue, ui, Te, Ti, efract, ifract, ion_z, ion_a, **epw,
        e_models=(SuperGaussian(),), i_models=(SuperGaussian(),),
        e_shapes=((pe_c[0],),), i_shapes=((pi_arr[0],),),
    )
    check(f"forward {case_name}", rel_err(new, old, floor=1e-8), 1e-12)

# IAW-like setup (two ion species D + C, red window around the probe)
iaw = dict(
    wavelengths=jnp.linspace(526.2e-9, 526.8e-9, 200),
    probe_wavelength=526.5e-9,
    probe_vec=jnp.array([1.0, 0.0, 0.0]),
    scatter_vec=jnp.array([0.0, 1.0, 0.0]),
    ue_dir=jnp.array([1.0, 0.0, 0.0]),
    ui_dir=jnp.array([1.0, 0.0, 0.0]),
)
Ti2 = jnp.stack([jnp.full(Nt, 300.0), jnp.full(Nt, 350.0)]) * kB_over_e
ui2 = jnp.stack([jnp.full(Nt, 1e5), jnp.full(Nt, 1e5)])
pi2 = jnp.full((2, Nt), 2.0)
ifract2 = jnp.stack([jnp.full(Nt, 0.5), jnp.full(Nt, 0.5)])
ion_z2 = jnp.array([1.0, 6.0]); ion_a2 = jnp.array([2.0, 12.0])

old = old_fwd.scattered_power_wavelength(
    n, ue, ui2, Te, Ti2, pe, pi2, efract, ifract2, ion_z2, ion_a2, **iaw)
new = new_fwd.scattered_power_wavelength(
    n, ue, ui2, Te, Ti2, efract, ifract2, ion_z2, ion_a2, **iaw,
    e_models=(SuperGaussian(),), i_models=(SuperGaussian(), SuperGaussian()),
    e_shapes=((pe[0],),), i_shapes=((pi2[0],), (pi2[1],)),
)
check("forward IAW D+C", rel_err(new, old, floor=1e-8), 1e-12)

# general-path super-Gaussian through the FULL forward model vs analytic
sgn = resolve_distribution({"model": "super_gaussian_numeric",
                            "x_max": 12.0, "n_points": 4001})
new_gen = new_fwd.scattered_power_wavelength(
    n, ue, ui2, Te, Ti2, efract, ifract2, ion_z2, ion_a2, **iaw,
    e_models=(sgn,), i_models=(sgn, sgn),
    e_shapes=((pe[0],),), i_shapes=((pi2[0],), (pi2[1],)),
)
check("forward IAW via general quadrature", rel_err(new_gen, old, floor=1e-8), 1e-3)

# ── 4. gradient sanity on the general path ───────────────────────────────────
print("\n4. Gradient finiteness through the general path (kappa)")
kappa_model = resolve_distribution("kappa")


def loss_fn(theta):
    Ti_k, kap = theta
    Ti_loc = jnp.full((1, Nt), Ti_k)
    new = new_fwd.scattered_power_wavelength(
        n, ue, Ti_loc * 0.0, Te, Ti_loc, efract, ifract, ion_z, ion_a, **iaw,
        e_models=(SuperGaussian(),), i_models=(kappa_model,),
        e_shapes=((pe[0],),), i_shapes=((jnp.full(Nt, kap),),),
    )
    return jnp.sum(new ** 2)


g = jax.grad(loss_fn)(jnp.array([300.0 * kB_over_e, 4.0]))
ok = bool(np.all(np.isfinite(np.asarray(g))))
print(f"  [{'PASS' if ok else 'FAIL'}] grad finite: {np.asarray(g)}")
if not ok:
    FAIL.append("kappa gradient")

print()
if FAIL:
    print(f"FAILURES: {FAIL}")
    sys.exit(1)
print("All parity checks passed.")
