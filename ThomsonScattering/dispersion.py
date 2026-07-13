"""Plasma dispersion functions: analytic super-Gaussian path + general quadrature.

Two routes to the (derivative of the) plasma dispersion function:

1. ``_Zprime(zeta, p)`` — the super-Gaussian fast path inherited from the
   original package: real part interpolated from a precomputed table
   (zeta in [0, 10], p in [2, 5]) with a Laurent far-field expansion, imaginary
   part analytic in gamma functions.

   Normalization note (validated numerically against the Faddeeva-function
   Maxwellian Z'): with the textbook result
   Z'_std(zeta) = P∫ g'(x)/(x - zeta) dx + i*pi*g'(zeta), the package
   convention is

       2 * _Zprime(zeta, p) = -Re(Z'_std) + i*Im(Z'_std)
                            =  P∫ g'(x)/(zeta - x) dx + i*pi*g'(zeta)

   where g is the normalized 1D reduced distribution on x = v / vth,
   vth = sqrt(2*T/m). (This is Sheffield's sign convention for chi: at
   zeta=0 it gives chi = +1/(k*lambda_D)^2.)

2. ``hilbert_disp(...)`` — the general path for arbitrary distributions.
   Returns the same convention as ``2 * _Zprime``:

       Zgen(zeta) = P∫ g'(x)/(zeta - x) dx  +  i*pi*g'(zeta)

   via singularity-subtraction quadrature on a fixed normalized grid:

       Re = -[ sum_x w(x) * (g'(x) - g'(zeta))/(x - zeta)
               + g'(zeta) * ln|(X - zeta)/(X + zeta)| ]
       Im = pi * g'(zeta)

   The subtracted integrand has a removable singularity at x = zeta (limit
   g''(zeta)); the analytic log term carries the tail of the 1/(x - zeta)
   kernel exactly, so the method has no FFT-style edge artifacts and remains
   accurate for phase velocities far outside the grid (where it reproduces
   the 1/zeta^2 cold-plasma asymptote through moment cancellation).

   Susceptibility: chi = wp^2 / (vth*k)^2 * Zgen(zeta)   (no leading 2).
"""
import os
import jax.numpy as jnp
import h5py
import interpax
from jax.scipy.special import gamma

# ── analytic super-Gaussian path (tabulated) ─────────────────────────────────

#Interpolation grid
_p = jnp.linspace(2, 5, 2000)
_zeta = jnp.linspace(0, 10, 2000)

#Tabulated values
_h5_path = os.path.join(os.path.dirname(__file__), 'dispersion_tables.h5')
with h5py.File(_h5_path, 'r') as hf:
    _Zprime_real = jnp.array(hf["Zprime_real"])

#Create the interpolator function for the real part. Imaginary part is purely
#analytic in terms of gamma functions; see _Zprime below.
_Zprime_real_interp = interpax.Interpolator2D(_zeta, _p, _Zprime_real)

#Derivative of the plasma dispersion function as a function of the phase velocity zeta (normalized by vth)
# and the supergaussian order p

def _Zprime(zeta, p, order = 8):
    # p stays at its natural (broadcastable) shape so p-only quantities
    # are not redundantly evaluated along axes where p is constant.

    g3 = gamma(3 / p)
    g5 = gamma(5 / p)
    A_inner = (1 / 3) * g5 / g3
    A = jnp.sqrt(A_inner)
    C = p / (2 * g3) * A_inner ** 1.5

    # Imaginary part can be written purely in terms of gamma functions:
    # Bryan note: my PACM report is *wrong*; the expression there for the imaginary part
    # is missing a factor of zeta (and maybe also sqrt2)!
    # Interestingly enough Fig. 5 in that is correct though...

    Az = A * zeta * jnp.sqrt(2)
    abs_zeta = jnp.abs(zeta)

    Zprime_imag = -jnp.sqrt(2) * zeta * jnp.pi * C * jnp.exp(-jnp.power(jnp.abs(Az), p))

    #Re(Zprime) is tabulated at small values of zeta. Interpax requires same-shape
    #ravelled inputs, so broadcast p only at this site.
    out_shape = jnp.broadcast_shapes(zeta.shape, p.shape)
    abs_zeta_b = jnp.broadcast_to(abs_zeta, out_shape)
    p_b = jnp.broadcast_to(p, out_shape)
    Zprime_real_near = _Zprime_real_interp(abs_zeta_b.ravel(), p_b.ravel()).reshape(out_shape)

    #for large zeta a Laurent expansion is used
    #the form of the Laurent expansion is apparently (from my year-old notes):
    #(2C/A) * sum_n [1/(A*zeta)**(2n+2) * (1/p) * gamma((2n+3)/p)]

    n = jnp.arange(order // 2)

    # Guard the far-branch inputs against zeta=0 so the VJP stays finite.
    # jnp.where evaluates the gradient of both branches regardless of the
    # condition; 1/Az^(2n+1) -> inf at zeta=0, and 0*inf = NaN in the VJP.
    # Substituting a safe dummy (1.0) in the discarded branch avoids this
    # without changing the selected (near-branch) output.
    safe_Az   = jnp.where(abs_zeta > 10, Az,   jnp.ones_like(Az))
    safe_zeta = jnp.where(abs_zeta > 10, zeta, jnp.ones_like(zeta))

    #compute the terms of the Laurent expansion (n appended as trailing axis)
    Zprime_real_far_expansion = 1 / jnp.power(safe_Az[..., jnp.newaxis], 2 * n + 1) * gamma((2 * n + 3) / p[..., jnp.newaxis])

    Zprime_real_far = -2 * C / (A ** 2 * safe_zeta * jnp.sqrt(2) * p) * jnp.sum(Zprime_real_far_expansion, axis = -1)

    # Use jnp.where (select) instead of mask-and-add — avoids NaN propagation
    # from the far branch (which is inf at zeta=0) when masked out.
    Zprime_real = jnp.where(abs_zeta > 10, Zprime_real_far, Zprime_real_near)

    return Zprime_real + 1.j * Zprime_imag


# ── general quadrature path ──────────────────────────────────────────────────

def simpson_grid(x_max, n_points):
    """Uniform grid on [-x_max, x_max] with composite-Simpson weights.

    ``n_points`` must be odd (composite Simpson needs an even interval count);
    it is bumped up by one if even. Returns ``(x_grid, weights)``.
    """
    n_points = int(n_points)
    if n_points % 2 == 0:
        n_points += 1
    x_grid = jnp.linspace(-float(x_max), float(x_max), n_points)
    dx = 2.0 * float(x_max) / (n_points - 1)
    w = jnp.ones(n_points)
    w = w.at[1:-1:2].set(4.0)
    w = w.at[2:-1:2].set(2.0)
    return x_grid, w * (dx / 3.0)


def hilbert_disp(zeta, gp_zeta, gpp_zeta, gp_grid, x_grid, weights):
    """Generalized dispersion derivative Zgen(zeta) for an arbitrary g.

    Parameters
    ----------
    zeta : (...,) real
        Normalized phase velocities (v_phi / vth, vth = sqrt(2T/m)).
    gp_zeta, gpp_zeta : same shape as zeta
        g'(zeta) and g''(zeta) (g'' only used for the removable-singularity
        substitution when a quadrature node lands within dx/2 of zeta).
    gp_grid : (Nx,)
        g' evaluated on ``x_grid``.
    x_grid, weights : (Nx,)
        Quadrature grid and weights from :func:`simpson_grid`.

    Returns
    -------
    complex array, same shape as zeta:
        P∫ g'(x)/(zeta - x) dx + i*pi*g'(zeta)
    (the package's Sheffield-sign convention, matching ``2 * _Zprime``)

    Notes
    -----
    The divided difference (g'(x) - g'(zeta))/(x - zeta) suffers float
    cancellation when |x - zeta| is tiny (error ~ eps_machine/|x-zeta|, and
    ~ eps_machine/|x-zeta|^2 in the VJP), so any node within dx/2 of zeta is
    replaced by the analytic limit g''(zeta). We deliberately do NOT add a
    (x - zeta)*g'''(zeta)/2 correction: g''' can legitimately diverge at
    zeta = 0 for distributions with limited smoothness there (e.g. projected
    super-Gaussians with 2 < p < 3), which would inject 0*inf NaNs into the
    VJP. The resulting local error is O(dx^2 * |g'''|) — increase the grid
    resolution rather than the substitution order if more accuracy is needed.
    """
    X = x_grid[-1]
    dx = x_grid[1] - x_grid[0]
    eps = dx / 2.0

    # GEMM-shaped restructure of sum_x w*(g'(x) - g'(zeta))/(x - zeta):
    # build the masked reciprocal kernel once, contract both required sums
    # (against w*g' and against w) in a single matmul, then add the
    # near-node g''(zeta) substitution via the masked weight sum. Identical
    # arithmetic to the naive elementwise form but ~25% faster under AD
    # (fewer large intermediates; the contraction hits the dot engine).
    denom = x_grid - zeta[..., jnp.newaxis]            # (..., Nx)
    near = jnp.abs(denom) < eps
    # Dummy 1.0 in the masked branch keeps 1/denom (and its VJP) finite.
    inv_denom = jnp.where(near, 0.0, 1.0 / jnp.where(near, 1.0, denom))
    rhs = jnp.stack([weights * gp_grid, weights], axis=1)   # (Nx, 2)
    sums = inv_denom @ rhs                                  # (..., 2)
    w_near = near @ weights
    integral = (sums[..., 0] - gp_zeta * sums[..., 1]
                + gpp_zeta * w_near)

    # Analytic tail term g'(zeta) * ln|(X - zeta)/(X + zeta)|. Clamp the log
    # arguments away from 0 so zeta = ±X (where g' should be negligible
    # anyway) doesn't produce -inf and 0 * inf = NaN in the VJP.
    tiny = 1e-300
    log_term = gp_zeta * (
        jnp.log(jnp.maximum(jnp.abs(X - zeta), tiny))
        - jnp.log(jnp.maximum(jnp.abs(X + zeta), tiny))
    )

    # The real (Hilbert) part enters with the package's Sheffield sign
    # convention: Re Zgen = P∫ g'(x)/(zeta - x) dx = -(integral + log_term).
    return -(integral + log_term) + 1j * jnp.pi * gp_zeta
