import os
import jax.numpy as jnp
import h5py
import interpax
from jax.scipy.special import gamma

#Calculates the plasma dispersion function chi

#First we load in the tabulated values of the derivative of the plasma dispersion function Z'(zeta):

#Interpolation grid
_p = jnp.linspace(2, 5, 2000)
_zeta = jnp.linspace(0, 10, 2000)

#Tabulated values
_h5_path = os.path.join(os.path.dirname(__file__), 'dispersion_tables.h5')
with h5py.File(_h5_path, 'r') as hf:
    _Zprime_real = jnp.array(hf["Zprime_real"])
    #_Zprime_imag = hf["Zprime_imag"]

#Create the interpolator functions
_Zprime_real_interp = interpax.Interpolator2D(_zeta, _p, _Zprime_real)
#_Zprime_imag_interp = interpax.Interpolator2D(_zeta, _p, _Zprime_imag)

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

    Zprime_imag = jnp.sqrt(2) * zeta * jnp.pi * C * jnp.exp(-jnp.power(jnp.abs(Az), p))

    #Re(Zprime) is tabulated at small values of zeta. Interpax requires same-shape
    #ravelled inputs, so broadcast p only at this site.
    out_shape = jnp.broadcast_shapes(zeta.shape, p.shape)
    abs_zeta_b = jnp.broadcast_to(abs_zeta, out_shape)
    p_b = jnp.broadcast_to(p, out_shape)
    Zprime_real_near = _Zprime_real_interp(abs_zeta_b.ravel(), p_b.ravel()).reshape(out_shape)

    #for large zeta a Laurent expansion is used
    #the form of the Laurent expansion is apparently (from my year-old notes):
    #(2C/A) * sum_n [1/(A*zeta)**(2n+2) * (1/p) * gamma((2n+3)/p)]
    #How did I ever derive this???

    n = jnp.arange(order // 2)

    #compute the terms of the Laurent expansion (n appended as trailing axis)
    Zprime_real_far_expansion = 1 / jnp.power(Az[..., jnp.newaxis], 2 * n + 1) * gamma((2 * n + 3) / p[..., jnp.newaxis])

    Zprime_real_far = -2 * C / (A ** 2 * zeta * jnp.sqrt(2) * p) * jnp.sum(Zprime_real_far_expansion, axis = -1)

    # Use jnp.where (select) instead of mask-and-add — avoids NaN propagation
    # from the far branch (which is inf at zeta=0) when masked out.
    Zprime_real = jnp.where(abs_zeta > 10, Zprime_real_far, Zprime_real_near)

    return Zprime_real + 1.j * Zprime_imag
