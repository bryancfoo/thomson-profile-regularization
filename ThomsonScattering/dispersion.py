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
    zeta_shape_in = zeta.shape

    #ravel the arrays or else interpax gets upset...
    zeta = zeta.ravel()
    p = p.ravel()
    #print(jnp.shape(zeta), jnp.shape(p))

    C = p / (2 * gamma(3 / p)) * (1 / 3 * gamma(5 / p) / gamma(3 / p)) ** (3 / 2)
    A = (1 / 3 * gamma(5 / p) / gamma(3 / p)) ** (1 / 2)

    # Imaginary part can be written purely in terms of gamma functions:
    # Bryan note: my PACM report is *wrong*; the expression there for the imaginary part
    # is missing a factor of zeta (and maybe also sqrt2)!
    # Interestingly enough Fig. 5 in that is correct though...

    Zprime_imag = jnp.sqrt(2) * zeta * jnp.pi * C * jnp.exp(-jnp.power(jnp.abs(A * zeta * jnp.sqrt(2)), p))

    #Re(Zprime) is tabulated at small values of zeta:
    Zprime_real_near = _Zprime_real_interp(jnp.abs(zeta), p)

    #for large zeta a Laurent expansion is used
    #the form of the Laurent expansion is apparently (from my year-old notes):
    #(2C/A) * sum_n [1/(A*zeta)**(2n+2) * (1/p) * gamma((2n+3)/p)]
    #How did I ever derive this???

    #Build the array of n to sum over along a dimension which is 1 more than the dimension of zeta
    #Should be compatible with making zeta, p the same size along different axes?
    #n = jnp.ones(jnp.concatenate([jnp.ones(zeta.ndim), [int(order // 2)]]))
    n = jnp.arange(order // 2)
    n = n[jnp.newaxis, :]

    #now do another reshaping...

    #compute the terms of the Laurent expansion
    Zprime_real_far_expansion = 1 / jnp.power((A * zeta * jnp.sqrt(2))[:, jnp.newaxis], 2 * n + 1) * gamma((2 * n + 3) / p[:, jnp.newaxis])

    Zprime_real_far = -2 * C / (A ** 2 * zeta * jnp.sqrt(2) * p) * jnp.sum(Zprime_real_far_expansion, axis = -1)

    #Mask out the values of zeta in the wrong regimes and add the two regimes
    Zprime_real = Zprime_real_far * (jnp.abs(zeta) > 10) + Zprime_real_near * (jnp.abs(zeta) <= 10)
    Zprime_flat = (Zprime_real + 1.j * Zprime_imag)
    Zprime_reshaped = jnp.reshape(Zprime_flat, zeta_shape_in)

    return Zprime_reshaped
