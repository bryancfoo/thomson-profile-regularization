import jax.numpy as jnp
from scipy.constants import c, k as kB, epsilon_0, e, m_p

#Contains helper functions called by the main forward model
#Various functions of the plasma parameters

def thermal_velocity(T, a, coef = 2):
    #note coef goes inside the sqrt
    return jnp.sqrt(coef * T * kB / (a * m_p))

def plasma_frequency(n, z, a):
    return jnp.sqrt(n * z**2 * e**2/ (m_p * epsilon_0) / a)
    #return jnp.sqrt(n * z**2 * e**2 / (a * m_p * epsilon_0))

def lam_Debye(ne, Te):
    return jnp.sqrt(epsilon_0 * Te / (ne * e**2))
