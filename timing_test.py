import numpy as np
import matplotlib.pyplot as plt
from jax import jit

from forward import _scattered_power_wavelength
from scipy.constants import c, k as kB, epsilon_0, e, m_p
import time



Nt = 51 #number of timesteps
t = 0*np.linspace(0, 3, Nt) #ns time
tau = 1.5 #time over which plasma parameters vary

#Plasma parameters
ne = 1e20 * np.exp(-t / tau)
ue = 0e6 * np.exp(-np.sqrt(t / tau))
ui = 0*np.exp(-np.sqrt(t / tau))#ue + 2e5 * np.exp(-t / tau) - 1e5
Te = 100 * np.exp(-t / tau)
Ti = 200 - np.sqrt(t) * 100
ifractC = 0.05 * t**2 #deuterons
ifractD = 1 - ifractC #carbon

nD = ne * ifractD
nC = ne * ifractC / 6

#plasmapy defines ifract differently...
ifractD_old = nD / (nD + nC)
ifractC_old = nC / (nD + nC)

def sph2cart(phi, theta):
    phi = phi / 180 * np.pi
    theta = theta / 180 * np.pi
    ans = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
    return ans / np.linalg.norm(ans)

probe_wavelength = 263.25 #4w on OMEGA
epw_lam = np.linspace(probe_wavelength - 30, probe_wavelength + 30, 1024)#
#epw_lam = np.linspace(235 - 20, 235 + 20, 1024) #Only measuring the blue peak
iaw_lam = np.linspace(probe_wavelength - 2, probe_wavelength + 2, 1024)


probe_vec = sph2cart(116.57, 18)
scatter_vec = -sph2cart(112.15, 162)
k_vec = scatter_vec - probe_vec
k_vec = k_vec / np.linalg.norm(k_vec)



def record_time(Ntimes):
    _jitted_scattered_power_wavelength = jit(_scattered_power_wavelength,
                                         static_argnames=('normalization_type', 'notch'))
    t0 = time.time()
    for i in range(Ntimes):
        
        _scattered_power_wavelength(
            n = ne*1e6,
            Te = np.array([Te]) * e / kB,
            Ti = np.array([Ti, Ti]) * e / kB,
            ue = np.array([ue]),
            ui = np.array([ui, ui]),
            pe = np.ones_like(ne) * 2,
            pi = np.ones_like(np.array([ui, ui])) * 2,
            efract = np.array([np.ones_like(ne)]),
            ifract = np.array([ifractD, ifractC]),
            ion_z = np.array([1, 6]),
            ion_a = np.array([2, 12]),
            wavelengths=epw_lam * 1e-9,
            probe_wavelength=probe_wavelength * 1e-9,
            probe_vec=probe_vec,
            scatter_vec=scatter_vec,
            ue_dir=k_vec,
            ui_dir=k_vec,
        )
    t1 = time.time()
    for i in range(Ntimes):
        _jitted_scattered_power_wavelength(
            n = ne*1e6,
            Te = np.array([Te]) * e / kB,
            Ti = np.array([Ti, Ti]) * e / kB,
            ue = np.array([ue]),
            ui = np.array([ui, ui]),
            pe = np.ones_like(ne) * 2,
            pi = np.ones_like(np.array([ui, ui])) * 2,
            efract = np.array([np.ones_like(ne)]),
            ifract = np.array([ifractD, ifractC]),
            ion_z = np.array([1, 6]),
            ion_a = np.array([2, 12]),
            wavelengths=epw_lam * 1e-9,
            probe_wavelength=probe_wavelength * 1e-9,
            probe_vec=probe_vec,
            scatter_vec=scatter_vec,
            ue_dir=k_vec,
            ui_dir=k_vec,
        )
    t2 = time.time()
    return t1 - t0, t2 - t1

Ntimes_array = np.logspace(0, 3, 10, dtype=int)
jtimes = np.zeros(len(Ntimes_array))
njtimes = np.zeros(len(Ntimes_array))
for i, Ntimes in enumerate(Ntimes_array):
    njtimes[i], jtimes[i] = record_time(Ntimes)
    print(Ntimes)

plt.semilogy(Ntimes_array, njtimes)
plt.plot(Ntimes_array, jtimes)
plt.show()