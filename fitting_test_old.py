"""Simple smoke test for fitting._log_posterior

This script constructs a minimal lmfit.Parameters object with the
naming scheme expected by the fitter (e.g. ne0_0, Te0_0, Ti0_0, ...)
and attempts to call `_log_posterior`. It prints the returned value
or any exception encountered (useful when dependencies or tables are
missing).

Run with:

python fitting_test.py

"""

import numpy as np
from lmfit import Parameters
import jax.numpy as jnp

from fitting import _log_posterior, build_minimal_params
from utility import extract_params_as_array
from forward import _scattered_power_wavelength
from scipy.constants import c, k as kB, epsilon_0, e, m_p


if __name__ == "__main__":
    Nelectrons = 1
    Nions = 1
    Nt = 10
    Nk = 10

    params = build_minimal_params(Nelectrons, Nions, Nt)

    measurement_settings = {
        "Nelectrons": Nelectrons,
        "ion_z": np.array([1.0]),
        "ion_a": np.array([1.0]),
        "wavelengths": np.linspace(200e-9, 300e-9, Nk),
        "probe_wavelength": 263e-9,
        "probe_vec": np.array([1.0, 0.0, 0.0]),
        "scatter_vec": np.array([0.0, 1.0, 0.0]),
        "ue_dir": np.array([1.0, 0.0, 0.0]),
        "ui_dir": np.array([1.0, 0.0, 0.0]),
        # optional entries left out if not needed
    }

    n = extract_params_as_array(params, "n", Nt)

    Te = jnp.zeros((Nelectrons, Nt))
    ue = jnp.zeros((Nelectrons, Nt))
    pe = jnp.zeros((Nelectrons, Nt))
    efract = jnp.zeros((Nelectrons, Nt))
    for i in range(Nelectrons):
        Te = Te.at[i, :].set(extract_params_as_array(params, f"Te{i}", Nt))
        ue = ue.at[i, :].set(extract_params_as_array(params, f"ue{i}", Nt))
        pe = pe.at[i, :].set(extract_params_as_array(params, f"pe{i}", Nt))
        efract = efract.at[i, :].set(extract_params_as_array(params, f"efract{i}", Nt))
    

    Ti = jnp.zeros((Nions, Nt))
    ui = jnp.zeros((Nions, Nt))
    pi = jnp.zeros((Nions, Nt))
    ifract = jnp.zeros((Nions, Nt))
    for i in range(Nions):
        Ti = Ti.at[i, :].set(extract_params_as_array(params, f"Ti{i}", Nt))
        ui = ui.at[i, :].set(extract_params_as_array(params, f"ui{i}", Nt))
        pi = pi.at[i, :].set(extract_params_as_array(params, f"pi{i}", Nt))
        ifract = ifract.at[i, :].set(extract_params_as_array(params, f"ifract{i}", Nt))

    # Read measurement/static settings required by the forward model
    # efract and ifract are now fitted moments (in `params`), so read
    # ion composition and optics from measurement_settings
    ion_z = measurement_settings["ion_z"]
    ion_a = measurement_settings["ion_a"]
    wavelengths = measurement_settings["wavelengths"]
    probe_wavelength = measurement_settings["probe_wavelength"]
    probe_vec = measurement_settings["probe_vec"]
    scatter_vec = measurement_settings["scatter_vec"]
    ue_dir = measurement_settings["ue_dir"]
    ui_dir = measurement_settings["ui_dir"]
    instr_func_arr = measurement_settings.get("instr_func_arr", None)
    normalization_type = measurement_settings.get("normalization_type", "integral")
    normalization_scale = measurement_settings.get("normalization_scale", 1)

    t = 0*np.linspace(0, 3, Nt) #ns time
    tau = 1.5 #time over which plasma parameters vary

    #Plasma parameters
    ne = 1e20 * np.exp(-t / tau)
    ue = 0e6 * np.exp(-np.sqrt(t / tau))
    ui = 0*np.exp(-np.sqrt(t / tau))#ue + 2e5 * np.exp(-t / tau) - 1e5
    Te = 100 * np.exp(-t / tau)
    Ti = 200 - np.sqrt(t) * 100

    # efract and ifract have been populated in the species loops above

    # Call forward model to get the fit
    fit = _scattered_power_wavelength(
        n=n * 1e6,
        ue=ue,
        ui=ui,
        Te= np.array([Te]) * e / kB,
        Ti= np.array([Ti]) * e / kB,
        pe=pe,
        pi=pi,
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_a=ion_a,
        wavelengths=wavelengths,
        probe_wavelength=probe_wavelength,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        ue_dir=ue_dir,
        ui_dir=ui_dir,
        instr_func_arr=instr_func_arr,
        normalization_type=normalization_type,
        normalization_scale=normalization_scale,
    )

    # Dummy data arrays (shapes may differ depending on forward model)
    Skw_data = np.ones((Nk, Nt))*1e-14
    Skw_var = (Skw_data * 0.05)**2

    try:
        result = _log_posterior(params, Skw_data, Skw_var, measurement_settings, penalty_settings=None, use_penalty=False)
        print("_log_posterior returned:", result)
    except Exception as e:
        print("_log_posterior raised an exception:")
        import traceback
        traceback.print_exc()
