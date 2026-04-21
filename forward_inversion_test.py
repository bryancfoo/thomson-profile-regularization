"""
Test script for fitting Thomson scattered spectra against FLASH simulation results.

Reads synthetic simulation data from LaserSlab_FLASH.h5, which contains plasma
moments in hf["fields"][{field_name}] and coordinates in hf["coords"]["t"] and ["x"].
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

H5_FILE = "LaserSlab_FLASH.h5"


def get_field(field_name: str) -> np.ndarray:
    """Read a field from the FLASH HDF5 file.

    Parameters
    ----------
    field_name : str
        Name of the field to read from hf["fields"][field_name].

    Returns
    -------
    np.ndarray
        The field data array, transposed to (x, t) order.
    """
    with h5py.File(H5_FILE, "r") as hf:
        return hf["fields"][field_name][:].T


def get_coords() -> tuple[np.ndarray, np.ndarray]:
    """Read t and x coordinate arrays from the FLASH HDF5 file.

    Returns t in ns (converted from s) and x in mm (converted from cm).
    """
    with h5py.File(H5_FILE, "r") as hf:
        t = hf["coords"]["t"][:] * 1e9   # s -> ns
        x = hf["coords"]["x"][:] * 10    # cm -> mm
    return t, x


def main():
    t, x = get_coords()

    fields = ["dens",    "velx",  "tele",  "tion"]
    labels = ["Density", "Vel. x", "T_e",   "T_i"]
    cmaps  = ["magma_r", "bwr",   "hot",   "hot"]
    norms  = [LogNorm(), None,    LogNorm(), LogNorm()]

    _, axes = plt.subplots(ncols=4, figsize=(18, 5))
    for i, (ax, field_name, label, cmap, norm) in enumerate(zip(axes, fields, labels, cmaps, norms)):
        data = get_field(field_name)
        pcm = ax.pcolormesh(t, x, data, shading="auto", cmap=cmap, norm=norm)
        plt.colorbar(pcm, ax=ax)
        ax.set_xlabel("t (ns)")
        ax.set_title(label)
        if i == 0:
            ax.set_ylabel("x (mm)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
