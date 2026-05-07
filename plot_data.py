import numpy as np
import matplotlib.pyplot as plt
import h5py


hf = h5py.File("example_data.h5", 'r')
print(hf.keys())
Pkl_data = np.array(hf["Pkl_data"])
wavelengths = np.array(hf["wavelengths"])
time = np.array(hf["time"])

print(Pkl_data)

plt.pcolormesh(Pkl_data)
plt.show()