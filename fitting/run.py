import numpy as np
import glob
import time
import os

import new_moonz

from astropy.io import fits


"""
os.chdir("../data/spectra_v0.1.0_1h/RI/")
IDs = np.array(glob.glob("*RI*"))
os.chdir("../../../fitting")

IDs = np.savetxt("IDs.txt", IDs, fmt="%s")
"""
IDs = np.loadtxt("IDs.txt", dtype="str")

hdulist = fits.open("../data/all_data_1hour.fits")
spec_wavs = hdulist[1].data
all_spectra = hdulist[2].data
all_spectra_errs = hdulist[3].data

"""
badz = np.loadtxt("bright_badz.txt", usecols=0, dtype="str")

mask = np.isin(IDs, badz)

all_spectra = all_spectra[mask, :]
all_spectra_errs = all_spectra_errs[mask, :]
IDs = IDs[mask]
"""

z_input = np.zeros(IDs.shape[0])
mag = np.zeros(IDs.shape[0])

for i in range(IDs.shape[0]):
    z_input[i] = IDs[i].split("_")[3][1:]
    mag[i] = IDs[i].split("_")[5][1:]


fit = new_moonz.batch_fit(IDs, spec_wavs, all_spectra, all_spectra_errs,
                          run="1_hour", make_plots=False, z_input=z_input,
                          n_components=20, n_train=10000)


time0 = time.time()
fit.fit()
print("Time taken:", time.time() - time0, "seconds")
