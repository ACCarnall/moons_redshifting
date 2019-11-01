import numpy as np
import glob
import time
import os

import new_moonz

from astropy.io import fits

# Exposure time for inputs
exp = 8

def get_wavs(ID, band):

    hdulist = fits.open("../data/spectra_v0.1.0_" + str(exp) + "h/" + band + "/" + ID + "_" + band + ".fits")
    min_wav = hdulist[1].header["CRVAL1"]
    dwav = hdulist[1].header["CDELT1"]
    max_wav = min_wav + dwav*(hdulist[1].data.shape[0])

    wavs = np.arange(min_wav, max_wav, dwav)

    if wavs.shape[0] > hdulist[1].data.shape[0]:
        return wavs[:-1]

    return wavs


def load_spectra(ID, wavs):
    # Load fits files
    hdulist_i = fits.open("../data/spectra_v0.1.0_" + str(exp) + "h/RI/" + ID + "_RI.fits")
    hdulist_j = fits.open("../data/spectra_v0.1.0_" + str(exp) + "h/YJ/" + ID + "_YJ.fits")
    hdulist_h = fits.open("../data/spectra_v0.1.0_" + str(exp) + "h/H/" + ID + "_H.fits")

    # Generate spectrum objects
    spec_i = np.c_[wavs[0], hdulist_i[1].data, hdulist_i[2].data]
    spec_j = np.c_[wavs[1], hdulist_j[1].data, hdulist_j[2].data]
    spec_h = np.c_[wavs[2], hdulist_h[1].data, hdulist_h[2].data]

    spec_i = spec_i[np.invert(hdulist_i[3].data.astype(bool)), :]
    spec_j = spec_j[np.invert(hdulist_j[3].data.astype(bool)), :]
    spec_h = spec_h[np.invert(hdulist_h[3].data.astype(bool)), :]

    # Bin spectra
    spec_i = bin(spec_i, 8)
    spec_j = bin(spec_j, 8)
    spec_h = bin(spec_h, 8)

    return np.concatenate([spec_i, spec_j, spec_h], axis=0)


def bin(spectrum, binn):
    """ Bins up two/three column spectrum by a specified factor. """

    binn = int(binn)
    nbins = len(spectrum)//binn
    binspec = np.zeros((nbins, spectrum.shape[1]))

    for i in range(binspec.shape[0]):
        spec_slice = spectrum[i*binn:(i+1)*binn, :]

        mask = (spec_slice[:, 1] != 0.)

        binspec[i, 0] = np.mean(spec_slice[:, 0])

        if np.sum(mask) == 0:
            binspec[i, 1] = 0.

        else:
            binspec[i, 1] = np.mean(spec_slice[mask, 1])

        if spectrum.shape[1] == 3:
            if np.sum(mask) == 0:
                binspec[i, 2] = 9.9*10**99

            else:
                binspec[i, 2] = (1./float(np.sum(mask))
                                 * np.sqrt(np.sum(spec_slice[mask, 2]**2)))

    return binspec


# Get list of objects to fit
os.chdir("../data/spectra_v0.1.0_" + str(exp) + "h/RI/")
IDs = glob.glob("*RI*")
os.chdir("../..")

#IDs = IDs[:10]

z_input = np.zeros(len(IDs))

for i in range(len(IDs)):
    IDs[i] = IDs[i][:-8]
    z_input[i] = float(IDs[i].split("_")[3][1:])

# Load first observed spectra to get its length
wavs_i = get_wavs(IDs[0], "RI")
wavs_j = get_wavs(IDs[0], "YJ")
wavs_h = get_wavs(IDs[0], "H")
wavs = [wavs_i, wavs_j, wavs_h]
spec = load_spectra(IDs[0], wavs)
spec_wavs = spec[:, 0] # Wavelength sampling of observed spectra

# Make 2D array to hold all observed spectra and all errors
all_spectra = np.zeros((len(IDs), spec.shape[0]))
all_spectra_errs = np.zeros((len(IDs), spec.shape[0]))

# Find nan pixels which are in ANY spectra: to be masked in ALL spectra
# Load up the spectral data and mask all nan pixels
for i in range(len(IDs)):
    print (i)
    spec_load = load_spectra(IDs[i], wavs)
    all_spectra[i, :] = spec_load[:, 1]
    all_spectra_errs[i, :] = spec_load[:, 2]
    norm = np.mean(all_spectra[i, :])
    all_spectra[i, :] /= norm
    all_spectra_errs[i, :] /= norm

hdus = [fits.PrimaryHDU(),
        fits.ImageHDU(name="wavs", data=spec_wavs),
        fits.ImageHDU(name="spectra", data=all_spectra),
        fits.ImageHDU(name="spectra_errs", data=all_spectra_errs)]

hdulist = fits.HDUList(hdus)

if os.path.exists("all_data_" + str(exp) + "hour.fits"):
    os.system("rm all_data_" + str(exp) + "hour.fits")

hdulist.writeto("all_data_" + str(exp) + "hour.fits")

fit = new_moonz.batch_fit(IDs, spec_wavs, all_spectra, all_spectra_errs)

time0 = time.time()
fit.fit()
print(time.time() - time0)
