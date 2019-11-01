import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import bagpipes as pipes
import sys

fig = plt.figure(figsize=(7, 5))
ax2 = plt.subplot()

exptimes = [8, 2, 1]
colors = ["red", "green", "blue"]

for j in range(3):
    run = str(exptimes[j]) + "_hour"

    results = np.loadtxt("best_z_" + run + ".txt", dtype="str")

    z_input = np.zeros(results.shape[0])
    mag = np.zeros(results.shape[0])

    for i in range(results.shape[0]):
        z_input[i] = results[i, 0].split("_")[3][1:]
        mag[i] = results[i, 0].split("_")[5][1:]

    z_best = results[:, 1].astype(float)

    correct = np.abs(z_best - z_input) < 0.01

    # Percentage success panel
    tot, edges = np.histogram(mag, bins=np.arange(18.75, 25.8, 0.5))
    corr, edges = np.histogram(mag[correct], bins=np.arange(18.75, 25.8, 0.5))
    perc = 100*corr/tot.astype(float)

    x, y = pipes.plotting.make_hist_arrays(edges, perc)
    ax2.plot(x, y, color=colors[j], lw=2, label="$\mathrm{" + str(exptimes[j]) + "\\ hours}$")
    ax2.plot([x[0], x[0]], [0., y[0]], color=colors[j], lw=2)
    ax2.plot([x[-1], x[-1]], [0., y[-1]], color=colors[j], lw=2)

    ax2.fill_between(x, np.zeros_like(y), y, color=colors[j], alpha=0.2)

    ax2.set_xlabel("$H_\mathrm{AB}$")
    ax2.set_ylabel("$\mathrm{Percentage\\ success}$")
    ax2.set_ylim(0., 100.)
    ax2.set_xlim(18.5, 26.)
    ax2.set_xticks(np.arange(19, 26))
    ax2.legend(frameon=False)
#ax2.set_title("$\mathrm{First\\ 1000\\ 1\\ hour\\ v0.1.0\\ spectra}$")
plt.savefig("plots/summary_comp.pdf", bbox_inches="tight", dpi=300)
