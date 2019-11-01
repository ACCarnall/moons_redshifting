import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import bagpipes as pipes
import sys

run = sys.argv[1]

results = np.loadtxt("best_z_" + run + ".txt", dtype="str")

z_input = np.zeros(results.shape[0])
mag = np.zeros(results.shape[0])

for i in range(results.shape[0]):
    z_input[i] = results[i, 0].split("_")[3][1:]
    mag[i] = results[i, 0].split("_")[5][1:]

z_best = results[:, 1].astype(float)

correct = np.abs(z_best - z_input) < 0.01

fig = plt.figure(figsize=(12, 5))
gs = mpl.gridspec.GridSpec(1, 2)

# z_input vs z_bestfit
ax1 = plt.subplot(gs[0])
ax1.scatter(z_input[np.invert(correct)], z_best[np.invert(correct)],
            color="dodgerblue", s=2, rasterized=True)

ax1.scatter(z_input[correct], z_best[correct], color="green", s=2,
            rasterized=True)

ax1.set_xlim(-0.1, 5.1)
ax1.set_ylim(-0.1, 5.1)
ax1.set_xlabel("$\mathrm{Input\\ redshift}$")
ax1.set_ylabel("$\mathrm{Best\\ fit\\ redshift}$")
ax1.plot([-0.1, 5.1], [-0.1, 5.1], color="gray", lw=1, alpha=0.75, zorder=1)
ax1.set_xticks(np.arange(0., 6.))

# Percentage success panel
ax2 = plt.subplot(gs[1])
tot, edges = np.histogram(mag, bins=np.arange(18.75, 25.8, 0.5))
corr, edges = np.histogram(mag[correct], bins=np.arange(18.75, 25.8, 0.5))
perc = 100*corr/tot.astype(float)

x, y = pipes.plotting.make_hist_arrays(edges, perc)
ax2.plot(x, y, color="red", lw=3)
ax2.plot([x[0], x[0]], [0., y[0]], color="red", lw=3)
ax2.plot([x[-1], x[-1]], [0., y[-1]], color="red", lw=3)

ax2.fill_between(x, np.zeros_like(y), y, color="red", alpha=0.3)

ax2.set_xlabel("$H_\mathrm{AB}$")
ax2.set_ylabel("$\mathrm{Percentage\\ success}$")
ax2.set_ylim(0., 100.)
ax2.set_xlim(18.5, 26.)
ax2.set_xticks(np.arange(19, 26))

#ax2.set_title("$\mathrm{First\\ 1000\\ 1\\ hour\\ v0.1.0\\ spectra}$")
plt.savefig("plots/summary_" + run + ".pdf", bbox_inches="tight", dpi=300)
