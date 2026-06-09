# -*- coding: utf-8 -*-
"""
plot_poisson_demo.py - illustrate the Poisson spike-generation stage of LNP.

Given a time-varying rate lambda(t) (the output of the static nonlinearity),
spikes are drawn as Bernoulli(lambda * dt) per timestep, independently across
trials. The resulting variability is what gives LNP its Fano factor near 1.

Three-panel figure:
  (1) rate curve lambda(t) -- a transient + plateau, like an SA1 ramp response
  (2) raster of 10 Poisson spike trials drawn from that rate
  (3) histogram of total spike counts across many trials vs a Poisson reference

Output: saved_figs/poisson_demo.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

DT = 0.001        # 1 ms
T_TOTAL = 0.7     # seconds
N_TRIALS_RASTER = 10
N_TRIALS_HIST = 500
RNG_SEED = 0

# ---------------------------------------------------------------------------
# Build an SA1-like rate: 100 ms ramp -> 500 ms hold -> 100 ms release
# ---------------------------------------------------------------------------
t = np.arange(0, T_TOTAL, DT)
lam = np.zeros_like(t)
# baseline
lam += 5.0
# transient at onset (100--150 ms) and offset (600--650 ms)
lam += 180.0 * np.exp(-((t - 0.10) ** 2) / (2 * 0.012 ** 2))
lam += 120.0 * np.exp(-((t - 0.60) ** 2) / (2 * 0.012 ** 2))
# sustained plateau during hold
hold_mask = (t > 0.10) & (t < 0.60)
lam[hold_mask] += 40.0


# ---------------------------------------------------------------------------
# Draw spike trains
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RNG_SEED)
spikes = rng.uniform(size=(N_TRIALS_HIST, t.size)) < lam * DT
raster_spikes = spikes[:N_TRIALS_RASTER]
spike_counts = spikes.sum(axis=1)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=False,
                        gridspec_kw=dict(height_ratios=[1, 1.4, 1], hspace=0.45))

# Panel 1: rate curve
axes[0].plot(t * 1000, lam, color='#ff7f0e', lw=1.6)
axes[0].set_ylabel('$\\lambda(t)$ [Hz]')
axes[0].set_xlim(0, T_TOTAL * 1000)
axes[0].set_title('(a) Rate from nonlinearity', loc='left', fontsize=10)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
axes[0].set_xticklabels([])

# Panel 2: raster of N_TRIALS_RASTER Poisson trials
for trial in range(N_TRIALS_RASTER):
    spike_t = t[raster_spikes[trial]] * 1000
    axes[1].vlines(spike_t, trial + 0.1, trial + 0.9, color='k', lw=0.8)
axes[1].set_xlim(0, T_TOTAL * 1000)
axes[1].set_ylim(0, N_TRIALS_RASTER)
axes[1].set_ylabel('Trial')
axes[1].set_xlabel('Time [ms]')
axes[1].set_title('(b) Poisson spike trains (10 trials from the same rate)',
                  loc='left', fontsize=10)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

# Panel 3: spike-count histogram with Poisson reference
mean_count = spike_counts.mean()
bins = np.arange(spike_counts.min(), spike_counts.max() + 2) - 0.5
axes[2].hist(spike_counts, bins=bins, density=True, color='lightgrey',
             edgecolor='k', linewidth=0.5, label=f'{N_TRIALS_HIST} trials')
ks = np.arange(spike_counts.min(), spike_counts.max() + 1)
axes[2].plot(ks, poisson.pmf(ks, mean_count), 'ro-', ms=4,
             label=f'Poisson($\\mu$={mean_count:.1f})')
fano = spike_counts.var() / spike_counts.mean()
axes[2].set_title(f'(c) Spike-count distribution --- Fano = {fano:.2f}',
                  loc='left', fontsize=10)
axes[2].set_xlabel('Spikes per trial')
axes[2].set_ylabel('Density')
axes[2].legend(fontsize=8)
axes[2].spines['top'].set_visible(False); axes[2].spines['right'].set_visible(False)

fig.tight_layout()
out = 'saved_figs/poisson_demo.png'
fig.savefig(out, bbox_inches='tight', dpi=200)
print('Saved figure to', out)
