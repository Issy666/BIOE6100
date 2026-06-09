# -*- coding: utf-8 -*-
"""
compare_lif_vs_lnp_isabel_palm.py - palm-skin counterpart of
compare_lif_vs_lnp_isabel.py.

Both name_on_palm_LIF.py (LIF) and name_on_palm.py (LNP) scan the same ISABEL
image across artificial palm skin (WIDTH=90 mm, HEIGHT=12 mm, SHIFT=0.2 mm,
SPEED=20 mm/s) and record the central receptor's Va trace per scan row.

Quantifies similarity with three metrics, per afferent type:
  (i)  r_map  : Pearson correlation of full 2-D spike-density maps
                (rows x position-bins).
  (ii) r_row  : Pearson correlation of per-row total spike counts.
  (iii)totals : total spike counts (calibration check).

Output: saved_figs/lif_vs_lnp_isabel_palm_corr.png + printed numbers.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# Must match name_on_palm.py / name_on_palm_LIF.py
WIDTH_MM   = 90.0
HEIGHT_MM  = 12.0
SHIFT      = 0.2
SPEED      = 20.0
SIMDT      = 0.001
SPIKE_VAL  = 0.04
N_X_BINS   = 60

LIF_PATH = 'Data/name_isabel_sim_res_palm_LIF.npy'
LNP_PATH = 'Data/name_isabel_sim_res_palm.npy'
OUT_PATH = 'saved_figs/lif_vs_lnp_isabel_palm_corr.png'


def va_to_spike_density_map(sim_res_per_type, n_x_bins=N_X_BINS):
    n_rows = len(sim_res_per_type)
    out = np.zeros((n_rows, n_x_bins), dtype=int)
    bin_edges = np.linspace(0, WIDTH_MM, n_x_bins + 1)
    for r in range(n_rows):
        va = np.asarray(sim_res_per_type[r])
        spike_x = SPEED * SIMDT * np.where(va == SPIKE_VAL)[0]
        out[r], _ = np.histogram(spike_x, bins=bin_edges)
    return out


def safe_pearson(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return float('nan'), float('nan')
    return pearsonr(a, b)


lif = np.load(LIF_PATH, allow_pickle=True)
lnp = np.load(LNP_PATH, allow_pickle=True)
types = ['SA1', 'RA1', 'PC']
colours = ['C2', 'C0', 'm']

maps_lif, maps_lnp, results = {}, {}, {}
for ch, name in enumerate(types):
    m_lif = va_to_spike_density_map(lif[ch])
    m_lnp = va_to_spike_density_map(lnp[ch])
    maps_lif[name] = m_lif
    maps_lnp[name] = m_lnp

    r_map, _ = safe_pearson(m_lif, m_lnp)
    r_row, _ = safe_pearson(m_lif.sum(axis=1), m_lnp.sum(axis=1))
    results[name] = dict(r_map=r_map, r_row=r_row,
                        total_lif=int(m_lif.sum()),
                        total_lnp=int(m_lnp.sum()))

print('\nLIF vs LNP "ISABEL" on palm -- similarity:')
print('-' * 62)
print(f"{'type':<5}  {'r(map)':>8}  {'r(row sum)':>11}  "
      f"{'total LIF':>10}  {'total LNP':>10}")
for name in types:
    r = results[name]
    print(f"{name:<5}  {r['r_map']:>8.3f}  {r['r_row']:>11.3f}  "
          f"{r['total_lif']:>10d}  {r['total_lnp']:>10d}")
print()


fig, axes = plt.subplots(3, 3, figsize=(14, 8))
extent = [0, WIDTH_MM, 0, HEIGHT_MM]
for ch, name in enumerate(types):
    m_lif = maps_lif[name]
    m_lnp = maps_lnp[name]
    vmax  = max(m_lif.max(), m_lnp.max(), 1)

    axes[ch, 0].imshow(m_lif, aspect='auto', cmap='Greys',
                       vmin=0, vmax=vmax, extent=extent, origin='lower')
    axes[ch, 0].set_title(f'{name} -- LIF map', fontsize=10)
    axes[ch, 0].set_ylabel('Scan row (y) [mm]')
    if ch == 2:
        axes[ch, 0].set_xlabel('Position (x) [mm]')

    axes[ch, 1].imshow(m_lnp, aspect='auto', cmap='Greys',
                       vmin=0, vmax=vmax, extent=extent, origin='lower')
    axes[ch, 1].set_title(f'{name} -- LNP map', fontsize=10)
    if ch == 2:
        axes[ch, 1].set_xlabel('Position (x) [mm]')

    axes[ch, 2].scatter(m_lif.ravel(), m_lnp.ravel(),
                        s=6, alpha=0.3, color=colours[ch])
    lim = max(vmax, 1)
    axes[ch, 2].plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5)
    axes[ch, 2].set_title(f"{name} -- pixel scatter, r = {results[name]['r_map']:.3f}",
                          fontsize=10)
    axes[ch, 2].set_xlabel('LIF spike count / bin')
    if ch == 1:
        axes[ch, 2].set_ylabel('LNP spike count / bin')
    axes[ch, 2].set_aspect('equal', 'box')
    axes[ch, 2].set_xlim(-0.5, lim + 0.5)
    axes[ch, 2].set_ylim(-0.5, lim + 0.5)

fig.suptitle('LIF vs LNP central-receptor spike density on the "ISABEL" scan -- PALM',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
print('Saved figure to', OUT_PATH)
