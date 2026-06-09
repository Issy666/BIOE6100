# -*- coding: utf-8 -*-
"""
plot_hand_densities.py - visualise afferent populations across all three
hand skin regions modelled in this codebase.

This is a Saal-Fig-1A-style summary of what's actually in the repo:
fingertip, back-of-finger, and palm. Each region has its own ROI polygon
and its own pre-generated SA1/RA1/PC receptor populations (from
`Data/loc_pos_buf_*.npy`). Densities follow Bensmaia / Saal et al. as
hard-coded in `simset.py`:

    fingertip:        SA1  72.2,  RA1 143.5,  PC  24.8  per cm²
    back-of-finger:   SA1  32.5,  RA1  40.9,  PC  11.3
    palm:             SA1   9.5,  RA1  26.3,  PC  11.1

Outputs:
  saved_figs/hand_densities.png   - receptor scatter + density bar chart
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import simset as mysim


# ---------------------------------------------------------------------------
# Load the three ROIs.
# `fingertiproi` is already centred at the origin by simset.py.
# `bfingerroi` and `plamroi` are loaded raw from the text files, so they
# stay in their original (uncentred) world coordinates.
# ---------------------------------------------------------------------------
fingertiproi = mysim.fingertiproi
bfingerroi   = np.loadtxt('Data/txtdata/bfinger_roi.txt')
plamroi      = np.loadtxt('Data/txtdata/plam_roi.txt')

# ---------------------------------------------------------------------------
# Load the three receptor populations. Each .npy contains a list of length 3
# (SA1, RA1, PC). For each afferent type:
#   pbuf[tp][0] = (N, 2) positions in mm  (same coord frame as ROI)
#   pbuf[tp][1] = (N, 2) grid indices for the resistance network
#   pbuf[tp][2] = scalar density in receptors / cm²
# ---------------------------------------------------------------------------
regions = [
    ('Fingertip',      fingertiproi,
        np.load('Data/loc_pos_buf_fingertip.npy', allow_pickle=True)),
    ('Back of finger', bfingerroi,
        np.load('Data/loc_pos_buf_bfinger.npy',  allow_pickle=True)),
    ('Palm',           plamroi,
        np.load('Data/loc_pos_buf_plam.npy',     allow_pickle=True)),
]

Ttype_buf = ['SA1', 'RA1', 'PC']
colors    = mysim.colors[:3]   # default project palette: ['g', 'b', 'm']


# ---------------------------------------------------------------------------
# Print a quick console summary so the numbers are visible even without
# looking at the figure.
# ---------------------------------------------------------------------------
print('{:<16} {:>16} {:>16} {:>16}'.format(
    'Region', 'SA1 (count/dens)', 'RA1 (count/dens)', 'PC (count/dens)'))
for name, _, pbuf in regions:
    cells = []
    for tp in range(3):
        n   = len(pbuf[tp][0])
        d   = pbuf[tp][2]
        cells.append('{:5d} / {:5.1f}'.format(n, d))
    print('{:<16} {:>16} {:>16} {:>16}'.format(name, *cells))


# ---------------------------------------------------------------------------
# Figure: top row = receptor scatter per region; bottom = density bar chart.
# Each region's scatter uses its own axis limits because the ROIs live in
# different world-coord ranges (fingertip ~10x12 mm, palm ~25+ mm).
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 8))

# --- Top row: 3 scatter panels, one per region ----------------------------
for i, (name, roi, pbuf) in enumerate(regions):
    ax = plt.subplot(2, 3, i + 1)

    # ROI outline + light fill so the skin patch is visible.
    roi_closed = np.vstack([roi, roi[0]])
    ax.fill(roi_closed[:, 0], roi_closed[:, 1],
            color='lightgray', alpha=0.3, zorder=0)
    ax.plot(roi_closed[:, 0], roi_closed[:, 1],
            color='k', lw=1.2, zorder=1)

    # Receptors per afferent type. Marker size kept small so the dense
    # fingertip layouts don't visually saturate.
    for tp, ttype in enumerate(Ttype_buf):
        positions = pbuf[tp][0]
        density   = pbuf[tp][2]
        count     = len(positions)
        ax.scatter(positions[:, 0], positions[:, 1],
                   s=6, c=colors[tp], alpha=0.7, edgecolors='none',
                   label='{}  n={}  ({:.1f}/cm²)'.format(
                       ttype, count, density))

    ax.set_aspect('equal')
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('x [mm]', fontsize=9)
    if i == 0:
        ax.set_ylabel('y [mm]', fontsize=9)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')

# --- Bottom row: grouped bar chart of densities ---------------------------
ax_bar = plt.subplot(2, 1, 2)

region_names = [name for name, _, _ in regions]
densities    = np.array([[pbuf[tp][2] for tp in range(3)]
                         for _, _, pbuf in regions])    # shape (3, 3)

x_pos = np.arange(len(region_names))
width = 0.25
for tp, ttype in enumerate(Ttype_buf):
    bars = ax_bar.bar(x_pos + (tp - 1) * width, densities[:, tp], width,
                      color=colors[tp], label=ttype,
                      edgecolor='k', linewidth=0.5)
    # Label each bar with its value.
    for j, b in enumerate(bars):
        ax_bar.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 1.5,
                    '{:.1f}'.format(densities[j, tp]),
                    ha='center', fontsize=8)

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(region_names, fontsize=10)
ax_bar.set_ylabel('Receptor density  [per cm²]', fontsize=10)
ax_bar.set_title('Afferent densities across hand regions  (Bensmaia / Saal)',
                 fontsize=11)
ax_bar.legend(loc='upper right', fontsize=9, framealpha=0.85)
ax_bar.spines['top'].set_color('None')
ax_bar.spines['right'].set_color('None')
ax_bar.set_ylim(0, densities.max() * 1.15)

fig.suptitle('Afferent layouts and densities across modelled hand regions',
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out_path = 'saved_figs/hand_densities.png'
fig.savefig(out_path, bbox_inches='tight', dpi=200)
print('\nSaved figure to', out_path)
plt.show()
