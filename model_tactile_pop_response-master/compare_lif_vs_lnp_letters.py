# -*- coding: utf-8 -*-
"""
compare_lif_vs_lnp_letters.py - side-by-side comparison of the LIF and LNP
spike generators on the embossed-letters fingertip scan, alongside the
published Saal 2017 reference (3-column figure: LIF | LNP | Saal).

Pipeline:
  1. Load `saved_figs/letters_120-12.jpg`.
  2. Convert to an EEPS.
  3. Build SA1 / RA1 / PC populations on the fingertip.
  4. Multi-row scan trajectory (Fig.8ab structure).
  5. Run LIF on every row, store per-row central-receptor Va traces.
  6. Run LNP on every row, store per-row central-receptor Va traces.
  7. Plot three columns:
        column 1 : LIF EPS reference + SA1/RA1/PC rasters
        column 2 : LNP EPS reference + SA1/RA1/PC rasters
        column 3 : `saved_figs/ob_letter_spking.png`  (Saal 2017 reference)

Outputs:
  Data/forms_letters_lif.npy   - LIF per-row Va traces
  Data/forms_letters_lnp.npy   - LNP per-row Va traces
  saved_figs/lnp_vs_lif_letters.png  - the combined comparison figure
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

import Receptors as rslib
import simset as mysim
import img_to_eqstimuli as imeqst
import Lnp as lnplib


# ---------------------------------------------------------------------------
# Stimulus / scan parameters - match Fig.8ab structure [1].
# ---------------------------------------------------------------------------
WIDTH_MM   = 120.0     # physical width of the letters image on the skin
HEIGHT_MM  = 12.0      # physical height of the letters image
SHIFT      = 0.2       # mm - vertical spacing between scan rows
SPEED      = 20        # mm/s - horizontal scan velocity
PF         = 0.35      # N - constant pressing force during the scan
SIMDT      = 0.001     # s - 1 ms timestep
SIMT       = WIDTH_MM / SPEED   # one sweep = WIDTH_MM/SPEED seconds
DOTH       = 1         # EEPS height scale (1 = unchanged binarised heights)

LETTER_IMG = 'saved_figs/letters_120-12.jpg'
OB_REF_IMG = 'saved_figs/ob_letter_spking.png'   # Saal-2017-style published reference
OUT_PATH   = 'saved_figs/lnp_vs_lif_letters.png'

# Per-afferent LNP parameters now live as PER_TYPE_DEFAULTS in Lnp.py.
# LnpReceptors picks them up automatically from base.Ttype; override here only
# if you want this script to differ from the canonical set.
LNP_SEED = 42


# ---------------------------------------------------------------------------
# Step 1-2 - load image and convert to EEPS
# ---------------------------------------------------------------------------
img = Image.open(LETTER_IMG)
res_buf, eq_stimuli, eeps, eps = imeqst.constructing_equivalent_probe_stimuli_from_image(
    img, WIDTH_MM, HEIGHT_MM, mysim.fingertiproi
)
Aeeps = eeps
Aeeps[1] = Aeeps[1] * DOTH


# ---------------------------------------------------------------------------
# Step 3 - build receptor populations on the fingertip
# ---------------------------------------------------------------------------
Ttype_buf = ['SA1', 'RA1', 'PC']
tsensors  = []
pbuf = np.load('Data/loc_pos_buf_fingertip.npy', allow_pickle=True)
for ch in range(len(Ttype_buf)):
    s = rslib.tactile_receptors(Ttype=Ttype_buf[ch])
    s.set_population(pbuf[ch][0], pbuf[ch][1],
                     simTime=SIMT, sample_rate=1/SIMDT,
                     Density=pbuf[ch][2], roi=mysim.fingertiproi)
    tsensors.append(s)


# ---------------------------------------------------------------------------
# Step 4 - multi-row scan trajectory
# ---------------------------------------------------------------------------
PF_arr = PF * np.ones(int(SIMT / SIMDT))
ips = imeqst.img_stimuli_scaning_with_uniformal_speed(
    SIMDT, SIMT, PF_arr, SPEED,
    0, WIDTH_MM,
    0, HEIGHT_MM,
    SHIFT,
)


# ---------------------------------------------------------------------------
# Step 5 - LIF scan
# ---------------------------------------------------------------------------
print('Scanning letters with LIF: {} types x {} rows...'.format(
    len(Ttype_buf), len(ips)))
sim_res_lif = []
for ch, name in enumerate(Ttype_buf):
    per_row = []
    for row in range(len(ips)):
        tsensors[ch].population_simulate(EEQS=Aeeps,
                                         Ips=[ips[row], 'Pressure'],
                                         noise=0, disinf=False)
        sel = tsensors[ch].points_mapping_entrys(np.array([[0, 0]]))[0]
        per_row.append(np.array(tsensors[ch].Va[sel, :]))
    sim_res_lif.append(per_row)
    print('  LIF {} done'.format(name))
np.save('Data/forms_letters_lif.npy', sim_res_lif)


# ---------------------------------------------------------------------------
# Step 6 - LNP scan
# Note: LnpReceptors wraps the same tsensors object and writes to its Va,
# overwriting the LIF traces. That's fine because we've already saved them.
# ---------------------------------------------------------------------------
print('Scanning letters with LNP: {} types x {} rows...'.format(
    len(Ttype_buf), len(ips)))
sim_res_lnp = []
for ch, name in enumerate(Ttype_buf):
    lnp = lnplib.LnpReceptors(tsensors[ch],
                              rng_seed=LNP_SEED + ch)
    per_row = []
    for row in range(len(ips)):
        lnp.population_simulate(EEQS=Aeeps,
                                Ips=[ips[row], 'Pressure'],
                                noise=0, disinf=False)
        sel = tsensors[ch].points_mapping_entrys(np.array([[0, 0]]))[0]
        per_row.append(np.array(tsensors[ch].Va[sel, :]))
    sim_res_lnp.append(per_row)
    print('  LNP {} done'.format(name))
np.save('Data/forms_letters_lnp.npy', sim_res_lnp)


# ---------------------------------------------------------------------------
# Step 7 - plot 3-column comparison: LIF | LNP | published Saal reference
# ---------------------------------------------------------------------------
def plot_compare():
    buf = eq_stimuli
    sres_lif = np.load('Data/forms_letters_lif.npy', allow_pickle=True)
    sres_lnp = np.load('Data/forms_letters_lnp.npy', allow_pickle=True)

    # Scan-row -> y mapping. Row 0 at top (HEIGHT_MM), last row at bottom (0).
    num = int(HEIGHT_MM / SHIFT)
    sel_points = np.vstack([0 * np.ones(num),
                            np.linspace(HEIGHT_MM, 0, num)]).T

    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           height_ratios=[1, 2, 2, 2],
                           width_ratios=[1, 1, 1.05],
                           hspace=0.35, wspace=0.25)

    def _eps_panel(ax, label):
        ax.scatter(buf[:, 0], buf[:, 1], s=0.01,
                   c=1e3 * buf[:, 5] * DOTH,
                   cmap=plt.cm.Greys, vmin=0, vmax=1)
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        ax.set_xticks(np.arange(0, WIDTH_MM + WIDTH_MM / 10, WIDTH_MM / 10))
        ax.tick_params(axis='x', labelsize=6)
        ax.set_yticks([0, HEIGHT_MM / 2, HEIGHT_MM])
        ax.tick_params(axis='y', labelsize=7)
        ax.set_title(label, fontsize=11, pad=2)
        ax.set_ylabel('EPS', fontsize=8)

    def _raster_panel(ax, sres_col, ch, anchor_ax):
        ax.sharex(anchor_ax)
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        rows_to_plot = min(num, len(sres_col[ch]))
        for i in range(rows_to_plot):
            spike_t = SIMDT * np.where(sres_col[ch][i] == 0.04)[0]
            ax.scatter(spike_t * SPEED,
                       sel_points[i, 1] * np.ones(len(spike_t)),
                       c=mysim.colors[ch], marker='.', s=0.01)
        ax.set_xticks(np.arange(0, WIDTH_MM + WIDTH_MM / 10, WIDTH_MM / 10))
        ax.tick_params(axis='x', labelsize=7)
        ax.set_yticks([0, 4, 8, 12])
        ax.tick_params(axis='y', labelsize=7)
        if ch == 2:
            ax.set_xlabel('Position [mm]', fontsize=9)
        if ch == 1:
            ax.set_ylabel('Distance [mm]', fontsize=9)
        ax2 = ax.twinx()
        ax2.spines['top'].set_color('None')
        ax2.spines['right'].set_color('None')
        ax2.set_yticks([])
        ax2.set_ylabel(Ttype_buf[ch], fontsize=9)

    # --- Column 1: LIF ---
    ax_lif_eps = fig.add_subplot(gs[0, 0])
    _eps_panel(ax_lif_eps, '(a) Ouyang LIF Model ')
    for ch in range(3):
        ax = fig.add_subplot(gs[ch + 1, 0])
        _raster_panel(ax, sres_lif, ch, ax_lif_eps)

    # --- Column 2: LNP ---
    ax_lnp_eps = fig.add_subplot(gs[0, 1])
    _eps_panel(ax_lnp_eps, '(b) LNP')
    for ch in range(3):
        ax = fig.add_subplot(gs[ch + 1, 1])
        _raster_panel(ax, sres_lnp, ch, ax_lnp_eps)

    # --- Column 3: published Saal 2017 reference (full height) ---
    ax_ref = fig.add_subplot(gs[:, 2])
    if os.path.exists(OB_REF_IMG):
        ref_img = Image.open(OB_REF_IMG)
        ax_ref.imshow(ref_img)
        ax_ref.set_title('(c) (Vega-Bermudez et al, 1991) SA1 Receptors ', fontsize=11, pad=2)
    else:
        ax_ref.text(0.5, 0.5, 'Reference image not found:\n' + OB_REF_IMG,
                    ha='center', va='center', fontsize=10, color='red',
                    transform=ax_ref.transAxes)
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])
    for s in ax_ref.spines.values():
        s.set_visible(False)

    plt.savefig(OUT_PATH, bbox_inches='tight', dpi=300)
    print('Saved figure to', OUT_PATH)


plot_compare()
plt.show()
