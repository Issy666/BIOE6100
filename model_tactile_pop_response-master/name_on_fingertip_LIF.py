# -*- coding: utf-8 -*-
"""
Ibtest.py - LIF equivalent of name_on_skin.py (name_on_fingertip).

Scans "ISABEL" across the fingertip using the original LIF spike generator
(tactile_receptors.population_simulate), then plots the per-row central-
receptor spike raster in the Fig.8b [1] style. Mirrors name_on_skin.py exactly,
EXCEPT the spike-generation stage uses LIF instead of LNP - so this is the
ground-truth model for comparison against the LNP rasters.

Pipeline:
  1. Generate (or load) the "ISABEL" image at WIDTH_MM x HEIGHT_MM.
  2. Convert to an EEPS via the standard image-processing pipeline.
  3. Build SA1 / RA1 / PC populations on the fingertip.
  4. Multi-row scan trajectory (Fig.8ab structure).
  5. For each (afferent, scan_row) pair, run LIF population_simulate and
     record the central receptor's Va trace.
  6. Plot Fig.8b-style raster: EPS reference on top + 3 raster panels below.

Outputs:
  saved_figs/letters_ISABEL.jpg            - source image (shared with the
                                              LNP script; auto-generated)
  saved_figs/name_isabel_population_LIF.png  - Fig.8b-style raster
  Data/name_isabel_sim_res_LIF.npy           - per-row Va traces

@author: Isabel Barton
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import Receptors as rslib
import simset as mysim
import img_to_eqstimuli as imeqst


# ---------------------------------------------------------------------------
# Stimulus / scan parameters - IDENTICAL to name_on_skin.py so the LIF and
# LNP rasters are directly comparable.
# ---------------------------------------------------------------------------
WIDTH_MM   = 90.0      # physical width of the ISABEL image on the skin (matches name_on_fingertip.py LNP)
HEIGHT_MM  = 12.0      # physical height (vertical extent of the letters)
SHIFT      = 0.2       # mm - vertical spacing between scan rows
SPEED      = 20        # mm/s - horizontal scan velocity
PF         = 0.35      # N - constant pressing force during the scan
SIMDT      = 0.001     # s - 1 ms timestep
SIMT       = WIDTH_MM / SPEED   # one sweep = WIDTH_MM/SPEED seconds
DOTH       = 1         # EEPS height scale (1 = unchanged binarised heights)

LETTER_IMG = 'saved_figs/letters_ISABEL.jpg'


# ---------------------------------------------------------------------------
# Step 1 - make (or load) the "ISABEL" source image (identical helper to
# the one in name_on_skin.py; duplicated here so Ibtest.py is self-contained).
# ---------------------------------------------------------------------------
def make_isabel_image(path, width_px=900, height_px=180, word='ISABEL'):
    """Black background, white bold sans-serif text."""
    img = Image.new('RGB', (width_px, height_px), 'black')
    draw = ImageDraw.Draw(img)
    font = None
    for font_path in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/Library/Fonts/Arial Bold.ttf',
        'C:/Windows/Fonts/arialbd.ttf',
    ]:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, int(height_px * 0.8))
            break
    if font is None:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((width_px - tw) // 2 - bbox[0],
           (height_px - th) // 2 - bbox[1])
    draw.text(pos, word, fill='white', font=font)
    img.save(path)
    return img


if not os.path.exists(LETTER_IMG):
    make_isabel_image(LETTER_IMG)
img = Image.open(LETTER_IMG)


# ---------------------------------------------------------------------------
# Step 2 - image -> EEPS
# ---------------------------------------------------------------------------
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
# Step 4 - multi-row scan trajectory (Fig.8ab structure)
# ---------------------------------------------------------------------------
PF_arr = PF * np.ones(int(SIMT / SIMDT))
ips = imeqst.img_stimuli_scaning_with_uniformal_speed(
    SIMDT, SIMT, PF_arr, SPEED,
    0, WIDTH_MM,
    0, HEIGHT_MM,
    SHIFT,
)


# ---------------------------------------------------------------------------
# Step 5 - LIF scan, central-receptor recording per scan row
# This is the SAME loop structure as name_on_skin.py but with
# tactile_receptors.population_simulate (LIF) instead of LnpReceptors (LNP).
# ---------------------------------------------------------------------------
print('Scanning ISABEL with LIF: {} types x {} rows...'.format(
    len(Ttype_buf), len(ips)))
sim_res = []
for ch, name in enumerate(Ttype_buf):
    per_row = []
    for row in range(len(ips)):
        tsensors[ch].population_simulate(EEQS=Aeeps,
                                         Ips=[ips[row], 'Pressure'],
                                         noise=0, disinf=False)
        sel = tsensors[ch].points_mapping_entrys(np.array([[0, 0]]))[0]
        per_row.append(np.array(tsensors[ch].Va[sel, :]))
    sim_res.append(per_row)
    print('  {} done'.format(name))

np.save('Data/name_isabel_sim_res_LIF.npy', sim_res)


# ---------------------------------------------------------------------------
# Step 6 - plot Fig.8b-style raster (identical layout to name_on_skin.py)
# ---------------------------------------------------------------------------
def print_name_raster_lif():
    plt.figure(figsize=(7, 4 * 0.7))
    plt.subplots_adjust(hspace=0.2)

    buf  = eq_stimuli
    sres = np.load('Data/name_isabel_sim_res_LIF.npy', allow_pickle=True)

    # Top panel - EPS reference (rasterised letter image).
    ax = plt.subplot(4, 1, 1)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.text(0, HEIGHT_MM + 2, '(b) Name on Fingertip - LIF', fontsize=12)
    ax.scatter(buf[:, 0], buf[:, 1], s=0.01,
               c=1e3 * buf[:, 5] * DOTH,
               cmap=plt.cm.Greys, vmin=0, vmax=1)
    plt.xticks(np.arange(0, WIDTH_MM + WIDTH_MM / 10, WIDTH_MM / 10),
               fontsize=6)
    plt.yticks([0, HEIGHT_MM / 2, HEIGHT_MM], fontsize=7)
    ax1 = ax.twinx()
    ax1.spines['top'].set_color('None')
    ax1.spines['right'].set_color('None')
    plt.yticks([])
    plt.ylabel('EPS', fontsize=8)

    # Scan-row -> y mapping. Row 0 at top (HEIGHT_MM), last row at bottom (0).
    num = int(HEIGHT_MM / SHIFT)
    sel_points = np.vstack([0 * np.ones(num),
                            np.linspace(HEIGHT_MM, 0, num)]).T

    # Per-afferent raster panels.
    for ch in range(len(Ttype_buf)):
        ax1 = plt.subplot(4, 1, ch + 2, sharex=ax)
        ax1.spines['top'].set_color('None')
        ax1.spines['right'].set_color('None')
        rows_to_plot = min(num, len(sres[ch]))
        for i in range(rows_to_plot):
            spike_t = SIMDT * np.where(sres[ch][i] == 0.04)[0]
            plt.scatter(spike_t * SPEED,
                        sel_points[i, 1] * np.ones(len(spike_t)),
                        c=mysim.colors[ch], marker='.', s=0.01)
        plt.xticks(np.arange(0, WIDTH_MM + WIDTH_MM / 10, WIDTH_MM / 10),
                   fontsize=7)
        plt.yticks([0, 4, 8, 12], fontsize=7)
        if ch == 2: plt.xlabel('Position [mm]', fontsize=10)
        if ch == 1: plt.ylabel('Distance [mm]', fontsize=10)
        ax2 = ax1.twinx()
        ax2.spines['top'].set_color('None')
        ax2.spines['right'].set_color('None')
        plt.yticks([])
        plt.ylabel(Ttype_buf[ch], fontsize=8)

    out = 'saved_figs/name_isabel_population_LIF.png'
    plt.savefig(out, bbox_inches='tight', dpi=300)
    print('Saved figure to', out)


print_name_raster_lif()
plt.show()
