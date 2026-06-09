# -*- coding: utf-8 -*-
"""
name_on_palm_LIF.py - LIF equivalent of name_on_palm.py.

p as name_on_palm.py (ISABEL scanned across PALM receptors with
the palm ROI), but spike generation uses the original LIF model
(tactile_receptors.population_simulate) instead of the LNP wrapper. This
gives the ground-truth model for the LIF-vs-LNP comparison on the palm.

Pipeline:
  1. Generate (or load) the "ISABEL" image at WIDTH_MM x HEIGHT_MM.
  2. Convert to an EEPS using the PALM ROI for padding (matching the
     contact-window size the palm sensor expects).
  3. Build SA1 / RA1 / PC populations on the PALM (loc_pos_buf_plam.npy +
     mysim.plamroi). Receptor positions are clipped 0.1 mm inside the ROI
     bounding box to dodge the set_population boundary edge case.
  4. Multi-row scan trajectory across the ISABEL image (Fig.8ab structure).
  5. For each (afferent, scan_row) pair, run LIF population_simulate and
     record the spike train of the receptor closest to the PALM CENTRE
     (not (0, 0), since the palm coords are uncentred).
  6. Plot the result as: top panel = EPS reference; below = SA1/RA1/PC
     spike rasters with probe x on the horizontal axis and scan-row y
     (Distance) on the vertical.

Outputs:
  saved_figs/letters_ISABEL.jpg                       - source image (shared)
  saved_figs/name_isabel_population_palm_LIF.png      - Fig.8b-style raster
  Data/name_isabel_sim_res_palm_LIF.npy               - per-row Va traces

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
# Stimulus / scan parameters - IDENTICAL to name_on_palm.py.
# ---------------------------------------------------------------------------
WIDTH_MM   = 90.0      # physical width of the ISABEL image on the skin
HEIGHT_MM  = 12.0      # physical height (vertical extent of the letters)
SHIFT      = 0.2       # mm - vertical spacing between scan rows
SPEED      = 20        # mm/s - horizontal scan velocity
PF         = 0.35      # N - constant pressing force during the scan
SIMDT      = 0.001     # s - 1 ms timestep
SIMT       = WIDTH_MM / SPEED   # one sweep duration
DOTH       = 1         # EEPS height scale (1 = unchanged binarised heights)

LETTER_IMG = 'saved_figs/letters_ISABEL.jpg'


# ---------------------------------------------------------------------------
# Step 1 - make (or load) the "ISABEL" source image (same helper as the
# LNP scripts; duplicated here so this file is self-contained)
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
# Step 2 - image -> EEPS using the PALM ROI for padding
# ---------------------------------------------------------------------------
# constructing_equivalent_probe_stimuli_from_image pads the letter image by
# the ROI's bounding-box dimensions on each side. We pad with the palm
# dimensions because the sensor (built with roi=plamroi) expects palm-sized
# contact slices; padding with the fingertip ROI causes the simulator to
# slice past the EEPS and triggers IndexError inside population_simulate.
res_buf, eq_stimuli, eeps, eps = imeqst.constructing_equivalent_probe_stimuli_from_image(
    img, WIDTH_MM, HEIGHT_MM, mysim.plamroi
)
Aeeps = eeps
Aeeps[1] = Aeeps[1] * DOTH


# ---------------------------------------------------------------------------
# Step 3 - build PALM receptor populations
# ---------------------------------------------------------------------------
Ttype_buf = ['SA1', 'RA1', 'PC']
tsensors  = []
pbuf = np.load('Data/loc_pos_buf_plam.npy', allow_pickle=True)

# Clip receptor positions 0.1 mm inside the palm ROI bounding box. set_population
# computes pixel indices via uint16((r_pos - roi.min()) / Wc * Nrc), so a
# receptor sitting exactly on roi.max() gets index Nrc - one past the last
# valid index. 0.1 mm = Dbp/2 is well below the simulator's spatial resolution
# but well above the float64 -> uint16 rounding edge case.
_x_lo = mysim.plamroi[:, 0].min() + 0.1
_x_hi = mysim.plamroi[:, 0].max() - 0.1
_y_lo = mysim.plamroi[:, 1].min() + 0.1
_y_hi = mysim.plamroi[:, 1].max() - 0.1

for ch in range(len(Ttype_buf)):
    s = rslib.tactile_receptors(Ttype=Ttype_buf[ch])
    positions = np.asarray(pbuf[ch][0], dtype=float).copy()
    positions[:, 0] = np.clip(positions[:, 0], _x_lo, _x_hi)
    positions[:, 1] = np.clip(positions[:, 1], _y_lo, _y_hi)
    s.set_population(positions, pbuf[ch][1],
                     simTime=SIMT, sample_rate=1/SIMDT,
                     Density=pbuf[ch][2], roi=mysim.plamroi)
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
# Step 5 - LIF scan; record the receptor closest to the PALM CENTRE per row
# ---------------------------------------------------------------------------
# Palm coords are uncentred (plamroi values are in original world coords),
# so we can't query points_mapping_entrys([[0, 0]]) like the fingertip
# version does - (0, 0) is outside the palm entirely. Use the centre of the
# palm ROI's bounding box as the recording target instead.
palm_centre = np.array([
    (mysim.plamroi[:, 0].max() + mysim.plamroi[:, 0].min()) / 2.0,
    (mysim.plamroi[:, 1].max() + mysim.plamroi[:, 1].min()) / 2.0,
]).reshape(1, 2)

print('Scanning ISABEL with LIF on the palm: {} types x {} rows...'.format(
    len(Ttype_buf), len(ips)))
sim_res = []
for ch, name in enumerate(Ttype_buf):
    per_row = []
    for row in range(len(ips)):
        tsensors[ch].population_simulate(EEQS=Aeeps,
                                         Ips=[ips[row], 'Pressure'],
                                         noise=0, disinf=False)
        sel = tsensors[ch].points_mapping_entrys(palm_centre)[0]
        per_row.append(np.array(tsensors[ch].Va[sel, :]))
    sim_res.append(per_row)
    print('  {} done'.format(name))

np.save('Data/name_isabel_sim_res_palm_LIF.npy', sim_res)


# ---------------------------------------------------------------------------
# Step 6 - plot Fig.8b-style raster (identical layout to name_on_palm.py)
# ---------------------------------------------------------------------------
def print_name_raster_palm_lif():
    plt.figure(figsize=(7, 4 * 0.7))
    plt.subplots_adjust(hspace=0.2)

    buf  = eq_stimuli
    sres = np.load('Data/name_isabel_sim_res_palm_LIF.npy', allow_pickle=True)

    # Top panel - EPS reference (rasterised letter image).
    ax = plt.subplot(4, 1, 1)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.text(0, HEIGHT_MM + 2, 'Name on Palm - LIF', fontsize=12)
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

    out = 'saved_figs/name_isabel_population_palm_LIF.png'
    plt.savefig(out, bbox_inches='tight', dpi=300)
    print('Saved figure to', out)


print_name_raster_palm_lif()
plt.show()
