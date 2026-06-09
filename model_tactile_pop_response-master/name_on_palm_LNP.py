# -*- coding: utf-8 -*-
"""
name_on_skin.py - scan "ISABEL" across artificial skin and produce a Fig.8b-
style spike raster per afferent type.

This is the primary deliverable for Problem 2: "simulate your name being
pressed onto artificial skin and assess how accurately it can be
reconstructed from mechanoreceptor population activity" (from the brief).

Pipeline (mirrors replicate_fig8_letters_lnp.py exactly, just with the ISABEL stim):
  1. Generate an image of the word "ISABEL" at WIDTH_MM x HEIGHT_MM.
  2. Convert to an EEPS via the standard image-processing pipeline.
  3. Build the same multi-row scan trajectory used by Fig.8ab - many
     y-shifted rows scanned left-to-right at constant speed and force.
  4. For each (afferent, scan_row) pair, run LNP and record the central
     receptor's Va trace.
  5. Plot the result as: top panel = EPS reference; below = SA1/RA1/PC
     spike rasters with probe x on the horizontal axis and scan-row y
     (Distance) on the vertical.

Outputs:
  saved_figs/letters_ISABEL.jpg              - source image (auto-generated)
  saved_figs/name_isabel_population_palm.png      - Fig.8b-style raster
  Data/name_isabel_sim_res_palm.npy               - per-row Va traces
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import Receptors as rslib
import simset as mysim
import img_to_eqstimuli as imeqst
import Lnp as lnplib


# ---------------------------------------------------------------------------
# Stimulus / scan parameters - match Fig.8ab structure.
# ---------------------------------------------------------------------------
WIDTH_MM   = 90.0      # physical width of the ISABEL image on the skin
HEIGHT_MM  = 12.0      # physical height (vertical extent of the letters)
SHIFT      = 0.2       # mm - vertical spacing between scan rows
SPEED      = 20        # mm/s - horizontal scan velocity
PF         = 0.35      # N - constant pressing force during the scan
SIMDT      = 0.001     # s - 1 ms timestep
SIMT       = WIDTH_MM / SPEED   # one sweep = WIDTH_MM/SPEED seconds
DOTH       = 1         # EEPS height scale (1 = unchanged binarised heights)

LETTER_IMG = 'saved_figs/letters_ISABEL.jpg'

# Per-afferent LNP parameters now live as PER_TYPE_DEFAULTS in Lnp.py.
# LnpReceptors picks them up automatically from base.Ttype; override here only
# if you want this script to differ from the canonical set.
LNP_SEED = 42


# ---------------------------------------------------------------------------
# Step 1 - make (or load) the "ISABEL" source image
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
# Build the EEPS with the PALM ROI, not the fingertip ROI. The function
# pads the letter image by the ROI's bounding-box dimensions on each side;
# if we pad with fingertip dimensions while the sensor expects palm-sized
# contact slices, the simulator slices past the EEPS and we get truncated
# SC -> shape mismatch in _compute_Uc -> IndexError.
res_buf, eq_stimuli, eeps, eps = imeqst.constructing_equivalent_probe_stimuli_from_image(
    img, WIDTH_MM, HEIGHT_MM, mysim.plamroi
)
Aeeps = eeps
Aeeps[1] = Aeeps[1] * DOTH


# ---------------------------------------------------------------------------
# Step 3 - build receptor populations
# ---------------------------------------------------------------------------
Ttype_buf = ['SA1', 'RA1', 'PC']
tsensors  = []
pbuf = np.load('Data/loc_pos_buf_plam.npy', allow_pickle=True)

# Clip receptor positions a hair inside the ROI bounding box. set_population
# computes pixel indices via uint16((r_pos - roi.min()) / Wc * Nrc), so a
# receptor sitting exactly on roi.max() gets index Nrc - one past the last
# valid index, which later triggers an IndexError inside _compute_Uc.
# The 1e-3 mm buffer is well below the simulator's spatial resolution
# (Dbp = 0.2 mm) so it doesn't visibly move receptors.
# Use a larger buffer than 1e-3 mm - float64 -> uint16 rounding can land at
# Nrc exactly when the receptor is within ~1e-5 mm of roi.max(). Half a
# probe-pixel (Dbp/2 = 0.1 mm) is well below visual resolution but gives
# generous headroom for any floating-point accumulation in OEs.
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
# Step 5 - LNP scan, central-receptor recording per scan row
# ---------------------------------------------------------------------------
print('Scanning ISABEL: {} types x {} rows...'.format(
    len(Ttype_buf), len(ips)))
sim_res = []
palm_center = np.array([
    (mysim.plamroi[:,0].max() + mysim.plamroi[:,0].min()) / 2.0,
    (mysim.plamroi[:,1].max() + mysim.plamroi[:,1].min()) / 2.0,
]).reshape(1,2)

for ch, name in enumerate(Ttype_buf):
    lnp = lnplib.LnpReceptors(tsensors[ch],
                              rng_seed=LNP_SEED + ch)
    per_row = []
    for row in range(len(ips)):
        lnp.population_simulate(EEQS=Aeeps,
                                Ips=[ips[row], 'Pressure'],
                                noise=0, disinf=False)
        sel = tsensors[ch].points_mapping_entrys(palm_center)[0]
        per_row.append(np.array(tsensors[ch].Va[sel, :]))
    sim_res.append(per_row)
    print('  {} done'.format(name))

np.save('Data/name_isabel_sim_res_palm.npy', sim_res)


# ---------------------------------------------------------------------------
# Step 6 - plot Fig.8b-style raster: EPS reference + SA1/RA1/PC rasters
# ---------------------------------------------------------------------------
def print_name_raster():
    plt.figure(figsize=(7, 4 * 0.7))
    plt.subplots_adjust(hspace=0.2)

    buf  = eq_stimuli
    sres = np.load('Data/name_isabel_sim_res_palm.npy', allow_pickle=True)

    # Top panel - EPS reference. Each point is an active probe pin in the
    # rasterised letter image (column 5 = height in metres, scaled to mm).
    ax = plt.subplot(4, 1, 1)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.text(0, HEIGHT_MM + 2, 'Name on Palm - LNP', fontsize=12)
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

    out = 'saved_figs/name_isabel_population_palm.png'
    plt.savefig(out, bbox_inches='tight', dpi=300)
    print('Saved figure to', out)


print_name_raster()
plt.show()
