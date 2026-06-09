# -*- coding: utf-8 -*-
"""
quantify_rf_area.py - measure receptive-field AREA (mm²) for each afferent
type using both LIF and LNP, in the style of Saal et al. 2017 Fig. 3C.

Methodology (follows Saal's "fixed amplitude relative to threshold" recipe):
  1. Single central receptor at (0, 0).
  2. Place a raised dot at each location on a 2-D grid of probe positions
     covering ±10 mm around the receptor.
  3. At each location, apply a step-wave indentation (200 um peak, same as
     Fig.6) and record the central receptor's firing rate.
  4. Define the RF as the set of locations where firing rate exceeds 1% of
     the afferent's maximum firing rate (`tsensor.maxfr`). This matches
     Fig.6's RF-area definition.
  5. RF area = (# supra-threshold locations) * (grid spacing)^2.

Saal et al. 2017 reports for the human glabrous skin:
  - SA1: ~10 mm²   (smallest)
  - RA1: ~15 mm²   (slightly larger)
  - PC : ~100 mm²+ (order of magnitude larger)

Outputs:
  saved_figs/quantify_rf_area.png    - 3x2 heatmaps + bar comparison
  Data/cmp_rf_area_metrics.npy       - per-afferent areas + raw firing rate maps
"""
import numpy as np
import matplotlib.pyplot as plt

import Receptors as rslib
import simset as mysim
import img_to_eqstimuli as imeqst
import Lnp as lnplib


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SIMDT     = 0.001
SIM_T     = 1                # s - same as Fig.6
INDENT    = 500e-6             # m - middle indent depth from Fig.6 indent_buf
DOT_RAD   = 0.25               # mm - dot radius (Fig.6 default)
DOT_H     = 0.85               # mm - dot height (Fig.6 default)

# RF boundary criterion. Two options:
#   'peak_fraction'  -> RF = where rate > THRESH_FRAC * peak_rate per receptor.
#                       This is the neurophysiological standard (FWHM-style).
#                       Robust to absolute firing-rate differences between
#                       LIF and LNP, and produces the ~10 / 15 / 100 mm² scale
#                       that Saal et al. 2017 report.
#   'maxfr_fraction' -> RF = where rate > THRESH_FRAC * tsensor.maxfr.
#                       This is the Fig.6 criterion. Very permissive - even a
#                       single spike crosses it, so RF gets inflated by any
#                       drive whatsoever (including noise + resistance-network
#                       leakage). Tends to saturate at the contact-window area.
THRESH_MODE = 'peak_fraction'
THRESH_FRAC = 0.1              # half-max for 'peak_fraction', 0.01 for Fig.6

# For PC: a slow step-wave has no energy at PC's 200-300 Hz tuning peak, so
# PC produces ~0 spikes regardless of probe location and the RF is undefined.
# Setting USE_VIBRATION_FOR_PC=True replaces PC's stimulus with a 250 Hz
# sinusoidal indentation (10 um amplitude) which drives PC properly.
USE_VIBRATION_FOR_PC = True
PC_VIB_FREQ_HZ       = 250.0
PC_VIB_AMP_M         = 10e-6   # 10 um - well above PC's submicrometer threshold

# Grid of probe locations around the central receptor at (0, 0).
GRID_EXTENT  = 10.0            # mm - half-width of the probed region (covers PC)
GRID_SPACING = 0.5             # mm - coarser = faster but rougher area estimate
xs = np.arange(-GRID_EXTENT, GRID_EXTENT + GRID_SPACING/2, GRID_SPACING)
ys = np.arange(-GRID_EXTENT, GRID_EXTENT + GRID_SPACING/2, GRID_SPACING)
xx, yy   = np.meshgrid(xs, ys)
all_locations = np.column_stack([xx.ravel(), yy.ravel()])   # (N_grid, 2)

# Keep only grid points that lie INSIDE the fingertip ROI polygon. Points
# outside the fingertip aren't skin and shouldn't count toward RF area.
# rslib.isPoisWithinPoly takes a closed polygon (first vertex repeated) and
# an array of test points; returns a boolean mask.
fingertip_poly = np.vstack([mysim.fingertiproi, mysim.fingertiproi[0, :]])
in_finger = rslib.isPoisWithinPoly(fingertip_poly, all_locations)
locations = all_locations[in_finger]
print('Grid: {} total locations, {} inside fingertip ({:.1f} mm² of skin)'
      .format(len(all_locations), len(locations),
              len(locations) * GRID_SPACING ** 2))

# Saal et al. 2017 reference values (Fig. 3C area data) - these are the
# observed RF areas of biological afferents at fixed-amplitude-relative-
# to-threshold stimulation in mm².
SAAL_RF_MM2 = {
    'SA1': 10.0,
    'RA1': 15.0,
    'PC':  100.0,
}

# LNP per-type knobs - copied from compare_ramp_response so the model
# behaviour is consistent across all comparison scripts.
LNP_PARAMS = {
    'SA1': dict(tau=0.005, tau_slow=0.015, tau_sustained=0.07, mix=0.85, gain=450, nonlinearity='relu',
                filter_kind='parallel'),
    'RA1': dict(tau=0.005, gain=13, nonlinearity='abs',
                filter_kind='bandpass', tau_slow=0.010),
    'PC':  dict(tau=0.001, gain=7,  nonlinearity='abs',
                filter_kind='resonant', tau_slow=0.002)
}
LNP_SEED = 42


# ---------------------------------------------------------------------------
# Build populations
# ---------------------------------------------------------------------------
Ttype_buf = ['SA1', 'RA1', 'PC']
tsensors  = []
pbuf = np.load('Data/loc_pos_buf_fingertip.npy', allow_pickle=True)
for ch in range(len(Ttype_buf)):
    s = rslib.tactile_receptors(Ttype=Ttype_buf[ch])
    s.set_population(pbuf[ch][0], pbuf[ch][1], simTime=SIM_T,
                     sample_rate=1/SIMDT, Density=pbuf[ch][2],
                     roi=mysim.fingertiproi)
    tsensors.append(s)

# Two stimulus traces: a slow ramp-and-hold (used for SA1/RA1, matches Fig.6)
# and a 250 Hz vibration (used for PC because slow ramps have no PC-relevant
# frequency content). Both are applied in 'Depth' mode.
stimulus_ramp = rslib.step_wave(
    tsensors[0].t, 0, 0.3,
    INDENT * 1e6 / 200,
    -INDENT * 1e6 / 200,
    INDENT
)
stimulus_vib = rslib.sin_wave(tsensors[0].t,
                              2 * np.pi * PC_VIB_FREQ_HZ,
                              PC_VIB_AMP_M)
# Per-afferent stimulus selection.
stim_per_type = {
    'SA1': stimulus_ramp,
    'RA1': stimulus_ramp,
    'PC':  stimulus_vib if USE_VIBRATION_FOR_PC else stimulus_ramp,
}


# ---------------------------------------------------------------------------
# Per-location runner
# ---------------------------------------------------------------------------
# IMPORTANT: We must pass the FULL grid of potential dot positions into
# constructing_probe_stimuli (with only one raised), not a single-dot pimage.
# constructing_probe_stimuli sizes its canvas (Wc, Hc) from the extent of
# the dot positions plus 2*rad - so a single dot would give a 0.5x0.5 mm
# canvas with the dot dead-centre, and the receptor at (0,0) would always
# see the dot at the same relative position regardless of probe_xy. Using
# the full grid makes the canvas span ±GRID_EXTENT, so a raised dot at
# locations[k] is offset from the canvas centre by exactly locations[k] -
# which is the receptor-relative position we want.
RADII_ALL   = np.ones((len(locations), 1)) * DOT_RAD


def build_eeps_with_raised_at(loc_idx):
    """
    Build a pimage that spans the full probe grid, with the dot at
    locations[loc_idx] raised (height DOT_H) and every other grid point
    flat (height 0). Returns the EEPS bundle and pimage.
    """
    heights = np.zeros((len(locations), 1))
    heights[loc_idx, 0] = DOT_H
    pimage, _ = mysim.constructing_probe_stimuli(
        np.hstack([locations, RADII_ALL, heights])
    )
    eeps = imeqst.constructing_equivalent_probe_stimuli_from_pimage(
        pimage[0], pimage[1], pimage[2], mysim.fingertiproi
    )
    return eeps, pimage


def run_one(sensor, lnp_kwargs, eeps_bundle, pimage, depth_trace):
    """Returns central-receptor firing rate (Hz) for one stimulus location."""
    DP = depth_trace.reshape(-1, 1)
    w  = pimage[1]
    h  = pimage[2]
    ips = [np.hstack([w/2 * np.ones((len(DP), 1)),
                      h/2 * np.ones((len(DP), 1)),
                      DP]), 'Depth']
    if lnp_kwargs is None:
        sensor.population_simulate(EEQS=eeps_bundle, Ips=ips,
                                   noise=0, disinf=False)
    else:
        lnp = lnplib.LnpReceptors(sensor, **lnp_kwargs)
        lnp.population_simulate(EEQS=eeps_bundle, Ips=ips,
                                noise=0, disinf=False)
    sel = sensor.points_mapping_entrys(np.array([[0, 0]]))[0]
    return np.sum(sensor.Va[sel, :] == 0.04) / SIM_T


# ---------------------------------------------------------------------------
# Sweep every (model, afferent, location)
# ---------------------------------------------------------------------------
n_loc   = len(locations)
results = {m: {n: np.zeros(n_loc) for n in Ttype_buf} for m in ('lif', 'lnp')}

print('Total simulations: {} locations x 3 afferents x 2 models = {}'.format(
    n_loc, n_loc * 3 * 2))
for i, loc in enumerate(locations):
    eeps_bundle, pimage = build_eeps_with_raised_at(i)
    if (i + 1) % 10 == 0 or i == n_loc - 1:
        print('  loc {}/{} ({:.1f}, {:.1f}) mm'.format(
            i + 1, n_loc, loc[0], loc[1]))
    for ch, name in enumerate(Ttype_buf):
        depth = stim_per_type[name]   # ramp for SA1/RA1, vibration for PC
        results['lif'][name][i] = run_one(
            tsensors[ch], None, eeps_bundle, pimage, depth)
        lnp_kw = dict(LNP_PARAMS[name],
                      rng_seed=LNP_SEED + ch * 10000 + i)
        results['lnp'][name][i] = run_one(
            tsensors[ch], lnp_kw, eeps_bundle, pimage, depth)


# ---------------------------------------------------------------------------
# Compute RF area. Two threshold modes selectable above (see THRESH_MODE):
#   'peak_fraction'  -> per-receptor adaptive threshold = THRESH_FRAC * peak
#                       firing rate observed across all probe locations.
#                       Boundary at half-max gives the FWHM-style RF size
#                       that Saal et al. 2017 quote.
#   'maxfr_fraction' -> THRESH_FRAC * tsensor.maxfr (Fig.6 criterion).
# Area = (# suprathreshold locations) * GRID_SPACING^2.
# ---------------------------------------------------------------------------
rf_area    = {m: {} for m in ('lif', 'lnp')}
thresholds = {m: {} for m in ('lif', 'lnp')}
for m in ('lif', 'lnp'):
    for ch, name in enumerate(Ttype_buf):
        rates = results[m][name]
        if THRESH_MODE == 'peak_fraction':
            peak = rates.max()
            threshold = THRESH_FRAC * peak if peak > 0 else float('inf')
        else:                                       # 'maxfr_fraction'
            threshold = THRESH_FRAC * tsensors[ch].maxfr
        thresholds[m][name] = threshold
        n_supra = int(np.sum(rates > threshold))
        rf_area[m][name] = n_supra * GRID_SPACING ** 2

print('\nRF area (mm²) - mode = {} (frac = {}):'.format(
    THRESH_MODE, THRESH_FRAC))
hdr = '{:>4}  {:>10}  {:>10}  {:>10}'.format(
    'type', 'LIF', 'LNP', 'Saal')
print(hdr)
for name in Ttype_buf:
    print('{:>4}  {:>10.1f}  {:>10.1f}  {:>10.1f}'.format(
        name, rf_area['lif'][name], rf_area['lnp'][name], SAAL_RF_MM2[name]))

np.save('Data/cmp_rf_area_metrics.npy',
        np.array([results, rf_area, SAAL_RF_MM2, locations], dtype=object))


# ---------------------------------------------------------------------------
# Plot: 3 rows (afferents) x 3 cols (LIF heatmap, LNP heatmap, area bars).
#
# `results` is sized by the IN-FINGERTIP locations only (locations was filtered
# with isPoisWithinPoly), so we can't directly reshape it to the full grid.
# Build a full-grid array of NaN and scatter the in-finger rates into the
# right slots; imshow renders NaN as transparent so out-of-fingertip cells
# don't show colour.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
extent = [xs.min(), xs.max(), ys.min(), ys.max()]
n_full = len(all_locations)            # full rectangular grid (pre-filter)

for ch, name in enumerate(Ttype_buf):
    for col, model_key in enumerate(('lif', 'lnp')):
        ax = axes[ch, col]
        # Build a full-grid image: NaN outside the fingertip, rate inside.
        full = np.full(n_full, np.nan)
        full[in_finger] = results[model_key][name]
        img  = full.reshape(len(ys), len(xs))
        im   = ax.imshow(img, extent=extent, origin='lower',
                         cmap='hot', aspect='equal')
        # Overlay the RF boundary at whichever threshold was actually used.
        thr = thresholds[model_key][name]
        if np.isfinite(thr) and (img > thr).any():
            ax.contour(xs, ys, img, levels=[thr],
                       colors='cyan', linewidths=1.0)
        ax.set_title('{} {}  (RF = {:.1f} mm²)'.format(
            name, model_key.upper(), rf_area[model_key][name]),
            fontsize=10)
        if col == 0: ax.set_ylabel('y [mm]')
        if ch == 2:  ax.set_xlabel('x [mm]')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Hz')

    # Right column: bar comparison
    ax = axes[ch, 2]
    bars = ['LIF', 'LNP', 'Saal']
    vals = [rf_area['lif'][name], rf_area['lnp'][name], SAAL_RF_MM2[name]]
    colors = ['black', mysim.colors[ch], 'gray']
    ax.bar(bars, vals, color=colors)
    ax.set_ylabel('RF area [mm²]')
    ax.set_title('{} - RF area comparison'.format(name), fontsize=10)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, '{:.1f}'.format(v),
                ha='center', fontsize=9)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')

fig.suptitle('Receptive-field area (mm²) - LIF vs LNP vs Saal et al. 2017 Fig. 3C',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out_path = 'saved_figs/quantify_rf_area.png'
fig.savefig(out_path, bbox_inches='tight', dpi=200)
print('\nSaved figure to', out_path)
plt.show()
