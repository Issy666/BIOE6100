# -*- coding: utf-8 -*-
"""
replicate_fig8_letters_lnp.py - produce a Fig.8b-style spike raster of
"ISA" letters scanned across the fingertip, using the LNP spike generator
instead of LIF. Mirrors Figs_from_Ouyang/Fig.8ab_Form_letters_scaning_sim
_single_repeat.py with the same stimulus and scan parameters.

Approach:
  - Same stimulus image (letters_120-12.jpg), same scan parameters
    (width=120, height=12, speed=20, pf=0.35) as Fig.8ab.
  - Same scan layout: many y-shifted rows, central receptor recorded on each.
  - Spike generation uses Lnp.LnpReceptors instead of
    tactile_receptors.population_simulate.
  - Final figure has the same structure as saved_figs/letter_spking.png:
    an EPS reference panel on top, then SA1 / RA1 / PC raster panels with
    probe-x on the horizontal axis and scan-row y (Distance) on the vertical.

Output: saved_figs/letter_spking_lnp.png

@author: Isabel Barton
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import Receptors as rslib
import simset as mysim
import img_to_eqstimuli as imeqst
import Lnp as lnplib


# ----------------------------------------------------------------------------
# Stimulus + scan parameters (verbatim from Fig.8ab so the rasters are
# directly comparable to saved_figs/letter_spking.png).
# ----------------------------------------------------------------------------
width   = 120
height  = 12
shift   = 0.2
speed   = 20
pf      = 0.35
simT    = width / speed
simdt   = 0.001
doth    = 1

# LNP knobs - per-afferent type so SA1 can use a different temporal filter
# from RA1 / PC. See model_param.md section 8 and lnp_explained.md §6.
#
# SA1 edge sensitivity (the change we just made):
#   - filter_kind='bandpass' makes the temporal filter act like a derivative
#     (peak when Uc is changing fastest = at letter edges).
#   - tau=0.010 s (fast) is matched to the ~10 ms edge transition time at
#     this scan speed (Dbp / speed = 0.2 mm / 20 mm/s).
#   - tau_slow=0.060 s gives a clear bandpass shape - the slow term cancels
#     the sustained part of the response so only edge transients survive.
#   - nonlinearity='relu' eliminates the diffuse background firing that
#     softplus leaks; spikes only happen on positive deflections (= edges).
#   - gain bumped to compensate for the smaller peak of the DoE kernel vs
#     the area-1 lowpass kernel.
LNP_PARAMS = {
    # SA1: bandpass + full-wave rectifier ('abs') => spikes on BOTH leading
    # and trailing edges of each letter. The bandpass output swings positive
    # on rising Uc (letter onset) and negative on falling Uc (letter offset);
    # 'abs' converts both swings into firing rate.
    'SA1': dict(tau=0.060, gain=300,  nonlinearity='relu',
                filter_kind='lowpass'),
    'RA1': dict(tau=0.010, gain=10,  nonlinearity='abs',
                filter_kind='bandpass', tau_slow=0.040),
    'PC':  dict(tau=0.003, gain=100,  nonlinearity='sigmoid',
                filter_kind='gabor')
}
LNP_SEED = 42


# ----------------------------------------------------------------------------
# Build the three afferent populations on the fingertip.
# ----------------------------------------------------------------------------
Ttype_buf = ['SA1', 'RA1', 'PC']
tsensors  = []
pbuf = np.load('Data/loc_pos_buf_fingertip.npy', allow_pickle=True)
for tp in range(len(Ttype_buf)):
    s = rslib.tactile_receptors(Ttype=Ttype_buf[tp])
    s.set_population(pbuf[tp][0], pbuf[tp][1],
                     simTime=simT, sample_rate=1/simdt,
                     Density=pbuf[tp][2], roi=mysim.fingertiproi)
    tsensors.append(s)


# ----------------------------------------------------------------------------
# Reuse Fig.8's cached EEPS if available; otherwise regenerate from the JPEG.
# ----------------------------------------------------------------------------
#cache_path = 'Data/forms_letters.npy'

cache_path = 'Data/forms_letters_isa.npy'
if os.path.exists(cache_path):
    bundle = np.load(cache_path, allow_pickle=True)
    Aeeps  = bundle[2]
else:
    #img1 = Image.open('saved_figs/letters_120-12.jpg')
    img1 = Image.open('saved_figs/letters_ISA.jpg')
    buf1, eq_stimuli, Aeeps, eps = imeqst.constructing_equivalent_probe_stimuli_from_image(
        img1, width, height, mysim.fingertiproi
    )
    np.save(cache_path,
            np.array([buf1, eq_stimuli, Aeeps, eps], dtype=object))
Aeeps[1] = Aeeps[1] * doth   # height-scale per ref. [35] (doth=1 is a no-op)


# ----------------------------------------------------------------------------
# Generate the full multi-row scan trajectory (same as Fig.8ab).
# Each entry of `ips` is a (T,3) array [x_mm, y_mm, force] for one scan row.
# ----------------------------------------------------------------------------
PF  = pf * np.ones(int(simT / simdt))
ips = imeqst.img_stimuli_scaning_with_uniformal_speed(
    simdt, simT, PF, speed,
    0, width,
    0, height,
    shift,
)


# ----------------------------------------------------------------------------
# Run the LNP simulation for every (afferent, scan-row) pair.
# Mirrors the structure of the LIF loop in Fig.8ab so that downstream
# plotting code can consume the result identically.
# ----------------------------------------------------------------------------
print('Running LNP simulation: {} types x {} rows'.format(
    len(Ttype_buf), len(ips)))
simulation_res = []
for tp in range(len(Ttype_buf)):
    # One LnpReceptors per type, configured from LNP_PARAMS so SA1, RA1, PC
    # can each use their own temporal filter / nonlinearity / gain.
    # Reusing the same LnpReceptors across scan rows keeps the seeded RNG
    # state consistent so spike streams are reproducible end-to-end.
    p = LNP_PARAMS[Ttype_buf[tp]]
    lnp = lnplib.LnpReceptors(tsensors[tp], rng_seed=LNP_SEED + tp, **p)
    per_row = []
    for row in range(len(ips)):
        # Drop-in: same EEQS/Ips/noise arguments as the LIF call.
        lnp.population_simulate(EEQS=Aeeps,
                                Ips=[ips[row], 'Pressure'],
                                noise=0,
                                disinf=False)
        # Same recording rule as Fig.8ab: central receptor (closest to (0,0)).
        sel = tsensors[tp].points_mapping_entrys(np.array([[0, 0]]))[0]
        per_row.append(np.array(tsensors[tp].Va[sel, :]))
    simulation_res.append(per_row)
    print('  {} done - {} rows'.format(Ttype_buf[tp], len(per_row)))

np.save('Data/letters_simulation_res_single_repeat_lnp.npy', simulation_res)


# ----------------------------------------------------------------------------
# Replicate the print_letter_spiking_trians plot from Fig.8ab.
# ----------------------------------------------------------------------------
def print_letter_spiking_trians_lnp():
    """
    Same layout as Fig.8ab's print_letter_spiking_trians but reads the LNP
    results. Top panel: EPS reference. Below: one raster panel per afferent
    type with probe x on the horizontal axis and scan-row y on the vertical.
    """
    plt.figure(figsize=(6, 4 * 0.7))
    plt.subplots_adjust(hspace=0.2)

    #buf  = np.load('Data/forms_letters.npy', allow_pickle=True)[1]
    buf  = np.load('Data/forms_letters_isa.npy', allow_pickle=True)[1]
    
    sres = np.load('Data/letters_simulation_res_single_repeat_lnp.npy',
                   allow_pickle=True)

    # EPS panel
    ax = plt.subplot(4, 1, 1)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.text(0, 14, '(b) LNP', fontsize=14)
    ax.scatter(buf[:, 0], buf[:, 1], s=0.01,
               c=1e3 * buf[:, 5] * doth,
               cmap=plt.cm.Greys, vmin=0, vmax=1)
    plt.xticks(np.arange(0, width + width/10, width/10), fontsize=6)
    plt.yticks([0, height/2, height], fontsize=7)
    ax1 = ax.twinx()
    ax1.spines['top'].set_color('None')
    ax1.spines['right'].set_color('None')
    plt.yticks([])
    plt.ylabel('EPS', fontsize=8)

    # Map scan-row index -> y-position on the vertical axis. Row 0 sits at
    # the top (height) and the last row at the bottom (0), matching Fig.8ab.
    num = int(height / shift)
    sel_points = np.vstack([0 * np.ones(num),
                            np.linspace(height, 0, num)]).T

    # One panel per afferent type, identical styling to Fig.8ab so the LNP
    # figure can be compared side-by-side with saved_figs/letter_spking.png.
    for ch in range(3):
        ax1 = plt.subplot(4, 1, ch + 2, sharex=ax)
        ax1.spines['top'].set_color('None')
        ax1.spines['right'].set_color('None')
        # In case sres has more/fewer rows than `num` (e.g. due to off-by-one
        # in img_stimuli_scaning_with_uniformal_speed), iterate over the
        # smaller count to avoid IndexError.
        rows_to_plot = min(num, len(sres[ch]))
        for i in range(rows_to_plot):
            spike_t = simdt * np.where(sres[ch][i] == 0.04)[0]
            plt.scatter(spike_t * speed,
                        sel_points[i, 1] * np.ones(len(spike_t)),
                        c=mysim.colors[ch], marker='.', s=0.01)
        plt.xticks(np.arange(0, width + width/10, width/10), fontsize=7)
        plt.yticks([0, 4, 8, 12], fontsize=7)
        if ch == 2: plt.xlabel('Position [mm]', fontsize=10)
        if ch == 1: plt.ylabel('Distance [mm]', fontsize=10)
        ax2 = ax1.twinx()
        ax2.spines['top'].set_color('None')
        ax2.spines['right'].set_color('None')
        plt.yticks([])
        plt.ylabel(Ttype_buf[ch], fontsize=8)
    #out_path = 'saved_figs/letter_spking_lnp.png'
    out_path = 'saved_figs/letter_spking_lnp_isa.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print('Saved LNP raster to', out_path)


print_letter_spiking_trians_lnp()
plt.show()
