# -*- coding: utf-8 -*-
"""
replicate_fig7_dots_lnp.py - LNP reproduction of Ouyang 2019 Fig. 7
(Figs_from_Ouyang/Fig.7_Textures_dots_single_repeat.py is the original LIF
version that this script mirrors with the LNP spike generator).

Replicates the dotted-texture scanning experiment from the paper, but with
spike generation handled by Lnp.LnpReceptors instead of the LIF model in
tactile_receptors.population_simulate.

Pipeline:
  1. Load dot positions from Data/txtdata/texture_dots.txt and build the
     EPS / EEPS for a raised-dot texture surface.
  2. Build SA1 / RA1 / PC populations on the fingertip.
  3. Scan the texture at speed=60 mm/s with force pf=0.6 N (same as Fig.7).
  4. Spike generation: Lnp.LnpReceptors per afferent type with per-type
     tunings (SA1 uses bandpass + abs to spike on both leading and trailing
     dot edges; RA1 / PC use lowpass + softplus).
  5. Plot a 4-panel raster (EPS on top, then SA1, RA1, PC) and a 2-panel
     MIPS / MIDP analysis against the observed reference data.

Outputs:
  saved_figs/dots_spking_repeat_single_lnp.png   - raster
  saved_figs/Tdots_relevant_lnp.png              - MIPS/MIDP + R² panels
  Data/dots_single_repeat_simulation_res_lnp.npy - per-row Va traces
  Data/sim_Frate_Tdots_lnp.npy                   - predicted MIPS curves
  Data/sim_MIPD_Tdots_lnp.npy                    - predicted MIDP curves

@author: Isabel Barton
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import utils as alt
import Receptors as rslib
import simset as mysim
import img_to_eqstimuli as imeqst
import Lnp as lnplib


# ----------------------------------------------------------------------------
# Stimulus parameters - verbatim from Fig.7 so the LNP runs on the same
# dotted texture as the published LIF result.
# ----------------------------------------------------------------------------
shift  = 0.2   # mm - vertical spacing between adjacent scan rows
speed  = 60    # mm/s - horizontal scan velocity (3x faster than Fig.8)
pf     = 0.6   # N - constant pressing force (higher than Fig.8)
rad    = 0.5   # mm - radius of each dot in the EPS
doth   = 0.5   # mm - dot height above the surrounding surface
baseh  = 1     # mm - base height of the surface (added to every EPS pixel)
simdt  = 0.001 # s - simulation timestep (1 kHz)

# Build the dot-array stimulus: each row of S is (x_mm, y_mm, radius, height).
dotspos = np.loadtxt('Data/txtdata/texture_dots.txt')
S = np.hstack([dotspos,
               rad  * np.ones([len(dotspos), 1]),
               doth * np.ones([len(dotspos), 1])])

# constructing_probe_stimuli rasterises the dot list into a pimage and
# returns [[pimage, Wc, Hc], eq_stimuli]. Wc and Hc are derived from the
# dot positions plus a 2*rad margin.
pimg, _ = mysim.constructing_probe_stimuli(S)
pimage, width, height = pimg[0], pimg[1], pimg[2]

# constructing_equivalent_probe_stimuli_from_pimage returns
# [pimageinf, EEPS, eq_stimuli, EPS]. EEQS[1] (the padded EEPS image) is
# what population_simulate / LnpReceptors actually consume.
# Adding `baseh` to pimage gives every surface pixel a non-zero baseline
# height; the dots stick up by `doth` on top of that.
Aeeps = imeqst.constructing_equivalent_probe_stimuli_from_pimage(
    pimage + baseh, width, height, mysim.fingertiproi
)

simT = width / speed   # scan duration; matches Fig.7


# ----------------------------------------------------------------------------
# LNP per-type parameters. See lnp_explained.md §6.
#   - SA1: bandpass + 'abs' => spikes on BOTH leading and trailing dot edges.
#   - RA1, PC: lowpass + softplus => sustained-style response.
# ----------------------------------------------------------------------------
LNP_PARAMS = {
    'SA1': dict(tau=0.010, gain=900,  nonlinearity='abs',
                filter_kind='bandpass', tau_slow=0.030),
    'RA1': dict(tau=0.010, gain=300,  nonlinearity='softplus',
                filter_kind='lowpass'),
    'PC':  dict(tau=0.010, gain=300,  nonlinearity='softplus',
                filter_kind='lowpass'),
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
# Generate the multi-row scan trajectory. Each entry of `ips` is a (T,3)
# array of [x_mm, y_mm, force] for one constant-y scan row.
# ----------------------------------------------------------------------------
PF  = pf * np.ones(int(simT / simdt))
ips = imeqst.img_stimuli_scaning_with_uniformal_speed(
    simdt, simT, PF, speed,
    0, width,
    0, height,
    shift,
)


# ----------------------------------------------------------------------------
# Run the LNP scan for every (afferent, scan-row) pair. Same loop layout as
# Fig.7's commented-out LIF block; tsensors[tp].Va is overwritten by each
# LnpReceptors.population_simulate call.
# ----------------------------------------------------------------------------
res_path = 'Data/dots_single_repeat_simulation_res_lnp.npy'
print('Running LNP scan: {} types x {} rows'.format(len(Ttype_buf), len(ips)))
simulation_res = []
for tp in range(len(Ttype_buf)):
    p   = LNP_PARAMS[Ttype_buf[tp]]
    lnp = lnplib.LnpReceptors(tsensors[tp], rng_seed=LNP_SEED + tp, **p)
    per_row = []
    for row in range(len(ips)):
        lnp.population_simulate(EEQS=Aeeps,
                                Ips=[ips[row], 'Pressure'],
                                noise=0,
                                disinf=False)
        # Central-receptor recording (same site as Fig.7).
        sel = tsensors[tp].points_mapping_entrys(np.array([[0, 0]]))[0]
        per_row.append(np.array(tsensors[tp].Va[sel, :]))
    simulation_res.append(per_row)
    print('  {} done - {} rows'.format(Ttype_buf[tp], len(per_row)))
np.save(res_path, simulation_res)


# ----------------------------------------------------------------------------
# Plot 1 - spike raster (mirror of Fig.7 print_dots_spiking_trians).
# ----------------------------------------------------------------------------
def print_dots_spiking_trians_lnp():
    """
    EPS reference panel on top, then one raster panel per afferent type with
    probe x on the horizontal axis and scan-row y (Distance) on the vertical.
    """
    plt.figure(figsize=(7, 4 * 0.7))
    buf  = Aeeps[2]   # eq_stimuli: (M,6) probe-pin positions
    sres = np.load(res_path, allow_pickle=True)

    # EPS panel.
    ax = plt.subplot(4, 1, 1)
    plt.text(-10, height + 6, '(b) LNP', fontsize=14)
    ax.scatter(buf[:, 0], buf[:, 1], s=0.02,
               c=1e3 * buf[:, 5], cmap=plt.cm.Greys,
               vmin=baseh, vmax=baseh + 1)
    plt.yticks([0, height/2, height], fontsize=6)
    plt.xticks(np.arange(0, width + width/10, width/10), fontsize=8)
    ax.twinx()
    plt.yticks([])
    plt.ylabel('EPS', fontsize=8)

    # Scan-row -> Distance mapping. Row 0 at top (height), last at bottom (0).
    num = int(height / shift)
    sel_points = np.vstack([0 * np.ones(num),
                            np.linspace(height, 0, num)]).T

    for ch in range(3):
        ax1 = plt.subplot(4, 1, ch + 2, sharex=ax)
        plt.subplots_adjust(hspace=0.1)
        # Guard against off-by-one mismatch between `num` and len(sres[ch]).
        rows_to_plot = min(num, len(sres[ch]))
        for i in range(rows_to_plot):
            spike_t = simdt * np.where(sres[ch][i] == 0.04)[0]
            plt.scatter(spike_t * speed,
                        sel_points[i, 1] * np.ones(len(spike_t)),
                        c=mysim.colors[ch], s=0.02)
        plt.xticks(np.arange(0, width + width/10, width/10), fontsize=8)
        plt.yticks([0, height/2, height], fontsize=6)
        plt.xlabel('Position [mm]', fontsize=8)
        if ch == 1: plt.ylabel('Distance [mm]', fontsize=7)
        ax1.twinx()
        plt.yticks([])
        plt.ylabel(Ttype_buf[ch], fontsize=8)

    out_path = 'saved_figs/dots_spking_repeat_single_lnp.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print('Saved raster to', out_path)


# ----------------------------------------------------------------------------
# Plot 2 - MIPS / MIDP analysis (mirror of Fig.7 print_PulsePS_and_PulsePdots).
# At each dot spacing, count spikes inside a sliding window whose width is
# 3x the local spacing, and compute:
#   MIPS = spikes / (window duration in s) / (#scan rows in window)
#   MIDP = spikes / (#dots in window)
# Then overlay against the observed reference curves.
# ----------------------------------------------------------------------------
wd = [0, 20]  # mm - moving-window definition per ref. [43]

def print_PulsePS_and_PulsePdots_lnp():
    sres = np.load(res_path, allow_pickle=True)
    buf1 = np.loadtxt('Data/txtdata/ob_Frate_Tdots_RA1_SA1.txt')
    buf2 = np.loadtxt('Data/txtdata/ob_MIPD_Tdots_RA1_SA1.txt')
    obMIPS = [buf1[12:24, 1], buf1[0:12, 1],
              np.loadtxt('Data/txtdata/ob_Frate_Tdots_PC.txt')[:, 1]]
    obMIPD = [buf2[12:24, 1], buf2[0:12, 1],
              np.loadtxt('Data/txtdata/ob_MIPD_Tdots_PC.txt')[:, 1]]
    # Cache observed for the R² panel (overwrites Fig.7's caches; that's OK
    # because the underlying data files are unchanged).
    np.save('Data/ob_Frate_Tdots.npy', obMIPS)
    np.save('Data/ob_MIPD_Tdots.npy',  obMIPD)

    spaces    = buf1[0:12, 0]
    N         = len(spaces)
    positions = np.linspace(197, 11, N)        # mm - scan x at each spacing
    ptimes    = (positions / width) * simT     # s  - time the scan reaches it
    rsites    = np.arange(-wd[1]/2, wd[1]/2, shift)
    num       = len(rsites)

    sim_Frate_Tdots = []
    sim_MIPD_Tdots  = []
    for ch in range(3):
        FPS = np.zeros(len(ptimes))
        FPD = np.zeros(len(ptimes))
        for m in range(len(ptimes)):
            wd[0]    = spaces[m] * 3
            duration = wd[0] / width * simT
            st       = int(ptimes[m] / tsensors[ch].dt)
            sn       = int(duration / tsensors[ch].dt)
            dotsn    = (wd[0] * wd[1]) / spaces[m]**2
            tmp      = np.array(sres[ch])[:, st:st + sn]
            FPS[m]   = np.sum(tmp == 0.04) / duration / num
            FPD[m]   = np.sum(tmp == 0.04) / dotsn
        sim_Frate_Tdots.append(FPS)
        sim_MIPD_Tdots.append(FPD)

    np.save('Data/sim_Frate_Tdots_lnp.npy', sim_Frate_Tdots)
    np.save('Data/sim_MIPD_Tdots_lnp.npy',  sim_MIPD_Tdots)

    simbuf    = [sim_Frate_Tdots, sim_MIPD_Tdots]
    obbuf     = [obMIPS, obMIPD]
    suptitles = ['MIPS', 'MIDP']

    for sigt in range(2):
        ax = plt.subplot(2, 2, sigt + 1)
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        if sigt == 0: plt.text(-1, 270, '(c) LNP', fontsize=14)
        plt.title(suptitles[sigt], fontsize=8)
        for ch in range(3):
            plt.plot(spaces, simbuf[sigt][ch], mysim.colors[ch],
                     marker=mysim.markers[ch], markerfacecolor='none',
                     markersize=6, label=Ttype_buf[ch])
            plt.plot(spaces, obbuf[sigt][ch], 'gray',
                     marker=mysim.markers[ch], markerfacecolor='none',
                     markersize=6)
        plt.xticks([1, 2, 3, 4, 5, 6], fontsize=6)
        if sigt == 0:
            plt.yticks([0, 50, 100, 150, 200, 250], fontsize=6)
            plt.ylabel('Mean impulses per second', fontsize=8)
        else:
            plt.yticks([0, 100, 200, 300, 400, 500], fontsize=6)
            plt.ylabel('Mean impulses per dot', fontsize=8)
        plt.xlabel('Dot spacing [mm]', fontsize=8)
        plt.legend(fontsize=8, edgecolor='w')


# ----------------------------------------------------------------------------
# Plot 3 - predicted-vs-observed scatter with R² (mirror of Fig.7
# plot_prediction_relevance).
# ----------------------------------------------------------------------------
def plot_prediction_relevance_lnp():
    suptitles  = ['MIPS', 'MIDP']
    plotlabels = ['SA1, ', 'RA1, ', 'PC, ']
    ticks      = [[0, 50, 100, 150], [0, 100, 200, 300, 400]]
    obdata     = [np.load('Data/ob_Frate_Tdots.npy',     allow_pickle=True),
                  np.load('Data/ob_MIPD_Tdots.npy',      allow_pickle=True)]
    simdata    = [np.load('Data/sim_Frate_Tdots_lnp.npy', allow_pickle=True),
                  np.load('Data/sim_MIPD_Tdots_lnp.npy',  allow_pickle=True)]

    for sigt in [0, 1]:
        ax = plt.subplot(2, 2, sigt + 3)
        if sigt == 0: plt.text(-50, 160, '(d) LNP', fontsize=14)
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        plt.title(suptitles[sigt], fontsize=8)
        plt.xlabel('Predicted ' + suptitles[sigt], fontsize=8)
        plt.ylabel('Observed '  + suptitles[sigt], fontsize=8)
        for ch in [0, 1, 2]:
            x = simdata[sigt][ch][:]
            y = obdata[sigt][ch][:]
            r, pval = stats.pearsonr(y, x)
            print('{} {} Pearson r={:.3f}, p={:.3g}'.format(
                Ttype_buf[ch], suptitles[sigt], r, pval))
            fit = alt.curve_fit(x, y)
            R2  = "{0:.3f}".format(alt.R2(fit[2], y))
            plt.scatter(x, y, color='w',
                        edgecolors=mysim.colors[ch],
                        marker=mysim.markers[ch], s=15,
                        label=plotlabels[ch] + ' $\\mathrm{R}^{2}$=' + str(R2))
            plt.plot(fit[0], fit[1], '--', color=mysim.colors[ch])
        plt.xticks(ticks[sigt], fontsize=6)
        plt.yticks(ticks[sigt], fontsize=6)
        plt.legend(loc=2, fontsize=6, edgecolor='gray')


# ----------------------------------------------------------------------------
# Build the combined MIPS/MIDP + R² figure, then the raster figure.
# ----------------------------------------------------------------------------
plt.figure(figsize=(5.2, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.5)
print_PulsePS_and_PulsePdots_lnp()
plot_prediction_relevance_lnp()
plt.savefig('saved_figs/Tdots_relevant_lnp.png', bbox_inches='tight', dpi=300)
print('Saved analysis to saved_figs/Tdots_relevant_lnp.png')

print_dots_spiking_trians_lnp()

plt.show()
