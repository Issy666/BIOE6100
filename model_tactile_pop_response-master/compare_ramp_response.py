# -*- coding: utf-8 -*-
"""
compare_ramp_response.py - compare LIF and LNP responses to a trapezoidal
ramp-and-hold indentation, for each of SA1, RA1, PC.

The ramp-and-hold is a textbook stimulus for separating afferent classes
(see Saal et al. 2017):
  * SA1 should fire during BOTH ramp-up and hold (sustained).
  * RA1 should fire at onset AND offset, but be silent during hold.
  * PC should respond weakly to slow ramps (no high-frequency content).

What this script does:
  1. Builds a flat "skin surface" EEPS (uniform 1 mm raised everywhere).
  2. Holds the probe stationary at (0, 0) on the fingertip.
  3. Drives a trapezoidal force trajectory:
       100 ms ramp-up -> 500 ms hold -> 100 ms ramp-down -> 300 ms zero.
  4. Repeats the trial N_TRIALS times for each of LIF and LNP, recording
     spikes from the central receptor.
  5. Builds PSTHs (peri-stimulus time histograms), computes mean firing
     rate and Fano factor per stimulus phase, and produces a comparison
     figure.

Outputs:
  saved_figs/compare_ramp_psth.png       - PSTH comparison figure
  Data/cmp_ramp_metrics.npy              - metrics dict (rates, Fanos)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

import Receptors as rslib
import simset as mysim
import Lnp as lnplib


# ---------------------------------------------------------------------------
# Stimulus + simulation parameters
# ---------------------------------------------------------------------------
SIMDT          = 0.001        # s - 1 ms timestep / 1 kHz sample rate
RAMP_UP_DUR    = 0.100        # s
HOLD_DUR       = 0.500        # s
RAMP_DOWN_DUR  = 0.100        # s
POST_DUR       = 0.300        # s - silent tail so offset response is visible
PEAK_FORCE     = 0.5          # N - peak force during the hold phase
N_TRIALS       = 30           # trials per model for PSTH / Fano factor
SURFACE_HEIGHT = 1.0          # mm - uniform raised surface under the probe
EEPS_SIZE_PX   = 200          # 200 px @ Dbp=0.2 mm = 40 mm - bigger than fingertip

TRIAL_DUR      = RAMP_UP_DUR + HOLD_DUR + RAMP_DOWN_DUR + POST_DUR
T              = int(TRIAL_DUR / SIMDT)

# Phase boundaries in timestep indices (for metric windows).
RAMP_UP_END    = int(RAMP_UP_DUR / SIMDT)
HOLD_END       = RAMP_UP_END + int(HOLD_DUR / SIMDT)
RAMP_DOWN_END  = HOLD_END + int(RAMP_DOWN_DUR / SIMDT)
POST_END       = T
PHASES = {
    'onset':  (0,             RAMP_UP_END),
    'hold':   (RAMP_UP_END,   HOLD_END),
    'offset': (HOLD_END,      RAMP_DOWN_END),
    'silent': (RAMP_DOWN_END, POST_END),
}

# LNP per-type knobs - same as the values you've been iterating in
# replicate_fig8_letters_lnp.py. Edit here if you want to test a different tuning.
LNP_PARAMS = {
    'SA1': dict(tau=0.005, tau_slow=0.015, tau_sustained=0.07, mix=0.85, gain=450, nonlinearity='relu',
                filter_kind='parallel'),
    'RA1': dict(tau=0.005, gain=13, nonlinearity='abs',
                filter_kind='bandpass', tau_slow=0.010),
    'PC':  dict(tau=0.001, gain=7,  nonlinearity='abs',
                filter_kind='resonant', tau_slow=0.002)
}

LNP_BASE_SEED = 42


# ---------------------------------------------------------------------------
# Build the force trajectory and the stationary probe path
# ---------------------------------------------------------------------------
def build_trajectory():
    """Return ips: (T, 3) array of [x_mm, y_mm, force] per timestep."""
    force = np.zeros(T)
    force[:RAMP_UP_END]                  = np.linspace(0, PEAK_FORCE, RAMP_UP_END)
    force[RAMP_UP_END:HOLD_END]          = PEAK_FORCE
    force[HOLD_END:RAMP_DOWN_END]        = np.linspace(PEAK_FORCE, 0,
                                                       RAMP_DOWN_END - HOLD_END)
    # force is already zero from RAMP_DOWN_END onwards
    ips = np.zeros((T, 3))
    ips[:, 0] = 0.0   # x_mm - probe stationary at fingertip centre
    ips[:, 1] = 0.0   # y_mm
    ips[:, 2] = force
    return ips, force


def build_uniform_eeps():
    """Flat raised surface so every receptor in contact sees the same height."""
    eepsimg = np.ones((EEPS_SIZE_PX, EEPS_SIZE_PX)) * SURFACE_HEIGHT
    # population_simulate / LnpReceptors read EEQS[1] for the image. The first
    # element (pimageinf metadata) isn't used by the LIF/LNP simulate paths
    # for the ramp test, so a None placeholder is fine.
    return [None, eepsimg]


# ---------------------------------------------------------------------------
# Build receptor populations (one tactile_receptors per afferent type)
# ---------------------------------------------------------------------------
def build_populations():
    Ttype_buf = ['SA1', 'RA1', 'PC']
    tsensors  = []
    pbuf = np.load('Data/loc_pos_buf_fingertip.npy', allow_pickle=True)
    for tp in range(len(Ttype_buf)):
        s = rslib.tactile_receptors(Ttype=Ttype_buf[tp])
        s.set_population(pbuf[tp][0], pbuf[tp][1],
                         simTime=TRIAL_DUR, sample_rate=1/SIMDT,
                         Density=pbuf[tp][2], roi=mysim.fingertiproi)
        tsensors.append(s)
    return Ttype_buf, tsensors


# ---------------------------------------------------------------------------
# Per-trial runners
# ---------------------------------------------------------------------------
def run_lif_trials(sensor, Aeeps, ips, n_trials):
    """
    Run the LIF model N times. Stochasticity comes from the membrane noise
    VN inside population_simulate, which is regenerated each call.

    Returns (n_trials, T) boolean array - True where the central receptor
    spiked on each trial.
    """
    spikes = np.zeros((n_trials, T), dtype=bool)
    for trial in range(n_trials):
        sensor.population_simulate(EEQS=Aeeps, Ips=[ips, 'Pressure'],
                                   noise=0, disinf=False)
        sel = sensor.points_mapping_entrys(np.array([[0, 0]]))[0]
        spikes[trial, :] = (sensor.Va[sel, :] == 0.04)
    return spikes


def run_lnp_trials(sensor, Aeeps, ips, lnp_params, n_trials, base_seed):
    """
    Run the LNP model N times with a different RNG seed per trial. The
    Poisson sampler is the only stochastic stage, so the seed change gives
    independent spike realisations.
    """
    spikes = np.zeros((n_trials, T), dtype=bool)
    for trial in range(n_trials):
        lnp = lnplib.LnpReceptors(sensor,
                                  rng_seed=base_seed + trial,
                                  **lnp_params)
        lnp.population_simulate(EEQS=Aeeps, Ips=[ips, 'Pressure'],
                                noise=0, disinf=False)
        sel = sensor.points_mapping_entrys(np.array([[0, 0]]))[0]
        spikes[trial, :] = (sensor.Va[sel, :] == 0.04)
    return spikes


# ---------------------------------------------------------------------------
# Metrics: PSTH, mean rate per phase, Fano factor per phase
# ---------------------------------------------------------------------------
def psth(spikes, bin_ms=5):
    """
    Average spike count per bin across trials, converted to Hz.

    Args:
        spikes : (n_trials, T) boolean array
        bin_ms : bin width in ms
    Returns:
        bin_centres_s : (n_bins,) array of bin-centre times in seconds
        rate_hz       : (n_bins,) array of mean firing rate in Hz
    """
    bin_size = int(bin_ms / 1000 / SIMDT)
    n_bins   = T // bin_size
    trimmed  = spikes[:, :n_bins * bin_size]
    binned   = trimmed.reshape(spikes.shape[0], n_bins, bin_size).sum(axis=2)
    rate_hz  = binned.mean(axis=0) / (bin_ms / 1000.0)
    bin_centres_s = (np.arange(n_bins) + 0.5) * (bin_ms / 1000.0)
    return bin_centres_s, rate_hz


def phase_rate(spikes, phase):
    """Mean firing rate (Hz) over the phase window, averaged across trials."""
    a, b = phase
    if b == a:
        return 0.0
    counts = spikes[:, a:b].sum(axis=1)
    duration = (b - a) * SIMDT
    return counts.mean() / duration


def phase_fano(spikes, phase):
    """Fano factor (var/mean) of spike counts across trials in the phase."""
    a, b = phase
    if b == a:
        return float('nan')
    counts = spikes[:, a:b].sum(axis=1)
    m = counts.mean()
    if m == 0:
        return float('nan')
    return counts.var() / m


# ---------------------------------------------------------------------------
# Main: simulate + plot + tabulate
# ---------------------------------------------------------------------------
def main():
    ips, force_trace = build_trajectory()
    Aeeps            = build_uniform_eeps()
    Ttype_buf, tsensors = build_populations()

    # --- run simulations -----------------------------------------------------
    results = {}    # results[Ttype][model] -> (n_trials, T) boolean spike array
    for tp, name in enumerate(Ttype_buf):
        print('Running {}... '.format(name), end='', flush=True)
        lif_spikes = run_lif_trials(tsensors[tp], Aeeps, ips, N_TRIALS)
        print('LIF done... ', end='', flush=True)
        lnp_spikes = run_lnp_trials(tsensors[tp], Aeeps, ips,
                                    LNP_PARAMS[name], N_TRIALS,
                                    base_seed=LNP_BASE_SEED + 1000 * tp)
        print('LNP done.')
        results[name] = {'lif': lif_spikes, 'lnp': lnp_spikes}

    # --- per-phase metrics ---------------------------------------------------
    metrics = {}
    for name in Ttype_buf:
        metrics[name] = {}
        for model_key in ('lif', 'lnp'):
            spikes = results[name][model_key]
            metrics[name][model_key] = {
                phase_name: {
                    'rate_hz': phase_rate(spikes, phase_window),
                    'fano':    phase_fano(spikes, phase_window),
                }
                for phase_name, phase_window in PHASES.items()
            }
    np.save('Data/cmp_ramp_metrics.npy',
            np.array([metrics, PHASES], dtype=object))

    # --- pretty-print summary table -----------------------------------------
    print('\nPer-phase mean firing rate (Hz) and Fano factor:')
    header = '{:>4} {:>4}  ' + '  '.join(['{:>20}'] * len(PHASES))
    print(header.format('type', 'mdl', *PHASES.keys()))
    for name in Ttype_buf:
        for model_key in ('lif', 'lnp'):
            row = '{:>4} {:>4}  '.format(name, model_key)
            cells = []
            for phase_name in PHASES.keys():
                m = metrics[name][model_key][phase_name]
                cells.append('{:6.1f} Hz  F={:5.2f}'.format(
                    m['rate_hz'], m['fano']))
            print(row + '  '.join(cells))

    # --- plot ----------------------------------------------------------------
    # Layout: 2 rows (LIF/LNP) x 3 cols (SA1/RA1/PC).
    models = ['lif', 'lnp']
    fig, axes = plt.subplots(2, 3, figsize=(14, 3.85), sharex=True)
    t_ms = np.arange(T) * SIMDT * 1000.0
    bin_ms = 5

    for row, model_key in enumerate(models):
        for tp, name in enumerate(Ttype_buf):
            ax = axes[row, tp]
            spikes = results[name][model_key]

            # Stimulus envelope (force) on a secondary axis as a backdrop.
            ax_f = ax.twinx()
            ax_f.plot(t_ms, force_trace, color='lightgray', lw=1.0, zorder=1)
            ax_f.set_ylim(0, PEAK_FORCE * 1.2)
            ax_f.set_yticks([])
            ax_f.spines['top'].set_color('None')
            ax_f.spines['right'].set_color('None')

            # Trial raster - each trial is one row of ticks.
            for trial in range(N_TRIALS):
                spike_ms = t_ms[spikes[trial, :]]
                ax.scatter(spike_ms,
                           np.full_like(spike_ms, trial),
                           s=2.5, c=mysim.colors[tp], marker='|')

            # PSTH overlaid on top of the raster (rescaled to fit the raster
            # row range so both are visible).
            bc_s, rate = psth(spikes, bin_ms=bin_ms)
            psth_scaled = (rate / max(rate.max(), 1e-9)) * (N_TRIALS - 1)
            ax.plot(bc_s * 1000.0, psth_scaled, color='k', lw=1.0, zorder=3)

            # Phase boundary lines.
            for boundary in (RAMP_UP_END, HOLD_END, RAMP_DOWN_END):
                ax.axvline(boundary * SIMDT * 1000.0,
                           color='red', lw=0.4, alpha=0.4)

            # Per-phase mean firing rate labels, positioned at each phase's
            # x-midpoint at the very top of the axes. White bbox so they
            # don't blur into the PSTH peak.
            phase_mids_ms = {
                'onset':  (RAMP_UP_END / 2.0)               * SIMDT * 1000.0,
                'hold':   ((RAMP_UP_END + HOLD_END) / 2.0)  * SIMDT * 1000.0,
                'offset': ((HOLD_END + RAMP_DOWN_END) / 2.0) * SIMDT * 1000.0,
            }
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            for phase_name, x_ms in phase_mids_ms.items():
                r = metrics[name][model_key][phase_name]['rate_hz']
                ax.text(x_ms, 1.0, '{:.0f} Hz'.format(r),
                        transform=trans, ha='center', va='top',
                        fontsize=6.5, color='#222',
                        bbox=dict(facecolor='white', edgecolor='none',
                                  boxstyle='round,pad=0.18', alpha=0.85))

            ax.set_ylim(-1, N_TRIALS + 1)
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=7)
            ax.spines['top'].set_color('None')
            ax.spines['right'].set_color('None')

            if row == 0:
                ax.set_title(name, fontsize=10, pad=2)
            if tp == 0:
                ax.set_ylabel('{}\n{} trials'.format(model_key.upper(),
                                                    N_TRIALS),
                              fontsize=9)
            if row == len(models) - 1:
                ax.set_xlabel('Time [ms]', fontsize=8)

    fig.suptitle(
        'Ramp-and-hold response: LIF vs LNP  ({} trials, peak {} N)'.format(
            N_TRIALS, PEAK_FORCE),
        fontsize=10, y=0.995)
    # tight_layout reclaims margin without changing the (14, 3.3) figsize
    # aspect, so the saved PNG keeps the ~4.2:1 ratio panel C expects.
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = 'saved_figs/compare_ramp_psth.png'
    # No bbox_inches='tight': that would re-trim the canvas and change the
    # saved aspect ratio, defeating the figsize-matching above.
    fig.savefig(out_path, dpi=200)
    print('\nSaved figure to', out_path)
    plt.show()


if __name__ == '__main__':
    main()
