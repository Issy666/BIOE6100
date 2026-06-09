# -*- coding: utf-8 -*-
"""
Lnp.py - Linear-Nonlinear-Poisson spike generator for tactile receptors.

Drop-in replacement for the LIF spike stage in
`Receptors.tactile_receptors.population_simulate`. The spatial pre-processing
(skin contact -> resistance network -> Uc) is reused from a base
tactile_receptors instance; only the temporal filter, nonlinearity, and spike
generation are replaced.

Cascade:
    drive(t)  = (kernel * Uc(t))               per receptor   (L)
    rate(t)   = phi(gain * drive(t))           firing rate Hz (N)
    spikes(t) ~ Bernoulli(rate(t) * dt)                       (P)

Output matches `tactile_receptors.population_simulate` so downstream code that
scans for `Va == 0.04` works unchanged.

@author: Isabel Barton
"""
import numpy as np
import time as timec
from scipy.signal import lfilter

import Receptors as rslib  # Dbp constant and butterworth_filter helper


# ----------------------------------------------------------------------------
# Module-level fallback defaults (used only when no PER_TYPE_DEFAULTS entry).
# ----------------------------------------------------------------------------
DEFAULT_TAU           = 0.030
DEFAULT_GAIN          = 200.0
DEFAULT_NONLINEARITY  = 'softplus'
DEFAULT_FILTER_KIND   = 'lowpass'
DEFAULT_TAU_SLOW      = 0.060
DEFAULT_TAU_SUSTAINED = 0.060
DEFAULT_MIX           = 0.5

# per-afferent settings. `LnpReceptors(base)` picks the matching row
# from `base.Ttype` automatically; pass explicit constructor args to override.
PER_TYPE_DEFAULTS = {
    'SA1': dict(tau=0.005, tau_slow=0.015, tau_sustained=0.07, mix=0.85,
                gain=450.0, nonlinearity='relu', filter_kind='parallel'),
    'RA1': dict(tau=0.005, tau_slow=0.010,
                gain=13.0,  nonlinearity='abs',  filter_kind='bandpass'),
    'PC':  dict(tau=0.001, tau_slow=0.002,
                gain=7.0,   nonlinearity='abs',  filter_kind='resonant'),
}

# Spike sentinel must match the LIF model so np.where(Va == 0.04) still works.
SPIKE_SENTINEL = 0.04
REST_POTENTIAL = -0.065


_AUTO = object()


# ----------------------------------------------------------------------------
# Temporal-filter kernels (L stage)
# ----------------------------------------------------------------------------
def exp_kernel(tau, dt, length=None):
    """Sum-normalised causal exponential. Lowpass / sustained branch."""
    if length is None:
        length = max(2, int(5 * tau / dt))
    t = np.arange(length) * dt
    k = np.exp(-t / tau)
    return k / k.sum()


def resonant_kernel(tau, dt, length=None, f0=250):
    """Damped 250 Hz sinusoid; peak-normalised, zero-DC. PC vibration tuning."""
    if length is None:
        length = max(2, int(5 * tau / dt))
    t = np.arange(length) * dt
    k = np.exp(-t / tau) * np.sin(2 * np.pi * f0 * t)
    k = k - k.mean()
    peak = np.abs(k).max()
    return k / peak if peak > 0 else k


def gabor_kernel(dt, f0=250, sigma=0.004, length=None):
    if length is None:
        length = int(5 * sigma / dt)
    t = np.arange(length) * dt
    k = np.exp(-(t**2) / (2 * sigma**2)) * np.cos(2 * np.pi * f0 * t)
    k = k - k.mean()
    return k / np.abs(k).max()


def doe_kernel(tau_fast, tau_slow, dt, length=None):
    if length is None:
        length = max(2, int(5 * max(tau_fast, tau_slow) / dt))
    t = np.arange(length) * dt
    k = (np.exp(-t / tau_fast) / tau_fast) - (np.exp(-t / tau_slow) / tau_slow)
    k = k - k.mean()
    peak = np.abs(k).max()
    return k / peak if peak > 0 else k


def parallel_kernel(tau_fast, tau_slow, tau_sustained, mix, dt, length=None):
    """
    DoE bandpass (transient) + exponential lowpass (sustained), in parallel:
        k(t) = mix * bandpass(t) + (1 - mix) * lowpass(t)

    LNP analogue of LIF's V1 + V2 architecture; mix~0.85 gives SA1-like onset
    spikes + sustained.
    """
    bp = doe_kernel(tau_fast, tau_slow, dt, length=length)
    bp = bp - bp.mean()
    bp = bp / np.abs(bp).max() if np.abs(bp).max() > 0 else bp
    lp = exp_kernel(tau_sustained, dt, length=length)

    L = max(len(bp), len(lp))
    bp_pad = np.zeros(L); bp_pad[:len(bp)] = bp
    lp_pad = np.zeros(L); lp_pad[:len(lp)] = lp
    return mix * bp_pad + (1.0 - mix) * lp_pad


def apply_temporal_filter(signal_matrix, kernel):
    """Causal 1-D convolution along the time axis for every receptor."""
    return lfilter(kernel, [1.0], signal_matrix, axis=1)


# ----------------------------------------------------------------------------
# Nonlinearity (N stage)
# ----------------------------------------------------------------------------
def apply_nonlinearity(x, name, gain):
    """Map linear drive -> non-negative firing rate (Hz). gain is applied first."""
    z = gain * x
    if name == 'relu':
        return np.maximum(z, 0.0)             # half-wave: leading edges only
    if name == 'abs':
        return np.abs(z)                      # full-wave: both edges
    if name == 'softplus':
        return np.log1p(np.exp(np.clip(z, -50.0, 50.0)))
    if name == 'sigmoid':
        return np.maximum(0.0, 50.0 / (1.0 + np.exp(-(z - 3.0))) - 25.0)
    raise ValueError("Unknown nonlinearity: " + repr(name))


# ----------------------------------------------------------------------------
# Poisson spike generator (P stage)
# ----------------------------------------------------------------------------
def poisson_spikes(rate_hz, dt, rng):
    """
    Bernoulli sampler - at the rates of interest (<= few hundred Hz, dt=1 ms),
    """
    prob = np.clip(rate_hz * dt, 0.0, 1.0)
    return rng.random(prob.shape) < prob


# ----------------------------------------------------------------------------
# Main wrapper class
# ----------------------------------------------------------------------------
class LnpReceptors:
    """
    Drop-in LNP replacement for tactile_receptors.population_simulate.

    Parameters
    ----------
    base : tactile_receptors
        Already-initialised receptor population. Its spatial matrices
        (Gi, Es, Cs, Ce, Wc, Hc, Nr, Nc, Nds, dt, t, Rm, ...) are reused.
    tau, gain, nonlinearity, filter_kind, tau_slow, tau_sustained, mix
        Override per-type defaults. Sentinels left untouched fall back to
        PER_TYPE_DEFAULTS[base.Ttype], then module-level DEFAULT_*.
    filter_kind : 'lowpass' | 'bandpass' | 'parallel' | 'resonant' | 'gabor'
    nonlinearity : 'relu' | 'abs' | 'softplus' | 'sigmoid'
    rng_seed : int | None
        Seeded numpy Generator for reproducible Poisson sampling.
    """

    def __init__(self, base,
                 tau=_AUTO,
                 gain=_AUTO,
                 nonlinearity=_AUTO,
                 filter_kind=_AUTO,
                 tau_slow=_AUTO,
                 tau_sustained=_AUTO,
                 mix=_AUTO,
                 rng_seed=None):
        self.base = base

        # Resolution priority: explicit arg > PER_TYPE_DEFAULTS > DEFAULT_*.
        type_defs = PER_TYPE_DEFAULTS.get(getattr(base, 'Ttype', None), {})

        def _resolve(name, given, module_default):
            if given is _AUTO:
                return type_defs.get(name, module_default)
            return given

        self.tau               = _resolve('tau',           tau,           DEFAULT_TAU)
        self.tau_slow          = _resolve('tau_slow',      tau_slow,      DEFAULT_TAU_SLOW)
        self.tau_sustained     = _resolve('tau_sustained', tau_sustained, DEFAULT_TAU_SUSTAINED)
        self.mix               = _resolve('mix',           mix,           DEFAULT_MIX)
        self.gain              = _resolve('gain',          gain,          DEFAULT_GAIN)
        self.nonlinearity_name = _resolve('nonlinearity',  nonlinearity,  DEFAULT_NONLINEARITY)
        self.filter_kind       = _resolve('filter_kind',   filter_kind,   DEFAULT_FILTER_KIND)
        self.rng = np.random.default_rng(rng_seed)

    # ----- public API ------------------------------------------------------
    def population_simulate(self, EEQS=None, Ips=([], 'Pressure'),
                            acquire_spikes=True, disinf=True, noise=0):
        """
        Same call signature as tactile_receptors.population_simulate.

        Returns (Va, rate_per_sec, spike_trains, elapsed). Va is shaped
        (Rm*Rn, T), set to REST_POTENTIAL everywhere with SPIKE_SENTINEL at
        spike timesteps.
        """
        base = self.base
        dt   = base.dt
        T    = base.t.size
        N    = base.Rm * base.Rn

        tc1 = timec.time()

        # Spatial pre-processing - identical to the LIF model.
        Uc = self._compute_Uc(EEQS, Ips, noise=noise)

        # L stage: temporal filter
        if self.filter_kind == 'lowpass':
            kernel = exp_kernel(self.tau, dt)
        elif self.filter_kind == 'bandpass':
            kernel = doe_kernel(self.tau, self.tau_slow, dt)
        elif self.filter_kind == 'parallel':
            kernel = parallel_kernel(self.tau, self.tau_slow,
                                     self.tau_sustained, self.mix, dt)
        elif self.filter_kind == 'resonant':
            kernel = resonant_kernel(self.tau, dt)
        elif self.filter_kind == 'gabor':
            kernel = gabor_kernel(dt)
        else:
            raise ValueError("Unknown filter_kind: " + repr(self.filter_kind))

        drive = apply_temporal_filter(Uc, kernel)

        # N stage
        rate = apply_nonlinearity(drive, self.nonlinearity_name, self.gain)

        # P stage
        spike_mask = poisson_spikes(rate, dt, self.rng)

        # Format outputs to match the LIF model.
        Va = np.full((N, T), REST_POTENTIAL, dtype=float)
        Va[spike_mask] = SPIKE_SENTINEL

        spike_trains = []
        if acquire_spikes:
            for i in range(base.Rm):
                spike_trains.append(dt * np.where(spike_mask[i, :])[0])

        # Mirror the LIF side effect so callers can read base.Va / base.spike_trains.
        base.Va = Va
        base.spike_trains = spike_trains

        elapsed = timec.time() - tc1
        rate_per_sec = elapsed / max(base.T, 1e-9)
        if disinf:
            print('LNP', base.Ttype, rate_per_sec)
        return Va, rate_per_sec, spike_trains, elapsed

    # ----- internal helpers ------------------------------------------------
    def _compute_Uc(self, EEQS, Ips, noise=0):
        """
        Replay the LIF spatial pipeline up to and including the resistance
        network, but stop before the bandpass / LIF stages.
        """
        base = self.base
        T    = base.t.size
        N    = base.Rm * base.Rn
        Dbp  = rslib.Dbp

        Cm_skin = (1.0 - base.poisson_v ** 2) / (2.0 * base.E_ym)

        if noise > 0:
            SN = 1e-6 * rslib.butterworth_filter(
                1, np.random.uniform(-noise, noise, T), 1000, 'low', 10e3
            )
        else:
            SN = np.zeros(T)

        Uc = np.zeros((N, T))
        Dt_t = np.zeros(N)

        for pt in range(T):
            # Skin contact slice
            se1 = int((Ips[0][pt, 1] + base.Hc / 2.0) / Dbp)
            se2 = int((Ips[0][pt, 0] + base.Wc / 2.0) / Dbp)
            SC  = EEQS[1][se1:se1 + base.Nr, se2:se2 + base.Nc]

            # Resistance-network input
            RI  = SC[0:base.Nr:base.Nds, 0:base.Nc:base.Nds] * 1e-3
            SCT = RI[base.Es[0], base.Es[1]]

            # Press depth
            if Ips[1] == 'Pressure':
                denom = base.Nds * Dbp * 1e-3 * np.sqrt(RI[RI > 0].size / 4.0)
                Dp = 0.0 if denom == 0 else Cm_skin * Ips[0][pt, 2] / denom
            else:
                Dp = Ips[0][pt, 2]

            # Per-receptor indentation (clamped at 0)
            Ht       = np.max(SCT, axis=1)
            Dt_t[:]  = Ht - np.max(Ht) + Dp
            Dt_t[Dt_t <= 0] = 0.0

            # Stimulus current and resistance-network propagation
            Is = base.Cs * (Dt_t + SN[pt]) + \
                 base.Ce * Dt_t * np.var(1e3 * SCT, axis=1)
            Uc[:, pt:pt + 1] = base.Gi * np.asmatrix(Is).T

        return Uc
