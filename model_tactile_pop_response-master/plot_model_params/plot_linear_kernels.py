# -*- coding: utf-8 -*-
"""
plot_kernels.py - visualise the LNP temporal-filter kernels actually used
in this project (one per afferent type) on a common time axis.

Parameters match LNP_PARAMS in compare_lif_vs_lnp_letters.py:
    SA1 -> parallel kernel  (bandpass + lowpass)
    RA1 -> bandpass kernel  (difference of exponentials)
    PC  -> resonant kernel

Output: saved_figs/kernels.png
"""
import numpy as np
import matplotlib.pyplot as plt

import Lnp as lnplib


# ---------------------------------------------------------------------------
# Kernel definitions - parameters must match LNP_PARAMS in compare_lif_vs_lnp_letters.py.
# ---------------------------------------------------------------------------
DT = 0.001          # 1 ms - matches the simulation timestep
LEN_MS = 120        # show first 120 ms of each kernel
L = int(LEN_MS / 1000 / DT)

afferents = [
    {
        'name': 'SA1',
        'colour': 'C2',                        # green - matches other LNP plots
        'kernel': lnplib.parallel_kernel(tau_fast=0.005, tau_slow=0.015,
                                         tau_sustained=0.07, mix=0.85,
                                         dt=DT, length=L),
        'label': ('SA1: parallel (bandpass + lowpass)\n'
                  r'$\tau{=}5$\,ms, $\tau_{\mathrm{slow}}{=}15$\,ms, '
                  r'$\tau_{\mathrm{sus}}{=}70$\,ms, mix$=0.85$'),
        'note':  'transient + sustained $\\Rightarrow$ slow adapt.',
    },
    {
        'name': 'RA1',
        'colour': 'C0',                        # blue
        'kernel': lnplib.doe_kernel(tau_fast=0.005, tau_slow=0.010,
                                    dt=DT, length=L),
        'label': ('RA1: bandpass (diff.\\ of exponentials)\n'
                  r'$\tau{=}5$\,ms, $\tau_{\mathrm{slow}}{=}10$\,ms'),
        'note':  'biphasic $\\Rightarrow$ rapid adapt., onset+offset',
    },
    {
        'name': 'PC',
        'colour': 'm',                         # magenta
        'kernel': lnplib.resonant_kernel(tau=0.001, dt=DT, length=L),
        'label': ('PC: resonant\n'
                  r'$\tau{=}1$\,ms'),
        'note':  'oscillatory $\\Rightarrow$ HF vibration tuning',
    },
]


# ---------------------------------------------------------------------------
# Two-panel plot: kernels on the left, step responses on the right.
# Step response makes "transient peak + sustained plateau" readable directly.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for a in afferents:
    k = a['kernel']
    t_ms = np.arange(len(k)) * DT * 1000.0
    axes[0].plot(t_ms, k, color=a['colour'], lw=2.0, label=a['label'])

    # Step response = causal cumulative integral of kernel against a unit step.
    step = np.ones_like(k)
    resp = np.cumsum(k * step)
    axes[1].plot(t_ms, resp, color=a['colour'], lw=2.0,
                 label='{}: {}'.format(a['name'], a['note']))

for ax in axes:
    ax.axhline(0, color='gray', lw=0.3)
    ax.set_xlabel('Time [ms]')
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    ax.legend(fontsize=7.5, loc='upper right')

axes[0].set_ylabel('Kernel amplitude (normalised)')
axes[0].set_title('LNP temporal kernels --- one per afferent type')
axes[1].set_ylabel('Step-response amplitude')
axes[1].set_title('Step response ($\\int$ kernel $\\cdot$ step)')

fig.tight_layout()
out = 'saved_figs/kernels.png'
fig.savefig(out, bbox_inches='tight', dpi=200)
print('Saved kernel figure to', out)
