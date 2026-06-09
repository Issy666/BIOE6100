# -*- coding: utf-8 -*-
"""
plot_nonlinearities.py - visualise the static nonlinearities used in the LNP
spike generator, one curve per afferent type, on a common axis.

The nonlinearity maps the filtered drive (output of the linear stage) to an
instantaneous firing rate (in Hz). SA1 uses ReLU (half-wave rectification),
RA1 and PC use abs (full-wave rectification - fire on both onset and offset).

Output: saved_figs/nonlinearities.png
"""
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Nonlinearity functions (match Lnp.py)
# ---------------------------------------------------------------------------
def relu(x, gain):
    return gain * np.maximum(0.0, x)

def absrect(x, gain):
    return gain * np.abs(x)

def sigmoid(x, gain, slope=3.0):
    return gain / (1.0 + np.exp(-slope * x))


# Per-afferent gains (illustrative, not the actual fit values --
# the shape is what matters for this figure).
GAIN_SA1 = 100.0
GAIN_RA1 = 60.0
GAIN_PC  = 50.0

x = np.linspace(-1.5, 1.5, 400)


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, relu(x, GAIN_SA1),    color='C2', lw=2.2,
        label='SA1: ReLU $\\phi(x)=\\mathrm{gain}\\cdot\\max(0,x)$')
ax.plot(x, absrect(x, GAIN_RA1), color='C0', lw=2.2,
        label='RA1: abs $\\phi(x)=\\mathrm{gain}\\cdot|x|$')
ax.plot(x, absrect(x, GAIN_PC),  color='m',  lw=2.2, linestyle='--',
        label='PC:  abs (resonant-driven)')

ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel('Filtered drive $k(t) * U_c(t)$')
ax.set_ylabel('Instantaneous rate $\\lambda$ (Hz)')
ax.set_title('LNP static nonlinearities $\\phi(\\cdot)$')
ax.legend(fontsize=9, loc='upper center')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
out = 'saved_figs/nonlinearities.png'
fig.savefig(out, bbox_inches='tight', dpi=200)
print('Saved figure to', out)
