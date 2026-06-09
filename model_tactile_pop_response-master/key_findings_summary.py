"""
key_findings_summary.py - one-page A4 summary of the project's key findings.

Produces saved_figs/key_findings_summary.png - a portrait A4 figure with:

  A  Architecture           - schematic: same skin model, LIF vs LNP cascade.
  B  Linear kernel + nonlinearity (loaded from saved_figs/nonlinearities.png).
  C  Ramp-and-hold PSTH    - loaded from saved_figs/compare_ramp_psth.png.
  D  Receptive-field maps  - LIF vs LNP vs Saal 2017.
  E  LIF-LNP correlations  - r_map and r_row, fingertip vs palm.
  F  "ISABEL" on the palm   - SA1 + RA1 rasters.
  G  "ISABEL" on the fingertip - SA1 + RA1 rasters.

Run from model_tactile_pop_response-master/:
    python key_findings_summary.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image

import simset as mysim
import Lnp as lnplib

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

TYPES_SR     = ['SA1', 'RA1']             # used in panels C, D, E
TYPE_COLS_SR = mysim.colors[:2]           # green, blue
LIF_COL      = '#444444'
LNP_COL      = '#d95f02'


# ---------------------------------------------------------------------------
# Figure scaffold - A4 portrait
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(8.27, 11.69))
fig.suptitle('Project 2 - LNP Population Model of Tactile Encoding: '
             'Key Findings',
             fontsize=13, fontweight='bold', y=0.985)

# Outer 5-row layout.
#   Row 0  A | B               (schematic | kernels)
#   Row 1  C                   (30-trial ramp, spans both cols)
#   Row 2  D | E               (receptive-field maps | row-correlation bars)
#   Row 3  F | G               ("ISABEL" palm | fingertip)
#   Row 4  footer
gs = GridSpec(5, 2,
              # Row 1 (panel C): tall for the ramp-and-hold PSTH.
              # Row 4 (caption): bumped to 1.7 so the wrapped paper-style
              # caption (~9 lines at fontsize 7) sits inside the cell
              # instead of bleeding off the bottom of the page.
              height_ratios=[1.35, 2.8, 2.1, 2.0, 1.7],
              hspace=0.70, wspace=0.22,
              # Asymmetric side margins: left=0.07 leaves room for the
              # y-axis labels of left-column panels (D's "RF area [mm²]" +
              # log tick labels, F's "EPS / SA1 / RA1" row labels), which
              # were being clipped against the page edge at left=0.045.
              # Right stays tight at 0.955 since no panel has right-side
              # labels needing the gutter.
              left=0.07, right=0.955, top=0.945, bottom=0.035)


# ===========================================================================
# (A)  Architecture schematic
# ===========================================================================
axA = fig.add_subplot(gs[0, 0])
axA.set_title('A. Architecture', loc='left', fontweight='bold', pad=6)
axA.set_xlim(0, 10); axA.set_ylim(0, 10)
axA.axis('off')

def box(ax, x, y, w, h, text, fc='#eeeeee', ec='black', lw=0.9,
        fontsize=7.5, fontweight='normal', va='center'):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                   boxstyle="round,pad=0.08",
                                   linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va=va,
            fontsize=fontsize, fontweight=fontweight)

# Linear flow: Image -> Skin -> Spike-generator -> Spikes.
# The spike-generator is drawn as a composite "swap-out" box containing
# LIF (greyed) vs LNP (highlighted) options, making the substitution
# obvious at a glance.

# Box 1: Image (top-left of the flow)
box(axA, 0.1, 4.0, 2.1, 2.5, 'Image\n(EEPS pins)', fc='#f5f5f5')
# Box 2: Skin mechanics
box(axA, 2.7, 3.0, 2.6, 4, 'Skin mechanics\n+ resistance\nnetwork',
    fc='#cfe8ff')
axA.annotate('', xy=(2.7, 5.1), xytext=(2.2, 5.1),
             arrowprops=dict(arrowstyle='->', lw=1.0))
axA.annotate('', xy=(5.8, 5.1), xytext=(5.3, 5.1),
             arrowprops=dict(arrowstyle='->', lw=1.0))

# Box 3: Spike-generator swap box (LIF vs LNP)
# Outer frame
outer = mpatches.FancyBboxPatch((5.8, 1.8), 3.3, 7.8,
                                boxstyle="round,pad=0.10",
                                linewidth=1.2, edgecolor='black',
                                facecolor='#fafafa')
axA.add_patch(outer)
axA.text(7.4, 7.85, 'Spike generator',
         ha='center', fontsize=7, fontweight='bold', color='#222')
# LIF option (top, greyed)
box(axA, 6.1, 5.5, 2.7, 1.9, 'LIF chain\n(bandpass + LIF)',
    fc='#dcdcdc', ec='#666', lw=0.7, fontsize=7)
axA.text(7.4, 4.6, 'OR', ha='center', fontsize=7.5,
         color='#777', fontweight='bold', fontstyle='italic')
# LNP option (bottom, highlighted as my contribution)
box(axA, 6.1, 2.4, 2.7, 1.9, 'LNP cascade\nL -> N -> P',
    fc='#ffd9b3', ec=LNP_COL, lw=1.7, fontsize=7.5, fontweight='bold')

# Bottom annotation summarising what changes
axA.text(5.0, 0.0,
         'Linear Intergrate and Fire Model (LIF) [1] replaced with Linear\n'
         'Non-Linear Poisson (LNP).  ~1.4x faster.',
         ha='center', va='center', fontsize=6.2,
         bbox=dict(boxstyle='round,pad=0.35', fc='#fff8e8', ec='#bba'))


# ===========================================================================
# (B)  Per-type LNP cascade: linear kernel (left) + nonlinearity (right).
# ===========================================================================
# Horizontal split inside the B cell - reading order matches the cascade
# L -> N -> P. No wrapper axis: panel title sits on the left sub-axis.
b_sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1],
                                wspace=0.45)

# Match plot_model_params/plot_linear_kernels.py + plot_nonlinearities.py:
# SA1=C2 (green), RA1=C0 (blue), PC='m' (magenta). PC is dashed in the
# nonlinearity panel to disambiguate it from RA1 (same shape, smaller gain).
LNP_COLOURS = {'SA1': 'C2', 'RA1': 'C0', 'PC': 'm'}
# Panel E (correlation bars) still uses mysim.colors for consistency with
# the rest of the codebase - 'g','b','m' is visually the same scheme.
type_cols_all = mysim.colors[:3]

# ---- Left sub-axis: linear kernels k(t) ----------------------------------
# Mirrors plot_model_params/plot_linear_kernels.py - same kernels, same
# 120 ms window, same colours; only fonts/lw are shrunk to fit the cell.
axBk = fig.add_subplot(b_sub[0])
DT      = 0.001
KLEN_MS = 120
klen    = int(KLEN_MS / 1000 / DT)
t_k_ms  = np.arange(klen) * DT * 1000.0

k_sa1 = lnplib.parallel_kernel(tau_fast=0.005, tau_slow=0.015,
                               tau_sustained=0.07, mix=0.85,
                               dt=DT, length=klen)
k_ra1 = lnplib.doe_kernel(tau_fast=0.005, tau_slow=0.010,
                          dt=DT, length=klen)
k_pc  = lnplib.resonant_kernel(tau=0.001, dt=DT, length=klen)

axBk.plot(t_k_ms, k_sa1, color=LNP_COLOURS['SA1'], lw=1.5,
          label='SA1 parallel')
axBk.plot(t_k_ms, k_ra1, color=LNP_COLOURS['RA1'], lw=1.5,
          label='RA1 bandpass')
axBk.plot(t_k_ms, k_pc,  color=LNP_COLOURS['PC'],  lw=1.5,
          label='PC resonant')
axBk.axhline(0, color='gray', lw=0.3)
axBk.set_xlim(0, KLEN_MS)
axBk.set_xlabel('Time [ms]', fontsize=7.5)
axBk.set_ylabel('Kernel amplitude (norm.)', fontsize=7.5)
axBk.tick_params(labelsize=6.5)
axBk.legend(fontsize=6.5, frameon=False, loc='upper right',
            handlelength=1.0, handletextpad=0.3)
axBk.set_title('B. (a) Linear kernel k(t)', fontsize=9.5,
               fontweight='bold', loc='left', pad=4)

# ---- Right sub-axis: static nonlinearities φ(x) ---------------------------
# Mirrors plot_model_params/plot_nonlinearities.py - illustrative per-type
# gains (100 / 60 / 50) so all three curves are visible together. The
# nonlinearity shape (relu vs abs) and the per-type colour are what matter;
# the real gains in PER_TYPE_DEFAULTS span ~64x and would hide RA1/PC at
# this scale.
axBn = fig.add_subplot(b_sub[1])
xn = np.linspace(-1.5, 1.5, 400)
GAIN_SA1, GAIN_RA1, GAIN_PC = 100.0, 60.0, 50.0
relu_sa1 = GAIN_SA1 * np.maximum(0.0, xn)
abs_ra1  = GAIN_RA1 * np.abs(xn)
abs_pc   = GAIN_PC  * np.abs(xn)
axBn.plot(xn, relu_sa1, color=LNP_COLOURS['SA1'], lw=1.5,
          label=r'SA1 ReLU  $g\cdot\max(0,x)$')
axBn.plot(xn, abs_ra1,  color=LNP_COLOURS['RA1'], lw=1.5,
          label=r'RA1 abs  $g\cdot|x|$')
axBn.plot(xn, abs_pc,   color=LNP_COLOURS['PC'],  lw=1.5, linestyle='--',
          label='PC abs (resonant-driven)')
axBn.axhline(0, color='gray', lw=0.3)
axBn.axvline(0, color='gray', lw=0.3)
axBn.set_xlabel(r'Filtered drive $k(t)*U_c(t)$', fontsize=7.5)
axBn.set_ylabel(r'rate $\lambda$ [Hz]', fontsize=7.5)
axBn.tick_params(labelsize=6.5)
axBn.legend(fontsize=6.0, frameon=False, loc='upper center',
            handlelength=1.0, handletextpad=0.3)
axBn.set_title(r'(b) Nonlinearity', fontsize=9,
               fontweight='bold', loc='left', pad=4)


# ===========================================================================
# (C)  Ramp-and-hold PSTH - loaded from saved_figs/compare_ramp_psth.png
# ===========================================================================
axC = fig.add_subplot(gs[1, :])
axC.set_title('C. 30-trial ramp-and-hold response: LIF vs LNP',
              loc='left', fontweight='bold', pad=2)
ramp_img = Image.open('saved_figs/compare_ramp_psth.png')
axC.imshow(np.array(ramp_img), aspect='auto')
axC.set_xticks([]); axC.set_yticks([])
for spine in axC.spines.values():
    spine.set_visible(False)


# ===========================================================================
# Helper: spatial-scan raster panel (used for D = palm, E = fingertip)
# ===========================================================================
def plot_spatial_panel(cell_spec, sres_path, eps_image_path,
                       width_mm, height_mm, speed, simdt,
                       title, scan_shift_mm=0.2):
    """
    Draws an EPS reference strip on top of stacked SA1 + RA1 rasters.

    cell_spec is the GridSpec cell allocated to this panel; this function
    populates it via a 3-row internal sub-grid (ref / SA1 / RA1).
    """
    # Title sits on an invisible wrapper axis.
    wrap = fig.add_subplot(cell_spec)
    wrap.set_title(title, loc='left', fontweight='bold', pad=8)
    wrap.axis('off')

    sub = GridSpecFromSubplotSpec(3, 1, subplot_spec=cell_spec,
                                  hspace=0.25, height_ratios=[0.55, 1.0, 1.0])

    # Top - reference EPS image strip
    ax_ref = fig.add_subplot(sub[0])
    img_ref = Image.open(eps_image_path).convert('L')
    ax_ref.imshow(np.array(img_ref),
                  extent=(0, width_mm, 0, 1),
                  cmap='Greys_r', aspect='auto')
    ax_ref.set_xlim(0, width_mm); ax_ref.set_ylim(0, 1)
    ax_ref.set_xticks([]); ax_ref.set_yticks([])
    ax_ref.set_ylabel('EPS', fontsize=7.5, rotation=0, ha='right', va='center')

    # SA1 + RA1 spike rasters
    sres = np.load(sres_path, allow_pickle=True)
    num_rows = len(sres[0])
    ys = np.linspace(height_mm, 0, num_rows)
    for i, name in enumerate(TYPES_SR):
        ax = fig.add_subplot(sub[i + 1])
        rows_data = sres[i]
        for row_idx in range(num_rows):
            times = simdt * np.where(np.asarray(rows_data[row_idx]) == 0.04)[0]
            if len(times):
                ax.scatter(times * speed,
                           np.full(len(times), ys[row_idx]),
                           s=0.25, c=TYPE_COLS_SR[i], marker='.')
        ax.set_xlim(0, width_mm)
        ax.set_ylim(0, height_mm)
        ax.set_ylabel(name, fontsize=8, rotation=0, ha='right', va='center')
        ax.set_yticks([0, height_mm/2, height_mm])
        if i == 1:
            ax.set_xlabel('Probe position x [mm]', fontsize=8)
        else:
            ax.set_xticklabels([])


# ===========================================================================
# (E)  LIF-LNP map-level + row-level Pearson r, fingertip vs palm
# ===========================================================================
# Computed live from the saved sim arrays so the numbers stay in sync with
# whatever simulation last wrote them. Two sub-panels share the cell:
#   top    : r_map  (pixel-by-pixel correlation on the 2D spike-density map)
#   bottom : r_row  (correlation on the per-scan-row spike totals)
SPIKE_VAL_  = 0.04
SPEED_      = 20.0
SIMDT_      = 0.001
WIDTH_MM_   = 90.0
N_X_BINS_   = 90
types_all   = ['SA1', 'RA1', 'PC']

def _va_to_map(rows_per_receptor):
    out = np.zeros((len(rows_per_receptor), N_X_BINS_), dtype=int)
    bin_edges = np.linspace(0, WIDTH_MM_, N_X_BINS_ + 1)
    for r in range(len(rows_per_receptor)):
        va        = np.asarray(rows_per_receptor[r])
        spike_idx = np.where(va == SPIKE_VAL_)[0]
        spike_x   = SPEED_ * SIMDT_ * spike_idx
        hist, _   = np.histogram(spike_x, bins=bin_edges)
        out[r]    = hist
    return out

def _pearson(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])

def _compute_correlations(lif_path, lnp_path):
    lif = np.load(lif_path, allow_pickle=True)
    lnp = np.load(lnp_path, allow_pickle=True)
    rmap, rrow = [], []
    for ch in range(3):
        m_lif = _va_to_map(lif[ch])
        m_lnp = _va_to_map(lnp[ch])
        rmap.append(_pearson(m_lif, m_lnp))
        rrow.append(_pearson(m_lif.sum(axis=1), m_lnp.sum(axis=1)))
    return rmap, rrow

ft_rmap, ft_rrow = _compute_correlations(
    'Data/name_isabel_sim_res_LIF.npy', 'Data/name_isabel_sim_res.npy')
palm_rmap, palm_rrow = _compute_correlations(
    'Data/name_isabel_sim_res_palm_LIF.npy', 'Data/name_isabel_sim_res_palm.npy')

# ===========================================================================
# Row 2 layout  - D | E with D given more width
# ===========================================================================
# quantify_rf_area.png is ~1.32:1 (nearly square), so a half-width cell forces
# matplotlib to height-constrain it and waste horizontal space. A 1.6:1
# width split gives D a roughly square cell where the image scales up
# legibly; E stays compact since its two bar charts are narrow anyway.
row2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :],
                               width_ratios=[1.0, 1.0], wspace=0.18)


# ===========================================================================
# (D)  Receptive-field areas - grouped bars (no heatmaps).
# ===========================================================================
# Pulls LIF/LNP/Saal numbers straight from the cached metrics so we don't
# re-embed the heatmap-laden quantify_rf_area.png. Three afferents x three
# models in a single grouped-bar axes; log y so SA1 (~few mm²) and PC
# (~tens-of-mm²) are both readable on one scale.
axD = fig.add_subplot(row2[0])
axD.set_title('D. Receptive-field areas - LIF vs LNP vs Saal et al. 2017',
              loc='left', fontsize=8, fontweight='bold', pad=2)

_rf_metrics = np.load('Data/cmp_rf_area_metrics.npy', allow_pickle=True)
_rf_area    = _rf_metrics[1]   # {'lif': {SA1,RA1,PC}, 'lnp': {SA1,RA1,PC}}
_rf_saal    = _rf_metrics[2]   # {'SA1':10, 'RA1':15, 'PC':100}

_xpos    = np.arange(3)
_bar_w   = 0.27
_lif     = [_rf_area['lif'][t] for t in types_all]
_lnp     = [_rf_area['lnp'][t] for t in types_all]
_saal    = [_rf_saal[t]        for t in types_all]

_bars_lif  = axD.bar(_xpos - _bar_w, _lif,  _bar_w,
                     color='#444444', edgecolor='black', linewidth=0.5,
                     label='LIF')
_bars_lnp  = axD.bar(_xpos,          _lnp,  _bar_w,
                     color=type_cols_all, edgecolor='black', linewidth=0.5,
                     label='LNP (per-type colour)')
_bars_saal = axD.bar(_xpos + _bar_w, _saal, _bar_w,
                     color='lightgray', edgecolor='black', linewidth=0.5,
                     hatch='..', label='Saal 2017')

# Inline value labels above each bar - readable even on a log scale.
for _group in (_bars_lif, _bars_lnp, _bars_saal):
    for _b in _group:
        _h = _b.get_height()
        axD.text(_b.get_x() + _b.get_width() / 2.0, _h * 1.06,
                 '{:.1f}'.format(_h),
                 ha='center', va='bottom', fontsize=7)

axD.set_yscale('log')
# Explicit padding on both axes so the bar groups and value labels don't
# crowd the cell edges. xlim leaves 0.6 of a bar-cluster gutter on each
# side; ylim caps well above the 100.0 Saal PC bar so the value label
# doesn't crowd the top spine.
axD.set_xlim(-0.65, 2.65)
axD.set_ylim(0.9, 350.0)
axD.set_xticks(_xpos)
axD.set_xticklabels(types_all, fontsize=9)
axD.set_ylabel('RF area  [mm²]', fontsize=8.5)
axD.tick_params(axis='y', labelsize=7.5)
axD.legend(loc='upper left', fontsize=7, frameon=False,
           handlelength=1.2, handletextpad=0.4)
axD.grid(axis='y', which='both', color='lightgrey', lw=0.4, alpha=0.6)
axD.set_axisbelow(True)


# ===========================================================================
# (E)  LIF-LNP map-level + row-level Pearson r, fingertip vs palm
# ===========================================================================
# Sub-titles follow panel B's "(a)/(b)" pattern: panel-letter on the top
# sub-axis, the bottom one just gets the (b) label. Cleaner than the old
# "top:/bottom:" prefixes baked into the title strings.
e_sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=row2[1],
                                hspace=0.55, height_ratios=[1.0, 1.0])

def _bar_pair(ax, fingertip, palm, ylabel, title):
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, fingertip, w, color=type_cols_all,
           edgecolor='black', linewidth=0.5, label='Fingertip')
    ax.bar(x + w/2, palm, w, color=type_cols_all,
           alpha=0.45, edgecolor='black', linewidth=0.5,
           hatch='//', label='Palm')
    for xi, v in zip(x - w/2, fingertip):
        ax.text(xi, v + 0.03, f'{v:.2f}', ha='center', fontsize=6.5)
    for xi, v in zip(x + w/2, palm):
        ax.text(xi, v + 0.03, f'{v:.2f}', ha='center', fontsize=6.5)
    ax.axhline(1.0, color='grey', linestyle=':', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(types_all, fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_title(title, fontsize=9, fontweight='bold', loc='left', pad=3)

axE_top = fig.add_subplot(e_sub[0])
_bar_pair(axE_top, ft_rmap, palm_rmap,
          ylabel=r'$r_{\mathrm{map}}$',
          title='E. (a) Pixel-map r')
axE_top.legend(frameon=False, loc='upper right', fontsize=6.5, ncol=2,
               handlelength=1.0, handletextpad=0.3)

axE_bot = fig.add_subplot(e_sub[1])
_bar_pair(axE_bot, ft_rrow, palm_rrow,
          ylabel=r'$r_{\mathrm{row}}$',
          title='(b) Row-sum r')


# ===========================================================================
# (F)  ISABEL on the palm - SA1 + RA1 rasters
# ===========================================================================
plot_spatial_panel(
    cell_spec     = gs[3, 0],
    sres_path     = 'Data/name_isabel_sim_res_palm.npy',
    eps_image_path= 'saved_figs/letters_ISABEL.jpg',
    width_mm      = 90.0,
    height_mm     = 12.0,
    speed         = 20,
    simdt         = 0.001,
    title         = 'F. "ISABEL" on the palm',
)


# ===========================================================================
# (G)  ISABEL on the fingertip - SA1 + RA1 rasters
# ===========================================================================
plot_spatial_panel(
    cell_spec     = gs[3, 1],
    sres_path     = 'Data/name_isabel_sim_res.npy',
    eps_image_path= 'saved_figs/letters_ISABEL.jpg',
    width_mm      = 90.0,
    height_mm     = 12.0,
    speed         = 20,
    simdt         = 0.001,
    title         = 'G. "ISABEL" on the fingertip',
)


# ===========================================================================
# Footer - research-paper-style figure caption.
# ===========================================================================
ax_ftr = fig.add_subplot(gs[4, :])
ax_ftr.axis('off')
ax_ftr.text(
    0.5, 0.5,
    "Figure 1. LNP population model of tactile encoding - summary of project "
    "results. "
    "(A) Architecture: spike-generator was swapped, from the "
    "Leaky Integrate-and-Fire model (LIF, grey) to a Linear-Nonlinear-Poisson "
    "cascade (LNP, orange); the front-end (EEPS -> skin mechanics + "
    "resistance network) is unchanged from [1]. "
    "(B) Per-afferent LNP parameters - (a) causal linear kernels k(t): SA1 "
    "parallel (DoE bandpass + lowpass, mix 0.85), RA1 DoE bandpass, PC "
    "damped 250 Hz resonant; (b) static nonlinearities: SA1 half-wave "
    "ReLU, RA1 / PC full-wave (illustrative gains). "
    "(C) Ramp-and-hold response, 30 trials per condition. Trapezoidal force "
    "(0.5 N peak; 100 ms ramp · 500 ms hold · 100 ms release · 300 ms "
    "tail) delivered to a uniform skin patch through the fingertip "
    "central receptor. Rows: LIF (top) vs LNP (bottom); columns: SA1 / RA1 "
    "/ PC. Each subplot shows the trial-by-trial spike raster (ticks), "
    "smoothed PSTH (black) and the stimulus force envelope (grey); mean "
    "firing rate over the onset, hold and offset phases is annotated at "
    "the top of each subplot. "
    "(D) Receptive-field area (mm²) at the per-afferent rate threshold - "
    "LIF vs LNP vs the Saal et al. 2017 benchmark; log scale. "
    "(E) Pearson correlation between LIF and LNP spike-density "
    "representations of the 'ISABEL' scan: (a) pixel-map r and (b) "
    "row-sum r, computed for the fingertip (solid) and palm (hatched) "
    "skin sites. "
    "(F, G) Multi-row scan of the printed word 'ISABEL' (90 x 12 mm "
    "letters; 20 mm/s probe speed; 0.35 N constant press) on the (F) palm "
    "and (G) fingertip; top strip: EPS reference; below: SA1 (green) and "
    "RA1 (blue) spike rasters of the central receptor on each scan row.",
    ha='center', va='center', fontsize=6, wrap=True,
    bbox=dict(boxstyle='round,pad=0.2', fc='#f6f0e2', ec='#aa9'))


# ---------------------------------------------------------------------------
out = 'saved_figs/key_findings_summary.png'
os.makedirs('saved_figs', exist_ok=True)
plt.savefig(out, dpi=200)
print('Saved', out)
plt.show()
