# -*- coding: utf-8 -*-
"""
speed_check_lif_vs_lnp.py -- measure the runtime speedup of LNP over LIF on
the same stimulus (single-row "ISABEL" scan across the fingertip), per
afferent type.

Times one population_simulate() call for each model, averaged over N_RUNS,
and reports the LIF/LNP ratio.

Run from the model_tactile_pop_response-master/ directory:
    python speed_check_lif_vs_lnp.py
"""
import time
import numpy as np
from PIL import Image

import Receptors as rslib
import simset as mysim
import img_to_eqstimuli as imeqst
import Lnp as lnplib


# ---------------------------------------------------------------------------
# Scan parameters -- matches name_on_fingertip.py / Ibtest.py
# ---------------------------------------------------------------------------
WIDTH_MM   = 90.0
HEIGHT_MM  = 12.0
SHIFT      = 0.2
SPEED      = 20
PF         = 0.35
SIMDT      = 0.001
SIMT       = WIDTH_MM / SPEED
LETTER_IMG = 'saved_figs/letters_120-12.jpg'

# Per-afferent LNP parameters now live as PER_TYPE_DEFAULTS in Lnp.py.
# LnpReceptors picks them up automatically from base.Ttype; override here only
# if you want this script to differ from the canonical set.
N_RUNS = 5


# ---------------------------------------------------------------------------
# Stimulus setup
# ---------------------------------------------------------------------------
img = Image.open(LETTER_IMG)
_, _, eeps, _ = imeqst.constructing_equivalent_probe_stimuli_from_image(
    img, WIDTH_MM, HEIGHT_MM, mysim.fingertiproi
)
Aeeps = eeps

PF_arr = PF * np.ones(int(SIMT / SIMDT))
ips = imeqst.img_stimuli_scaning_with_uniformal_speed(
    SIMDT, SIMT, PF_arr, SPEED, 0, WIDTH_MM, 0, HEIGHT_MM, SHIFT,
)
mid_row = len(ips) // 2   # one representative scan row

pbuf = np.load('Data/loc_pos_buf_fingertip.npy', allow_pickle=True)


# ---------------------------------------------------------------------------
# Time both models per afferent type
# ---------------------------------------------------------------------------
print(f'Speed check: LIF vs LNP, single sweep ({SIMT:.1f}s of simulation), '
      f'{N_RUNS} runs averaged.')
print(f'Stimulus: scan one row of "ISABEL" at v={SPEED} mm/s, force={PF} N.\n')

header = f'{"type":<5} {"n":>5} {"LIF mean (s)":>14} {"LNP mean (s)":>14} {"speedup":>10}'
print(header)
print('-' * len(header))

results = []
for ch, name in enumerate(['SA1', 'RA1', 'PC']):
    s = rslib.tactile_receptors(Ttype=name)
    s.set_population(pbuf[ch][0], pbuf[ch][1],
                     simTime=SIMT, sample_rate=1/SIMDT,
                     Density=pbuf[ch][2], roi=mysim.fingertiproi)
    n_rec = len(pbuf[ch][0])

    # LIF -- one warm-up run then N_RUNS timed
    s.population_simulate(EEQS=Aeeps, Ips=[ips[mid_row], 'Pressure'],
                          noise=0, disinf=False)
    lif_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        s.population_simulate(EEQS=Aeeps, Ips=[ips[mid_row], 'Pressure'],
                              noise=0, disinf=False)
        lif_times.append(time.perf_counter() - t0)

    # LNP -- wrap once, then warm-up + timed
    lnp = lnplib.LnpReceptors(s, rng_seed=42)
    lnp.population_simulate(EEQS=Aeeps, Ips=[ips[mid_row], 'Pressure'],
                            noise=0, disinf=False)
    lnp_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        lnp.population_simulate(EEQS=Aeeps, Ips=[ips[mid_row], 'Pressure'],
                                noise=0, disinf=False)
        lnp_times.append(time.perf_counter() - t0)

    lif_mean = float(np.mean(lif_times))
    lnp_mean = float(np.mean(lnp_times))
    speedup  = lif_mean / lnp_mean if lnp_mean > 0 else float('inf')

    print(f'{name:<5} {n_rec:>5d} {lif_mean:>14.4f} {lnp_mean:>14.4f} '
          f'{speedup:>9.2f}x')
    results.append((name, n_rec, lif_mean, lnp_mean, speedup))

# Aggregate across all afferent types
total_lif = sum(r[2] for r in results)
total_lnp = sum(r[3] for r in results)
print('-' * len(header))
print(f"{'all':<5} {sum(r[1] for r in results):>5d} "
      f"{total_lif:>14.4f} {total_lnp:>14.4f} "
      f"{total_lif/total_lnp:>9.2f}x")
