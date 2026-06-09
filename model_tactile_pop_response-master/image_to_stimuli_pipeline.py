# -*- coding: utf-8 -*-
"""
image_to_stimuli_pipeline.py - visualise the image -> EEPS pipeline.

Runs the standard `constructing_equivalent_probe_stimuli_from_image` pipeline
on the "ISA" letter image and saves a one-column figure showing each stage:
original -> grayscale -> height (edge-enhanced) -> EPS -> EEPS.

Used as a presentation/report figure to explain what the stimulus front-end
does to a raw photograph before the receptor model sees it.

Output:
  saved_figs/image_to_stimuli_pipeline.png

Usage (from the model_tactile_pop_response-master/ directory):
  python image_to_stimuli_pipeline.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import img_to_eqstimuli as imeqst
import simset as mysim


SOURCE_IMG = 'saved_figs/letters_ISA.jpg'
WIDTH_MM   = 120.0
HEIGHT_MM  = 12.0
OUT_PATH   = 'saved_figs/image_to_stimuli_pipeline.png'


def main():
    img = Image.open(SOURCE_IMG)
    buf = imeqst.constructing_equivalent_probe_stimuli_from_image(
        img, WIDTH_MM, HEIGHT_MM, mysim.fingertiproi
    )[0]

    fig = plt.figure(figsize=(10, 18))
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    # s=-1 populates the left column of the 8x2 grid used by print_figs,
    # giving a single-column 5-panel layout.
    imeqst.print_figs(-1, buf, fig)

    plt.savefig(OUT_PATH, bbox_inches='tight', dpi=300)
    print('Saved figure to', OUT_PATH)


if __name__ == '__main__':
    main()
