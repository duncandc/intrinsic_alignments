# Literature IA Models

A collection of models from the literature relavent for modelling instrinsic alignments.


## Misalignment Models

Models for galaxy-halo misalignments can be found in `literature_ia_model_components.py`.  Currently this includes:

* [Bett (2012)](https://arxiv.org/abs/1108.3717)
* [Knebe et al. (2008)](https://arxiv.org/abs/0802.1917)
* [Okumura et al. (2009)](https://arxiv.org/abs/0809.3790)

See `literature_models_demo.ipynb` for an demonstration of the misalignment angle distributions in these models.  These models are explored in some detail in [Joachimi et al. (2012)](https://arxiv.org/abs/1305.5791).


## Correlation Function Models

We provide some fitting functions for galaxy-alignment correlation functions:

* [HRH* model](https://arxiv.org/abs/astro-ph/0310174)
* [HRH model](https://arxiv.org/abs/astro-ph/0005269)

These models can be found in `hrh_model.py`. See [Mandelbaum et al. (2006)](https://arxiv.org/abs/astro-ph/0509026) for the default parameters used in the HRH* class and [Heymans et al. (2004)](https://arxiv.org/abs/astro-ph/0310174) for the default parameters used in the HRH class.


## (N)LA Models

Work in progress...

See:
`linear_alignment_models.py`