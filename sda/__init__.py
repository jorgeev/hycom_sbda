"""Score-based data assimilation (SDA) on the unconditional priors of ``diffusion/``.

Reproduces GenDA's inference (Martin et al. 2025, ``src/sda.py`` +
``inference/OSSE_inference.py``; Rozet & Louppe 2023) against the EDM priors
this repo trains:

    obs.py            observation operators A(x), masks, noise, YAML spec
    vpsde.py          VP-cosine schedule, EDM<->eps adapter, predictor-corrector
                      and EDM-native samplers
    guidance.py       Gaussian-likelihood guidance (Tweedie estimate, autograd
                      through the network), gradient checkpointing wrapper
    case.py           store -> case_<size>.npz (truth, climatology, masks, y)
    assimilate.py     case + checkpoint -> samples_sda_<size>.npz
    diagnostics_sda.py  per-day panels + observed/unobserved skill vs a control

Layering: imports ``diffusion.*``, ``eval.kernels`` and ``eval.kernels_cond``
only. Nothing in ``diffusion/`` or ``eval/`` is modified, and the output npz
follows the ``eval.diagnostics_cond.CondSamples`` schema so that script scores
it unchanged.
"""
