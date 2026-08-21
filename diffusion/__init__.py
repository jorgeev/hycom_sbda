"""Score-based diffusion priors for gridded ocean state, over any zarr store.

An EDM-preconditioned diffusion model that either

  * reconstructs high-resolution truth fields from a k-step window of
    satellite-like degraded channels (conditional super-resolution), or
  * learns an unconditional joint prior over the full state, for use as a
    Bayesian prior in score-based data assimilation (GenDA / SDA framing).

Which store it trains on is data, not code: a descriptor in ``datasets/*.yaml``
declares the layout, and an experiment config selects one with ``dataset:``.
See the top-level README.
"""
