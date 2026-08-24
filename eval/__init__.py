"""Sample-space evaluation for the unconditional diffusion prior.

Deliberately outside ``diffusion/``: that package stays a pure training core
with no plotting dependencies, as README.md's "Not included" section describes.
Nothing in ``diffusion/`` imports anything from here.

Two steps, two scripts:

    python -m eval.gen_prior    --ckpt <run>/best.pt --size full --out s.npz
    python -m eval.diagnostics  --samples s.npz --out figs/

with the shared numerics in ``eval.kernels`` (``--selftest`` runs them against
the store).
"""
