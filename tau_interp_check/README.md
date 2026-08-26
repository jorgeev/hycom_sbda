# tau_interp_check — provenance of the tau_x/tau_y ~7.3 km spectral spike

Temporary analysis code (2026-08). fig2_spectra of `runs/prior_gulfstream/eval_step1140000`
shows a sharp spike in the real tau spectra at exactly 4·dx ≈ 7.27 km (with a 2·dx
harmonic). Hypothesis under test: it is an artifact of the bilinear regridding used by
`~/scrach/hycomhd_ds/make_gulfstream_dataset.py`.

Three spectra, identical PSD path (`eval.kernels.radial_psd`), identical patch sampling
(the `real_days`/`real_pos` stored in `samples_128.npz`):

1. `eval`/`store` — the current training store (bilinear regrid);
2. `source` — raw `/unity/f1/ozavala/DATA/ATLc0.02_exp_04.3/{taux,tauy}` on the native
   HYCOM grid, untouched by us;
3. `bicubic` — tau re-regridded with a C² `RectBivariateSpline(k=3)`, stored at
   `/unity/f1/ozavala/DATA/ATLc0.02_exp_04.3/auxiliar_datasets/gulfstream_tau_bicubic.zarr`.

If the source curve already carries the spike, the dataset build is exonerated and the
artifact is upstream (in the HYCOM run's atmospheric-forcing interpolation).

Run order (from the repo root):

```bash
j1=$(sbatch --parsable tau_interp_check/build_bicubic.slurm)
sbatch --dependency=afterok:$j1 tau_interp_check/spectra.slurm
```

Outputs: `figs/tau_spectra_compare.npz`, `figs/fig_tau_spectra_compare.png`.
