# Experiment 3 Fit Report
## Random Intercepts + Random Slopes Model

### Convergence Status

**Status**: CONVERGED

All convergence criteria met:
- Max R-hat: 1.0047 (target: <1.01)
- Min ESS_bulk: 898.7 (target: >400)
- Min ESS_tail: 1009.3 (target: >400)
- Max MCSE/SD: 0.0337 (target: <0.05)

### Key Parameters

See `summary.csv` for full parameter estimates.

### Scientific Question

**Question**: Is treatment effect heterogeneity (tau_beta) meaningfully different from zero?

Check the tau_beta posterior in `posterior_plots.png` and `summary.csv`.
If the 95% credible interval excludes zero and is substantively meaningful,
this supports the hypothesis that treatment effects vary across schools.

### Output Files

- `posterior.nc`: ArviZ InferenceData (NetCDF format) with log_likelihood for LOO-CV
- `summary.csv`: Full parameter summary statistics
- `trace_plots.png`: Trace plots for key parameters
- `rank_plots.png`: Rank plots for convergence assessment
- `energy_plot.png`: Energy diagnostic for HMC
- `posterior_plots.png`: Posterior distributions
- `autocorr_plots.png`: Autocorrelation plots