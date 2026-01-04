# Parameter Recovery Report: A3-Robust Model (Student-t Errors)

## Summary

**RESULT: PASS**

The A3-Robust model successfully recovers known parameters from synthetic data. All 5 simulations converged without divergences, and 90% credible intervals achieved 96% overall coverage. The degrees of freedom parameter (nu) shows expected high uncertainty but remains identifiable.

## Test Configuration

| Setting | Value |
|---------|-------|
| True alpha | 3.1 |
| True beta_weight | -0.9 |
| True beta_year | 0.03 |
| True sigma | 0.12 |
| True nu | 15.0 |
| Simulations | 5 |
| Observations per sim | 200 |
| CI level | 90% |

## Convergence Diagnostics

All simulations passed convergence checks:

- Max R-hat: 1.00 (threshold < 1.01)
- Min ESS bulk: 1705 (threshold > 400)
- Total divergences: 0

The model samples efficiently on synthetic data with no computational issues.

## Parameter Recovery

### 90% CI Coverage

| Parameter | Coverage | Notes |
|-----------|----------|-------|
| alpha | 80% (4/5) | One simulation missed due to sampling variation |
| beta_weight | 100% (5/5) | Excellent recovery |
| beta_year | 100% (5/5) | Excellent recovery |
| sigma | 100% (5/5) | Excellent recovery |
| nu | 100% (5/5) | Wide intervals but covers true value |
| **Overall** | **96%** | Exceeds 90% target |

### Mean Estimation Error

| Parameter | Mean Error | Interpretation |
|-----------|------------|----------------|
| alpha | -0.009 | Negligible bias |
| beta_weight | +0.021 | Slight positive bias, within uncertainty |
| beta_year | -0.001 | Negligible bias |
| sigma | -0.001 | Negligible bias |
| nu | +3.22 | Positive bias typical for nu |

## nu Identifiability Analysis

The degrees of freedom parameter (nu=15) presents a known identification challenge. At nu=15, the Student-t distribution is close to Gaussian, making it harder to distinguish tail behavior from measurement noise.

Key findings:
- Mean posterior nu: 18.2 (true: 15.0)
- Mean posterior SD: 10.8
- 90% CI coverage: 100%

The wide posterior intervals (ranging from ~13 to ~39 across simulations) reflect genuine uncertainty about tail behavior when nu is moderate. This is expected behavior, not a model defect. The posteriors correctly include the true value while acknowledging the difficulty of precise nu estimation with N=200 observations.

Visual inspection of `recovery_nu_focus.png` shows all intervals contain the true value despite substantial width variation across simulations.

## Visual Evidence

Recovery plots confirm the quantitative findings:

- `recovery_intervals.png`: Shows 90% CIs for all parameters across simulations. Green indicates coverage, red indicates miss. Only one alpha interval missed (simulation 1).

- `recovery_scatter.png`: Posterior means vs true values. Regression parameters (alpha, beta_weight, beta_year, sigma) cluster tightly near the identity line. nu shows expected scatter but remains centered appropriately.

- `recovery_zscore.png`: Z-score distribution is roughly centered at zero with most values within [-2, 2], consistent with well-calibrated posteriors.

- `recovery_nu_focus.png`: Dedicated nu visualization showing wide but well-calibrated intervals.

## Pass Criteria Assessment

| Criterion | Status |
|-----------|--------|
| All simulations converged | PASS |
| No divergent transitions | PASS |
| Coverage >= 60% | PASS (96%) |
| Max R-hat < 1.02 | PASS (1.00) |
| nu identifiable (coverage >= 60%) | PASS (100%) |

## Conclusion

The A3-Robust model demonstrates sound parameter recovery. The model is ready for fitting to real data. Key observations:

1. Regression coefficients (alpha, beta_weight, beta_year) recover precisely with minimal bias
2. Scale parameter (sigma) recovers accurately
3. Degrees of freedom (nu) shows expected high uncertainty but remains well-calibrated
4. No computational pathologies

The wide nu posteriors should be interpreted as an honest reflection of what the data can tell us about tail behavior, not as a model failure.

## Files Generated

- `run_recovery.py`: Recovery test script
- `recovery_results.csv`: Detailed results for all simulations
- `recovery_summary.json`: Machine-readable summary
- `recovery_intervals.png`: CI coverage visualization
- `recovery_scatter.png`: Posterior mean vs true scatter plot
- `recovery_zscore.png`: Z-score calibration check
- `recovery_nu_focus.png`: Detailed nu recovery visualization
