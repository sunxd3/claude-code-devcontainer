# Parameter Recovery Assessment: A1-Baseline Model

## Summary

**Result: PASS**

The A1-Baseline model successfully recovers known parameters from synthetic data. All convergence criteria were met and the recovery rate (93%) exceeds the 80% threshold.

## Model Description

The A1-Baseline model is a simple log-log regression: `log(mpg) ~ log(weight)`. Parameters:
- `alpha`: intercept (log-MPG at mean log-weight)
- `beta_weight`: elasticity of MPG with respect to weight
- `sigma`: residual standard deviation on log scale

## Test Configuration

| Setting | Value |
|---------|-------|
| Number of recovery tests | 5 |
| Observations per test | 392 |
| True alpha | 3.1 |
| True beta_weight | -0.9 |
| True sigma | 0.25 |
| Predictor SD (log_weight_c) | 0.28 |

## Results

### Convergence Diagnostics

All 5 tests passed convergence criteria:

| Metric | Threshold | Achieved |
|--------|-----------|----------|
| Max R-hat | < 1.01 | 1.00 (all tests) |
| Min ESS bulk | > 400 | 3308-3921 |
| Min ESS tail | > 400 | 2522-2945 |
| Divergences | 0 | 0 (all tests) |

The model samples efficiently with no computational issues.

### Parameter Recovery

Recovery within 90% HDI:

| Parameter | Recovered | Rate |
|-----------|-----------|------|
| alpha | 4/5 | 80% |
| beta_weight | 5/5 | 100% |
| sigma | 5/5 | 100% |
| **Total** | **14/15** | **93%** |

The single miss for alpha in Test 5 is expected sampling variability (90% CI will miss ~10% of the time by design). The z-score for this case was -1.69, not extreme.

### Bias Assessment

Average bias across tests:

| Parameter | Mean Bias | Mean |z-score| |
|-----------|-----------|----------------|
| alpha | -0.009 | 0.73 |
| beta_weight | +0.002 | 0.36 |
| sigma | +0.003 | 0.66 |

All biases are small relative to posterior uncertainty (z-scores well within 2), indicating unbiased recovery.

## Visual Evidence

Three diagnostic plots were generated:

1. **`recovery_scatter.png`**: Posterior means vs true values with 90% CIs. Points track near the identity line.

2. **`recovery_intervals.png`**: Interval plot showing true values (dashed) with posterior credible intervals. Green intervals contain the true value; the single red interval (alpha, Test 5) represents expected coverage error.

3. **`trace_example.png`**: Trace plots from Test 1 showing good chain mixing (well-mixed caterpillars).

## Conclusion

The model is computationally stable, converges reliably, and recovers known parameters without systematic bias. The 93% empirical coverage is consistent with the expected 90% nominal coverage.

**Recommendation**: Proceed to fitting real data.

## Files Generated

- `run_recovery.py`: Parameter recovery test script
- `recovery_results.json`: Detailed results in machine-readable format
- `recovery_scatter.png`: Posterior mean vs true scatter plot
- `recovery_intervals.png`: Credible interval coverage plot
- `trace_example.png`: Example trace plot
- `report.md`: This assessment report
