# Auto-MPG Exploratory Data Analysis Report

## Executive Summary

This report presents an exploratory analysis of the Auto-MPG dataset containing 398 vehicles from model years 1970-1982. The primary goal is to understand the data structure and inform Bayesian modeling of fuel efficiency (MPG).

Key findings:
- MPG is right-skewed with a positive trend over time (+1.2 mpg/year); log-transformation improves normality
- Weight is the strongest predictor (r = -0.83), with nonlinear relationships suggesting quadratic or inverse terms
- Severe multicollinearity among engine size variables (displacement, horsepower, weight) with VIF > 10
- Strong grouping effects by origin (Japan/Europe vs USA ~10 mpg difference) and cylinders
- Only 6 missing horsepower values (1.5%); missingness appears random (MCAR)
- Few outliers; data quality is excellent

## Data Overview

**Shape**: 398 observations, 9 variables

| Column | Type | Description |
|--------|------|-------------|
| mpg | continuous | Target: miles per gallon (9.0 - 46.6) |
| cylinders | discrete | 3, 4, 5, 6, or 8 (mostly 4, 6, 8) |
| displacement | continuous | Engine size in cubic inches (68 - 455) |
| horsepower | continuous | Engine power (46 - 230), 6 missing |
| weight | continuous | Vehicle weight in lbs (1613 - 5140) |
| acceleration | continuous | 0-60 mph time in seconds (8.0 - 24.8) |
| model_year | ordinal | Year 1970-1982 (coded 70-82) |
| origin | categorical | 1=USA, 2=Europe, 3=Japan |
| car_name | string | Vehicle identifier |

## Data Quality Assessment

Data quality is excellent with minimal issues:

- **Missingness**: Only horsepower has missing values (6 observations, 1.5%). The missing cases are distributed across years (1971-1982) and origins (4 USA, 2 Europe). No systematic pattern detected in Mann-Whitney tests against other variables (all p > 0.05), suggesting Missing Completely At Random (MCAR).

- **Duplicates**: No duplicate rows. Some car names repeat (93 duplicates), representing different model years of the same vehicle.

- **Value ranges**: All values fall within plausible ranges. No zeros or negative values in numeric columns.

- **Type issues**: None. Data parsed correctly with horsepower "?" values treated as missing.

See `quality_summary.csv` for the complete quality table.

## Target Variable: MPG

MPG ranges from 9.0 to 46.6 with mean 23.5 and median 23.0.

**Distribution characteristics**:
- Moderate right skew (skewness = 0.46)
- Platykurtic (kurtosis = -0.52), slightly flatter than normal
- Shapiro-Wilk test rejects normality (p < 0.001)
- Log-transformation reduces skewness to -0.14 and improves normality

As shown in `mpg_distribution.png`, the Q-Q plot reveals heavier right tail than normal. The log-transformed distribution shows better normal approximation.

**Boundary behavior**:
- 7 observations below 12 mpg (mostly large American V8s)
- 9 observations above 40 mpg (mostly Japanese/European 4-cylinders from 1980+)
- No evidence of censoring or truncation; these appear to be genuine extreme values

**Implications for modeling**: A lognormal or gamma likelihood may be more appropriate than normal. Alternatively, model log(MPG) with normal errors.

## Predictor Relationships

### Correlation Structure

Correlations with MPG (all highly significant, p < 0.001):

| Predictor | Pearson r | Spearman rho |
|-----------|-----------|--------------|
| weight | -0.83 | -0.88 |
| displacement | -0.80 | -0.86 |
| horsepower | -0.78 | -0.85 |
| model_year | +0.58 | +0.57 |
| acceleration | +0.42 | +0.44 |

Weight, displacement, and horsepower are all strongly negatively correlated with MPG. Spearman correlations are slightly stronger, suggesting monotonic but potentially nonlinear relationships.

### Multicollinearity

Severe multicollinearity exists among engine size variables:

| Pair | Correlation |
|------|-------------|
| displacement - weight | 0.93 |
| displacement - horsepower | 0.90 |
| horsepower - weight | 0.87 |

Variance Inflation Factors:
- displacement: VIF = 10.7 (HIGH)
- weight: VIF = 10.3 (HIGH)
- horsepower: VIF = 8.8 (HIGH)
- acceleration: VIF = 2.6 (acceptable)

**Modeling implication**: Do not include all three engine size variables simultaneously. Either:
1. Use weight alone (strongest predictor, most interpretable)
2. Use a single composite (e.g., PCA on engine variables)
3. Use regularizing priors to handle collinearity

### Nonlinearity

Residual plots in `residual_plots.png` show systematic curvature for weight, displacement, and horsepower. Quadratic fits improve R-squared:

| Predictor | Linear R2 | Quadratic R2 | Improvement |
|-----------|-----------|--------------|-------------|
| horsepower | 0.61 | 0.69 | +0.08 |
| displacement | 0.65 | 0.69 | +0.04 |
| weight | 0.69 | 0.71 | +0.02 |

Curvature is strongest for horsepower. Consider: log(horsepower), 1/horsepower, or polynomial terms.

## Grouping Structures

### Origin Effects

As shown in `grouping_temporal.png`, origin creates meaningful clusters:

| Origin | N | Mean MPG | SD |
|--------|---|----------|-----|
| USA | 249 | 20.1 | 6.4 |
| Europe | 70 | 27.9 | 6.7 |
| Japan | 79 | 30.5 | 6.1 |

One-way ANOVA: F = 98.5, p < 0.001
Effect size (eta-squared) = 0.33 (large effect)

Japanese and European cars average 10 mpg more than American cars. This reflects differences in vehicle size and market preferences.

### Cylinder Effects

Cylinders strongly segment the data:

| Cylinders | N | Mean MPG |
|-----------|---|----------|
| 4 | 204 | 29.3 |
| 6 | 84 | 20.0 |
| 8 | 103 | 15.0 |

The 3-cylinder (4 obs) and 5-cylinder (3 obs) categories are too sparse for reliable inference.

### Origin-Cylinder Confounding

Cross-tabulation reveals structural zeros:

- All 8-cylinder cars are American (n=103)
- No European or Japanese 8-cylinders in the data
- 4-cylinder cars are roughly evenly distributed across origins

This confounding means origin and cylinder effects cannot be fully separated. Including both in a model requires careful interpretation.

## Temporal Patterns

MPG increased substantially over the study period:

- Linear trend: +1.22 mpg per year (r = 0.58, p < 0.001)
- Sharp increase in 1980: mean MPG jumped from 25.1 (1979) to 33.7 (1980)
- Post-1980 levels stabilized around 31 mpg

This likely reflects:
1. Oil crisis effects (1973, 1979)
2. CAFE standards implementation
3. Shift toward smaller, more efficient vehicles

The trend is consistent across origins:
- USA: +1.12 mpg/year
- Europe: +1.03 mpg/year
- Japan: +0.95 mpg/year

American cars showed the largest absolute improvement, partially closing the gap with imports.

## Outliers and Unusual Observations

### Statistical Outliers

- **IQR method**: Only 1 MPG outlier (46.6 mpg, VW Rabbit Diesel)
- **Z-score > 3**: 5 horsepower outliers (all > 220 hp), 2 acceleration outliers
- **Mahalanobis distance**: No multivariate outliers at alpha = 0.001

### Unusual Observations

**High-MPG V8s** (4 observations > 20 mpg):
These are late-model (1978-1981) American cars with lower horsepower, likely representing downsized V8s responding to fuel economy regulations.

**Heavy 4-cylinders** (7 observations > 3000 lbs):
Mostly European diesel sedans (Peugeot, Mercedes, Volvo) which achieve good MPG despite weight.

These observations are not errors but represent real-world variation. They may be influential in models and warrant attention.

## Data-Generating Process Hypotheses

### Hypothesis 1: Physical Efficiency Model

MPG is fundamentally determined by physics: energy required to move mass and overcome drag. This suggests:

```
MPG = f(weight, aerodynamics) + engine_efficiency + noise
```

Weight should dominate. The relationship should be approximately inverse (gallons/mile ~ weight). Model: `log(MPG) ~ log(weight)` or `MPG ~ 1/weight`.

**Evidence**: Weight has highest correlation (-0.83); nonlinear residual patterns support inverse relationship.

### Hypothesis 2: Market Segmentation Model

Vehicles are designed for market segments with different priorities (economy vs. performance). This suggests:

```
MPG = baseline(segment) + within-segment_variation
```

Origin and cylinders define segments. Model: Hierarchical with group-level intercepts.

**Evidence**: Origin explains 33% of variance (eta-squared); clear separation between Japanese/European and American cars.

### Hypothesis 3: Technological Progress Model

Fuel efficiency improved over time due to technology and regulation. This suggests:

```
MPG = time_trend + vehicle_characteristics + noise
```

Model year should have independent predictive power beyond physical characteristics.

**Evidence**: Even after controlling for weight, model_year has significant partial correlation with MPG.

### Recommended Synthesis

Combine all three mechanisms:

```
log(MPG) ~ 1 + log(weight) + model_year + (1 | origin)
```

Or equivalently:
- Normal likelihood on log(MPG)
- Log-weight as primary predictor (inverse physical relationship)
- Model year for temporal trend
- Hierarchical origin intercepts for market segmentation

## Modeling Recommendations

### Likelihood Choice

| Option | Rationale | Complexity |
|--------|-----------|------------|
| Normal on log(MPG) | Best normality; interpretable as multiplicative effects | Low |
| Lognormal | Equivalent to above; respects positivity | Low |
| Gamma | Flexible for positive continuous; handles right skew | Medium |
| Normal on MPG | Simpler but worse residual behavior | Low |

**Recommendation**: Start with Normal on log(MPG).

### Predictor Selection

**Include**:
- `log(weight)` - primary predictor, physically meaningful
- `model_year` (centered) - captures temporal trend
- `origin` - as categorical or hierarchical grouping

**Consider**:
- `cylinders` - but confounded with origin; may be redundant with weight
- `horsepower` - strong predictor but collinear with weight

**Avoid**:
- Including weight, displacement, and horsepower together (VIF > 10)
- Using acceleration alone (weak predictor, r = 0.42)

### Scale and Centering Notes

For setting priors:

| Variable | Mean | SD | Suggested Transform |
|----------|------|-----|---------------------|
| log(MPG) | 3.09 | 0.33 | None (already on log scale) |
| weight | 2970 | 847 | Center and scale by 1000 |
| log(weight) | 7.96 | 0.28 | Center at mean |
| model_year | 76.0 | 3.7 | Center at 76 (midpoint) |

Typical MPG magnitude: 10-45 mpg, log scale: 2.3-3.8

### Recommended Starting Model

```stan
data {
  int<lower=0> N;
  vector[N] log_mpg;           // log(mpg)
  vector[N] log_weight_c;      // log(weight) - mean(log(weight))
  vector[N] year_c;            // model_year - 76
  array[N] int<lower=1,upper=3> origin;
}
parameters {
  real alpha;                  // intercept
  real beta_weight;            // effect of log(weight)
  real beta_year;              // effect of year
  vector[3] alpha_origin;      // origin intercepts
  real<lower=0> sigma;         // residual SD
}
model {
  alpha ~ normal(3.1, 0.5);
  beta_weight ~ normal(-1, 0.5);    // expect negative, ~elasticity
  beta_year ~ normal(0.05, 0.03);   // ~1.2 mpg/year on original scale
  alpha_origin ~ normal(0, 0.3);
  sigma ~ exponential(3);

  log_mpg ~ normal(alpha + alpha_origin[origin] +
                   beta_weight * log_weight_c +
                   beta_year * year_c, sigma);
}
```

### Extensions to Consider

1. **Interactions**: year x origin to capture differential trends
2. **Varying slopes**: Allow weight effect to vary by origin
3. **Student-t errors**: For robustness to outliers
4. **Missing data model**: Jointly model horsepower missingness (though likely MCAR)

## Output Files

| File | Description |
|------|-------------|
| `quality_summary.csv` | Per-column data quality metrics |
| `univariate_summary.csv` | Descriptive statistics for each variable |
| `mpg_distribution.png` | Target variable distribution diagnostics |
| `scatterplot_matrix.png` | Pairwise relationships among key variables |
| `mpg_vs_predictors.png` | MPG scatterplots with linear fits |
| `residual_plots.png` | Residuals from linear fits (nonlinearity check) |
| `grouping_temporal.png` | Origin/cylinder boxplots and temporal trends |
| `outliers_influence.png` | Outlier diagnostics and influential points |
| `01_data_loading.py` | Data loading and quality checks |
| `02_univariate_analysis.py` | Univariate statistics and target profiling |
| `03_relationships.py` | Correlation and relationship analysis |
| `04_grouping_temporal.py` | Grouping structure and temporal patterns |
| `05_missing_outliers.py` | Missing data and outlier analysis |

## Caveats

1. **Temporal confounding**: Vehicle characteristics changed systematically over time. Year effects may partly reflect changes in the vehicle mix.

2. **Selection effects**: This dataset represents vehicles sold in the US market. Generalization to other markets or time periods requires caution.

3. **Measurement era**: EPA testing procedures from 1970-1982 differ from modern methods. Absolute MPG values may not be directly comparable to current ratings.

4. **Observational data**: Causal interpretation of coefficients requires domain assumptions. Weight is likely causal; origin effects may partly reflect unmeasured confounders.
