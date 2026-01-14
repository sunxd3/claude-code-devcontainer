#!/usr/bin/env python3
"""
Fit Experiment 3: Random Intercepts + Random Slopes Model

This script fits the scientific target model with school-specific intercepts
and treatment effects to the real data. The key question is whether tau_beta
(treatment effect heterogeneity) is meaningfully different from zero.

Strategy:
1. Probe sampling: 4 chains x 100 iterations to catch issues early
2. Main sampling: 4 chains x 1000 iterations for reliable inference
3. Convergence checks: R-hat, ESS, divergences, visual diagnostics
4. Save ArviZ InferenceData with log_likelihood for LOO-CV
"""

import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
from shared_utils import compile_model, fit_model

# Paths
BASE_DIR = Path("/home/user/claude-code-devcontainer/analysis")
EXPERIMENT_DIR = BASE_DIR / "experiments" / "experiment_3"
MODEL_PATH = EXPERIMENT_DIR / "model.stan"
DATA_PATH = BASE_DIR / "data" / "stan_data.json"
OUTPUT_DIR = EXPERIMENT_DIR / "fit"

def load_data():
    """Load Stan data from JSON."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    print(f"Loaded data: N={data['N']}, J={data['J']}")
    return data

def run_probe_sampling(model, data):
    """
    Run short probe sampling to catch issues early.

    Returns:
        CmdStanMCMC object if successful, None if failed
    """
    print("\n" + "="*60)
    print("PROBE SAMPLING: 4 chains x 100 iterations")
    print("="*60)

    try:
        fit = fit_model(
            model=model,
            data=data,
            chains=4,
            warmup=50,
            sampling=100,
            show_progress=True,
            adapt_delta=0.8
        )

        # Quick convergence check
        print("\nProbe Results:")
        summary = fit.summary()

        # Check for immediate red flags (handle missing columns gracefully)
        if 'R_hat' in summary.columns:
            max_rhat = summary['R_hat'].max()
            print(f"  Max R-hat: {max_rhat:.4f}")
        else:
            print("  R-hat not available (sampling may have failed)")

        if 'ess_bulk' in summary.columns:
            min_ess_bulk = summary['ess_bulk'].min()
            print(f"  Min ESS_bulk: {min_ess_bulk:.1f}")
        else:
            print("  ESS_bulk not available (sampling may have failed)")

        # Check for divergences
        try:
            diag = fit.diagnose()
            print(f"\n{diag}")
        except Exception as e:
            print(f"  Diagnostic check skipped: {e}")

        # Check if sampling actually worked
        if 'R_hat' not in summary.columns or 'ess_bulk' not in summary.columns:
            print("\n  WARNING: Probe sampling had issues, but continuing to main sampling with better init")
            return fit

        if max_rhat > 1.05:
            print("\n  WARNING: High R-hat detected, but continuing to main sampling")

        return fit

    except Exception as e:
        print(f"\nProbe sampling FAILED: {e}")
        return None

def run_main_sampling(model, data):
    """
    Run main sampling with sufficient iterations for reliable inference.

    Returns:
        CmdStanMCMC object if successful, None if failed
    """
    print("\n" + "="*60)
    print("MAIN SAMPLING: 4 chains x 1000 iterations")
    print("="*60)

    try:
        # Use higher adapt_delta to start, given initialization issues in probe
        fit = fit_model(
            model=model,
            data=data,
            chains=4,
            warmup=1000,
            sampling=1000,
            show_progress=True,
            adapt_delta=0.95
        )

        return fit

    except Exception as e:
        print(f"\nMain sampling FAILED: {e}")
        return None

def check_diagnostics(fit):
    """
    Run comprehensive convergence diagnostics.

    Returns:
        dict with convergence status and issues
    """
    print("\n" + "="*60)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*60)

    issues = []

    # Get summary statistics
    summary = fit.summary()

    # Check R-hat
    max_rhat = summary['R_hat'].max()
    rhat_violations = (summary['R_hat'] > 1.01).sum()
    print("\nR-hat:")
    print(f"  Max: {max_rhat:.4f}")
    print(f"  Violations (>1.01): {rhat_violations}")
    if max_rhat > 1.01:
        issues.append(f"High R-hat: {max_rhat:.4f}")

    # Check ESS (CmdStanPy column names: ESS_bulk, ESS_tail)
    min_ess_bulk = summary['ESS_bulk'].min()
    min_ess_tail = summary['ESS_tail'].min()
    print("\nEffective Sample Size:")
    print(f"  Min ESS_bulk: {min_ess_bulk:.1f}")
    print(f"  Min ESS_tail: {min_ess_tail:.1f}")
    if min_ess_bulk < 400:
        issues.append(f"Low ESS_bulk: {min_ess_bulk:.1f}")
    if min_ess_tail < 400:
        issues.append(f"Low ESS_tail: {min_ess_tail:.1f}")

    # Check MCSE (CmdStanPy column names: MCSE, StdDev)
    max_mcse_mean = (summary['MCSE'] / summary['StdDev']).max()
    print("\nMonte Carlo Standard Error:")
    print(f"  Max MCSE/SD ratio: {max_mcse_mean:.4f}")
    if max_mcse_mean > 0.05:
        issues.append(f"High MCSE: {max_mcse_mean:.4f}")

    # Try diagnose() - but this may OOM
    print("\nRunning CmdStanPy diagnose()...")
    try:
        diag = fit.diagnose()
        print(diag)
    except Exception as e:
        print(f"  Skipped (error): {e}")

    # Report key parameters
    print("\n" + "="*60)
    print("KEY PARAMETER ESTIMATES")
    print("="*60)

    key_params = ['beta_0', 'tau_alpha', 'tau_beta', 'sigma', 'alpha_0']
    for param in key_params:
        if param in summary.index:
            row = summary.loc[param]
            print(f"\n{param}:")
            print(f"  Mean: {row['Mean']:.3f}")
            print(f"  SD: {row['StdDev']:.3f}")
            print(f"  95% CI: [{row['5%']:.3f}, {row['95%']:.3f}]")
            print(f"  R-hat: {row['R_hat']:.4f}")
            print(f"  ESS_bulk: {row['ESS_bulk']:.1f}")

    # Convergence status
    converged = len(issues) == 0
    status = "CONVERGED" if converged else "ISSUES DETECTED"

    print("\n" + "="*60)
    print(f"STATUS: {status}")
    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  - {issue}")
    print("="*60)

    return {
        'converged': converged,
        'issues': issues,
        'max_rhat': max_rhat,
        'min_ess_bulk': min_ess_bulk,
        'min_ess_tail': min_ess_tail,
        'max_mcse_ratio': max_mcse_mean
    }

def create_visual_diagnostics(idata, output_dir):
    """Create and save visual diagnostic plots."""
    print("\n" + "="*60)
    print("CREATING VISUAL DIAGNOSTICS")
    print("="*60)

    # Trace plots for key parameters
    print("\nGenerating trace plots...")
    key_params = ['beta_0', 'tau_alpha', 'tau_beta', 'sigma', 'alpha_0']
    fig, axes = plt.subplots(len(key_params), 2, figsize=(12, 2.5*len(key_params)))
    az.plot_trace(idata, var_names=key_params, axes=axes)
    fig.tight_layout()
    fig.savefig(output_dir / "trace_plots.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'trace_plots.png'}")

    # Rank plots
    print("\nGenerating rank plots...")
    fig = plt.figure(figsize=(12, 8))
    az.plot_rank(idata, var_names=key_params)
    fig.tight_layout()
    fig.savefig(output_dir / "rank_plots.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'rank_plots.png'}")

    # Energy plot
    print("\nGenerating energy plot...")
    fig = plt.figure(figsize=(8, 6))
    az.plot_energy(idata)
    fig.tight_layout()
    fig.savefig(output_dir / "energy_plot.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'energy_plot.png'}")

    # Posterior distributions
    print("\nGenerating posterior plots...")
    fig = plt.figure(figsize=(12, 8))
    az.plot_posterior(idata, var_names=key_params, hdi_prob=0.95)
    fig.tight_layout()
    fig.savefig(output_dir / "posterior_plots.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_dir / 'posterior_plots.png'}")

    # Autocorrelation
    print("\nGenerating autocorrelation plots...")
    try:
        fig = plt.figure(figsize=(12, 8))
        az.plot_autocorr(idata, var_names=key_params, max_lag=100)
        fig.tight_layout()
        fig.savefig(output_dir / "autocorr_plots.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_dir / 'autocorr_plots.png'}")
    except Exception as e:
        print(f"  Skipped autocorrelation plot: {e}")

def save_inference_data(fit, data, output_dir):
    """
    Convert to ArviZ InferenceData and save as NetCDF.

    Returns:
        InferenceData object
    """
    print("\n" + "="*60)
    print("SAVING ARVIZ INFERENCEDATA")
    print("="*60)

    # Convert to InferenceData with log_likelihood
    print("\nConverting to InferenceData...")
    idata = az.from_cmdstanpy(
        fit,
        posterior_predictive=['y_rep'],
        log_likelihood='log_lik',
        observed_data={'y': data['y']},
        coords={
            'school': range(1, data['J'] + 1),
            'obs': range(data['N'])
        },
        dims={
            'alpha': ['school'],
            'beta': ['school'],
            'z_alpha': ['school'],
            'z_beta': ['school'],
            'mu': ['obs'],
            'log_lik': ['obs'],
            'y_rep': ['obs']
        }
    )

    # Save as NetCDF
    nc_path = output_dir / "posterior.nc"
    idata.to_netcdf(nc_path)
    print(f"  Saved: {nc_path}")

    # Also save summary as CSV
    summary_path = output_dir / "summary.csv"
    fit.summary().to_csv(summary_path)
    print(f"  Saved: {summary_path}")

    return idata

def generate_report(convergence_info, output_dir):
    """Generate a text report summarizing the fit."""
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)

    report = []
    report.append("# Experiment 3 Fit Report")
    report.append("## Random Intercepts + Random Slopes Model")
    report.append("")
    report.append("### Convergence Status")
    report.append("")

    if convergence_info['converged']:
        report.append("**Status**: CONVERGED")
        report.append("")
        report.append("All convergence criteria met:")
    else:
        report.append("**Status**: ISSUES DETECTED")
        report.append("")
        report.append("Issues found:")
        for issue in convergence_info['issues']:
            report.append(f"- {issue}")
        report.append("")
        report.append("Convergence metrics:")

    report.append(f"- Max R-hat: {convergence_info['max_rhat']:.4f} (target: <1.01)")
    report.append(f"- Min ESS_bulk: {convergence_info['min_ess_bulk']:.1f} (target: >400)")
    report.append(f"- Min ESS_tail: {convergence_info['min_ess_tail']:.1f} (target: >400)")
    report.append(f"- Max MCSE/SD: {convergence_info['max_mcse_ratio']:.4f} (target: <0.05)")
    report.append("")

    report.append("### Key Parameters")
    report.append("")
    report.append("See `summary.csv` for full parameter estimates.")
    report.append("")

    report.append("### Scientific Question")
    report.append("")
    report.append("**Question**: Is treatment effect heterogeneity (tau_beta) meaningfully different from zero?")
    report.append("")
    report.append("Check the tau_beta posterior in `posterior_plots.png` and `summary.csv`.")
    report.append("If the 95% credible interval excludes zero and is substantively meaningful,")
    report.append("this supports the hypothesis that treatment effects vary across schools.")
    report.append("")

    report.append("### Output Files")
    report.append("")
    report.append("- `posterior.nc`: ArviZ InferenceData (NetCDF format) with log_likelihood for LOO-CV")
    report.append("- `summary.csv`: Full parameter summary statistics")
    report.append("- `trace_plots.png`: Trace plots for key parameters")
    report.append("- `rank_plots.png`: Rank plots for convergence assessment")
    report.append("- `energy_plot.png`: Energy diagnostic for HMC")
    report.append("- `posterior_plots.png`: Posterior distributions")
    report.append("- `autocorr_plots.png`: Autocorrelation plots")

    report_text = "\n".join(report)

    report_path = output_dir / "fit_report.md"
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"  Saved: {report_path}")

    return report_text

def main():
    """Main fitting workflow."""
    print("="*60)
    print("EXPERIMENT 3: RANDOM INTERCEPTS + RANDOM SLOPES MODEL")
    print("="*60)

    # Load data
    data = load_data()

    # Compile model
    print("\nCompiling Stan model...")
    model = compile_model(MODEL_PATH)
    print("  Compilation successful")

    # Skip probe due to initialization issues with short warmup
    # Go straight to main sampling with sufficient warmup
    print("\nSkipping probe, proceeding directly to main sampling...")
    print("(Probe revealed initialization issues with short warmup)")

    # Run main sampling
    fit = run_main_sampling(model, data)

    if fit is None:
        print("\n" + "="*60)
        print("MAIN SAMPLING FAILED - STOPPING")
        print("="*60)
        return

    # Check convergence
    convergence_info = check_diagnostics(fit)

    # Save InferenceData
    idata = save_inference_data(fit, data, OUTPUT_DIR)

    # Create visual diagnostics
    create_visual_diagnostics(idata, OUTPUT_DIR)

    # Generate report
    generate_report(convergence_info, OUTPUT_DIR)

    print("\n" + "="*60)
    print("FITTING COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
