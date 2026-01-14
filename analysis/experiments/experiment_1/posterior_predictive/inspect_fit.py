"""Inspect posterior InferenceData structure."""

from shared_utils import load_posterior

idata = load_posterior('/home/user/claude-code-devcontainer/analysis/experiments/experiment_1/fit')

print('Groups:', list(idata.groups()))
print('\nPosterior variables:', list(idata.posterior.data_vars))
if 'observed_data' in idata.groups():
    print('Observed data variables:', list(idata.observed_data.data_vars))
if 'posterior_predictive' in idata.groups():
    print('Posterior predictive variables:', list(idata.posterior_predictive.data_vars))

print(f'\nDimensions: {dict(idata.posterior.dims)}')
print(f'Number of observations: {idata.posterior.dims.get("obs_id", "N/A")}')
