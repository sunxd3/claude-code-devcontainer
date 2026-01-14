# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas"]
# ///
"""Generate synthetic hierarchical data for Bayesian modeling demonstration.

This creates a dataset simulating student test scores across multiple schools,
with a treatment effect and school-level variation - a classic setup for
hierarchical/multilevel modeling.

Data generating process:
- J schools, each with n_j students
- School-level intercepts (random effects)
- Treatment effect (some students received intervention)
- Student-level noise

True parameters:
- mu (grand mean): 70
- tau (school SD): 8
- theta_j (school effects): drawn from N(mu, tau)
- beta (treatment effect): 5
- sigma (residual SD): 10
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# True parameters
MU = 70.0  # Grand mean
TAU = 8.0  # Between-school SD
BETA = 5.0  # Treatment effect
SIGMA = 10.0  # Within-school (residual) SD

# Study design
N_SCHOOLS = 8
STUDENTS_PER_SCHOOL = (15, 20, 25, 18, 22, 17, 19, 24)  # Unbalanced


def generate_data():
    """Generate the hierarchical dataset."""
    # School-level effects
    theta = np.random.normal(MU, TAU, N_SCHOOLS)

    records = []
    for j, (school_effect, n_students) in enumerate(
        zip(theta, STUDENTS_PER_SCHOOL), start=1
    ):
        # ~50% of students in each school get treatment
        treatment = np.random.binomial(1, 0.5, n_students)

        # Generate outcomes
        y = (
            school_effect
            + BETA * treatment
            + np.random.normal(0, SIGMA, n_students)
        )

        for i, (yi, ti) in enumerate(zip(y, treatment), start=1):
            records.append(
                {
                    "student_id": f"S{j:02d}_{i:03d}",
                    "school_id": j,
                    "school_name": f"School_{j}",
                    "treatment": int(ti),
                    "score": round(yi, 1),
                }
            )

    df = pd.DataFrame(records)
    return df, theta


def main():
    output_dir = Path(__file__).parent

    # Generate data
    df, true_theta = generate_data()

    # Save as CSV
    df.to_csv(output_dir / "student_scores.csv", index=False)

    # Save as JSON for Stan
    stan_data = {
        "N": len(df),
        "J": N_SCHOOLS,
        "school": df["school_id"].tolist(),
        "treatment": df["treatment"].tolist(),
        "y": df["score"].tolist(),
    }
    with open(output_dir / "stan_data.json", "w") as f:
        json.dump(stan_data, f, indent=2)

    # Save true parameters for validation
    true_params = {
        "mu": MU,
        "tau": TAU,
        "beta": BETA,
        "sigma": SIGMA,
        "theta": true_theta.tolist(),
    }
    with open(output_dir / "true_parameters.json", "w") as f:
        json.dump(true_params, f, indent=2)

    # Summary statistics
    print("Data Generation Complete")
    print("=" * 50)
    print(f"Total students: {len(df)}")
    print(f"Schools: {N_SCHOOLS}")
    print(f"Students per school: {dict(df.groupby('school_id').size())}")
    print("\nTreatment split:")
    print(df.groupby("treatment")["score"].agg(["count", "mean", "std"]))
    print("\nSchool means:")
    print(df.groupby("school_id")["score"].agg(["count", "mean", "std"]))
    print(f"\nTrue school effects (theta): {true_theta.round(1)}")
    print(f"\nFiles saved to: {output_dir}")


if __name__ == "__main__":
    main()
