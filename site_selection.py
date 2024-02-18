import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan

## These are needed when running in a notebook
import nest_asyncio
nest_asyncio.apply()

# Load data set
country = pd.read_parquet("data/country.parquet").set_index("country_id")
target = pd.read_parquet("data/target.parquet").set_index(["trial_id", "site_id"])
site = pd.read_parquet("data/trial_site.parquet")
trial = pd.read_parquet("data/trial.parquet").set_index("trial_id")
df = site.join(trial, on="trial_id").join(target, on=["trial_id", "site_id"]).join(country, on="country_id")

# Define target variable
df["enrolment_speed"] = df["no_of_patients"] / df["enrolment_months"]
df.loc[df["enrolment_speed"] <= 0, "enrolment_speed"] = np.NaN
target_stats = df["enrolment_speed"].describe()
iqr = target_stats["75%"] - target_stats["25%"]
three_sigma_upper = target_stats["75%"] + 1.5 * iqr
three_sigma_lower = max(target_stats["25%"] - 1.5 * iqr, 0)

# Tag sites that need to be imputed
df["impute"] = np.logical_or(np.logical_or(df["enrolment_speed"] < three_sigma_lower, df["enrolment_speed"] > three_sigma_upper), df["enrolment_speed"].isna())

# Feature engineering
features = df[["site_id", "trial_id", "enrolment_speed", "impute", "minimum_age", "who_gho_ghed_che_pc_ppp_sha2011_curr_health_exp_per_capita_ppp"]].rename(columns={
    "who_gho_ghed_che_pc_ppp_sha2011_curr_health_exp_per_capita_ppp": "gdp_pc"
})
features = features[np.logical_not(features["gdp_pc"].isna())]

# Sites with complete data
complete = features[np.logical_not(features["impute"])].copy()
complete["minimum_age"] = complete["minimum_age"] - min(complete["minimum_age"])
max_observed_speed = max(complete["enrolment_speed"])
complete["enrolment_speed"] = complete["enrolment_speed"] / max_observed_speed
max_gdp_pc = max(complete["gdp_pc"])
complete["gdp_pc"] = complete["gdp_pc"] / max_gdp_pc

# Sites to be imputed
incomplete = features[features["impute"]].copy()
incomplete["minimum_age"] = incomplete["minimum_age"] - min(incomplete["minimum_age"])
incomplete["gdp_pc"] = incomplete["gdp_pc"] / max_gdp_pc

# Define a model and its data
data = {
    "N": len(complete),
    "N_inc": len(incomplete),
    "D": 2,
    "y": complete["enrolment_speed"].to_list(),
    "X": complete[["minimum_age", "gdp_pc"]].to_numpy(),
    "X_inc": incomplete[["minimum_age", "gdp_pc"]].to_numpy()
}

stan_code = """
// Bayesian Linear Regression
data {
    int<lower = 0> N;
    int<lower = 0> N_inc;
    vector[N] y; 
    int<lower=0> D;
    matrix[N, D] X;  
    matrix[N_inc, D] X_inc;
}
parameters {
    vector[D] beta;
    real<lower=0> sigma;
}
model {
    sigma ~ exponential(1);
    beta ~ normal(0, 1);
    y ~ normal(X * beta, sigma);
}
generated quantities {
    vector[N_inc] y_imputed;
    vector[N] log_lik;
    for (n in 1:N_inc) {
        y_imputed[n] = X_inc[n, ] * beta;
    }
    for (n in 1:N) {
        log_lik[n] = normal_lpdf(y[n] | X[n, ] * beta, sigma);
    }
}
"""

# Fit our model and impute in one step
posterior = stan.build(
    program_code=stan_code,
    data = data,
    random_seed=4444
)

fit = posterior.sample(num_samples=1000, num_chains=4)

# Approximate Leave One Out Cross Validation
idata = az.from_pystan(posterior=fit, posterior_model=posterior, log_likelihood="log_lik")
az.loo(idata)

# Example distribution for an imputed target value
plt.hist(fit["y_imputed"][0, :])
