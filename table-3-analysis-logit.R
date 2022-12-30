#--------------------Replication of Morris et al. (2022)------------------------
#-Author: A. Jordan Nafa----------------------------Created: December 29, 2022-#
#-R Version: 4.2.1----------------------------------Revised: December 29, 2022-#

## Load the necessary libraries
pacman::p_load(
  "tidyverse", 
  "brms", 
  "cmdstanr",
  "arrow",
  "haven",
  "sjlabelled",
  install = FALSE
)

# Load the data
df <- read_dta("data/morris et al (2022)_data.dta") %>% 
  mutate(
    # Apply value labels to covariates
    across(
      group:ethnicgrp, 
      ~ factor(
        .x,
        levels = get_values(.x),
        labels = get_labels(.x)
        )),
    # generate a missingness vector
    missing = is.na(anytest)
    )

#------------------------------------------------------------------------------#
#----------------Inverse Probability Weights for Missing Data-------------------
#------------------------------------------------------------------------------#

# Model for the denominator of the IPTW weights
denom_logit <- bf(
  missing ~ gender*group + age*group + partners*group + msm*group + 
    ethnicgrp*group,
  family = brmsfamily("bernoulli", link = "logit")
)

# Priors for the parameters
propensity_priors <- prior(normal(0, 1), class = b) +
  prior(normal(0, 1.5), class = Intercept)
  
# Logistic regression for the missing outcome
bayes_missing_logit <- brm(
  formula = denom_logit,
  prior = propensity_priors,
  data = df,
  cores = 6, # Number of cores to use for parallel chains
  chains = 6, # Number of chains, should be at least 4
  iter = 15e3, # Total iterations = Warm-Up + Sampling
  thin = 5, # Keep every fifth post warm-up draw
  warmup = 5e3, # Warm-Up Iterations
  refresh = 5e3, # Print progress every 5k iterations
  control = list(max_treedepth = 12),
  save_pars = save_pars(all = TRUE),
  backend = "cmdstanr", # Requires cmdstanr and cmdstan be installed
  file = "output/fits/propensity_model_logit.rds"
)

# Generate predicted probability of complete case
miss_preds <- posterior_epred(bayes_missing_logit)

# Posterior Distribution of inverse probability of non-missingness weights
miss_preds_ipw <- apply(miss_preds, MARGIN = 2, function(x){1 / (1 - x)})

# Add the weights and probabilities to the data
df_preds <- df %>% 
  mutate(
    # Probability of being missing
    ipmw_mean = colMeans(miss_preds_ipw),
    # Uncertainty in the missingness
    ipmw_sd = apply(miss_preds_ipw, MARGIN = 2, sd)
    )

# Data for the outcome models
models_df <- df_preds %>% 
  # Subset the relevant columns
  select(anytest:anytreat, group:ethnicgrp, age, ipmw_mean:ipmw_sd) %>% 
  # Rescale age by a numeric constant
  mutate(age = age/5) %>% 
  # Complete cases
  drop_na()

#------------------------------------------------------------------------------#
#-----------------------IPTW, Complete Cases Models-----------------------------
#------------------------------------------------------------------------------#

# Compile the Stan model
ipwm_outcome_mod <- cmdstan_model("output/fits/bernoulli_ipwm.stan")

# Design matrix
x_data <- model.matrix(
  ~ group + gender + age + partners + msm + ethnicgrp,
  data = models_df
)

# List object with the data fro Stan
test_stan_data <- list(
  N = nrow(x_data),
  Y = models_df$anytest,
  X = x_data[, 2:20],
  K = ncol(x_data[, 2:20]),
  treat_pos = 1,
  lambda = models_df$ipmw_mean,
  delta = models_df$ipmw_sd,
  sd_prior_shape1 = 2,
  sd_prior_shape2 = 2
)

# Fit the Outcome-Stage Model
ipwm_outcome_test_fit <- ipwm_outcome_mod$sample(
  data = test_stan_data,
  refresh = 100,
  sig_figs = 5,
  parallel_chains = 6,
  chains = 6,
  iter_warmup = 3000,
  iter_sampling = 5000,
  max_treedepth = 13,
  adapt_delta = 0.9
)

# Write the fit to the disk
ipwm_outcome_test_fit$save_object(
  file = "output/fits/outcome_ipwm_test_model_logit.rds"
  )

# Extract the draws
ipwm_test_draws <- ipwm_outcome_test_fit$draws(
  variables = c("ATE", "EYX1", "EYX0", "b", "b_Intercept", "w_tilde"), 
  format = "draws_df"
  ) %>% 
  # Calculate the log risk ratios
  mutate(logrr = log(EYX1/EYX0), .before = 2)

# Write the draws to a parquet file
write_parquet(
  ipwm_test_draws,
  "output/predictions/test_ipwm_posterior_draws.gz.parquet",
  compression = "gzip",
  compression_level = 7
)

# List object with the data for Stan
treat_stan_data <- list(
  N = nrow(x_data),
  Y = models_df$anytreat,
  X = x_data[, 2:20],
  K = ncol(x_data[, 2:20]),
  treat_pos = 1,
  lambda = models_df$ipmw_mean,
  delta = models_df$ipmw_sd,
  sd_prior_shape1 = 2,
  sd_prior_shape2 = 2
)

# Fit the Outcome-Stage Model
ipwm_outcome_treat_fit <- ipwm_outcome_mod$sample(
  data = treat_stan_data,
  refresh = 100,
  sig_figs = 5,
  parallel_chains = 6,
  chains = 6,
  iter_warmup = 3000,
  iter_sampling = 5000,
  max_treedepth = 13,
  adapt_delta = 0.9
)

# Write the fit to the disk
ipwm_outcome_treat_fit$save_object(
  file = "output/fits/outcome_ipwm_treat_model_logit.rds"
)

# Extract the draws
ipwm_treat_draws <- ipwm_outcome_treat_fit$draws(
  variables = c("ATE", "EYX1", "EYX0", "b", "b_Intercept", "w_tilde"), 
  format = "draws_df"
) %>% 
  # Calculate the log risk ratios
  mutate(logrr = log(EYX1/EYX0), .before = 2)

# Write the draws to a parquet file
write_parquet(
  ipwm_treat_draws,
  "output/predictions/treat_ipwm_posterior_draws.gz.parquet",
  compression = "gzip",
  compression_level = 7
)

#------------------------------------------------------------------------------#
#--------------Standardized Differences, Complete Cases Models------------------
#------------------------------------------------------------------------------#

# Compile the Stan model
std_outcome_mod <- cmdstan_model("output/fits/bernoulli_logit.stan")

# List object with the data for Stan
test_std_stan_data <- list(
  N = nrow(x_data),
  Y = models_df$anytest,
  X = x_data[, 2:20],
  K = ncol(x_data[, 2:20]),
  treat_pos = 1
)

# Fit the Outcome-Stage Model
std_outcome_test_fit <- std_outcome_mod$sample(
  data = test_std_stan_data,
  refresh = 100,
  sig_figs = 5,
  parallel_chains = 6,
  chains = 6,
  iter_warmup = 3000,
  iter_sampling = 5000,
  max_treedepth = 13,
  adapt_delta = 0.9
)

# Write the fit to the disk
std_outcome_test_fit$save_object(
  file = "output/fits/outcome_std_test_model_logit.rds"
)

# Extract the draws
std_test_draws <- std_outcome_test_fit$draws(
  variables = c("ATE", "EYX1", "EYX0", "b", "b_Intercept"), 
  format = "draws_df"
)

# Write the draws to a parquet file
write_parquet(
  std_test_draws,
  "output/predictions/test_std_posterior_draws.gz.parquet",
  compression = "gzip",
  compression_level = 7
)

# List object with the data for Stan
treat_std_stan_data <- list(
  N = nrow(x_data),
  Y = models_df$anytreat,
  X = x_data[, 2:20],
  K = ncol(x_data[, 2:20]),
  treat_pos = 1
)

# Fit the Outcome-Stage Model
std_outcome_treat_fit <- std_outcome_mod$sample(
  data = treat_std_stan_data,
  refresh = 100,
  sig_figs = 5,
  parallel_chains = 6,
  chains = 6,
  iter_warmup = 3000,
  iter_sampling = 5000,
  max_treedepth = 13,
  adapt_delta = 0.9
)

# Write the fit to the disk
std_outcome_treat_fit$save_object(
  file = "output/fits/outcome_std_treat_model_logit.rds"
)

# Extract the draws
std_treat_draws <- std_outcome_treat_fit$draws(
  variables = c("ATE", "EYX1", "EYX0", "b", "b_Intercept"), 
  format = "draws_df"
)

# Write the draws to a parquet file
write_parquet(
  std_treat_draws,
  "output/predictions/treat_std_posterior_draws.gz.parquet",
  compression = "gzip",
  compression_level = 7
)
