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
  "posterior",
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
  ) %>% 
  # Calculate the log risk ratios
  mutate(logrr = log(EYX1/EYX0), .before = 2)

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
  ) %>% 
  # Calculate the log risk ratios
  mutate(logrr = log(EYX1/EYX0), .before = 2)

# Write the draws to a parquet file
write_parquet(
  std_treat_draws,
  "output/predictions/treat_std_posterior_draws.gz.parquet",
  compression = "gzip",
  compression_level = 7
)

# Put everything together in a draws tibble
table_draws <- tibble(
  ATE_Treat_IPWM = ipwm_treat_draws$ATE,
  LRR_Treat_IPWM = ipwm_treat_draws$logrr,
  ATE_Treat = std_treat_draws$ATE,
  LRR_Treat = std_treat_draws$logrr,
  ATE_Test_IPWM = ipwm_test_draws$ATE,
  LRR_Test_IPWM = ipwm_test_draws$logrr,
  ATE_Test = std_test_draws$ATE,
  LRR_Test =  std_test_draws$logrr,
  .chain = std_test_draws$.chain,
  .iteration = std_test_draws$.iteration,
  .draw = std_test_draws$.draw
) %>% 
  as_draws_df()

# Write the draws to a parquet file
write_parquet(
  table_draws,
  "output/predictions/posterior_draws.gz.parquet",
  compression = "gzip",
  compression_level = 7
)

#------------------------------------------------------------------------------#
#-------------Standardized Differences brms, Complete Cases Models--------------
#------------------------------------------------------------------------------#

# Define the custom stanvars for the bootstrapped g-formula
gformula_vars <- stanvar(
  x = 1,
  name = "treat_pos",
  block = "data",
  scode = "int<lower = 1, upper = K> treat_pos; // Treatment Position"
) +
  stanvar(
    scode = "// Bootstrap Probabilities
  vector[N] boot_probs = rep_vector(1.0/N, N);
  
  // Matrices for the bootstrapped predictions
  matrix[N, Kc] Xa; 
  matrix[N, Kc] Xb;
  
  // Potential Outcome Y(X = 1, Z)
  Xa = X[, 2:K];
  Xa[, treat_pos] = ones_vector(N);
  
  // Potential Outcome Y(X = 0, Z)
  Xb = X[, 2:K];
  Xb[, treat_pos] = zeros_vector(N);",
  position = "end",
  block = "tdata"
  ) +
  stanvar(
    scode = "// Row index to be sampled for bootstrap
int row_i;

// Calculate Effect Estimates in the Bootstrapped sample
real ATE = 0.00;
array[N] real Y_X1; // Potential Outcome Y(X = 1, Z)
array[N] real Y_X0; // Potential Outcome Y(X = 0, Z)

for (n in 1:N) {
  // Sample the Baseline Covariates
  row_i = categorical_rng(boot_probs);
  
  // Sample Y(x) where x = 1 and x = 0
  Y_X1[n] = bernoulli_logit_rng(b_Intercept + Xa[row_i] * b);
  Y_X0[n] = bernoulli_logit_rng(b_Intercept + Xb[row_i] * b);
  
  // Add Contribution of the ith Observation to the Bootstrapped AME
  ATE = ATE + (Y_X1[n] - Y_X0[n])/N;
}

// Take the mean of the posterior expectations
real EYX1 = mean(Y_X1); // E[Y | X = 1, Z]
real EYX0 = mean(Y_X0); // E[Y | X = 0, Z]",
  position = "end",
  block = "genquant"
  )

# Test Outcome Model
anytest_bf <- bf(
  anytest ~ group + gender + age + partners + msm + ethnicgrp,
  family = bernoulli(link = "logit")
   )

# Priors for the parameters
anytest_priors <- prior(normal(0, 1), class = b) +
  prior(normal(0, 1.5), class = Intercept)

# Logistic regression for the anytest outcome
brms_anytest_logit <- brm(
  formula = anytest_bf,
  prior = anytest_priors,
  data = df,
  cores = 6,
  chains = 6,
  iter = 8000, 
  warmup = 3000,
  refresh = 1e3, 
  control = list(max_treedepth = 12),
  stanvars = gformula_vars,
  save_pars = save_pars(all = TRUE),
  backend = "cmdstanr", 
  file = "output/fits/brms_anytest_std_model_logit.rds"
)

# Print the quantities of interest
as_draws_df(brms_anytest_logit) %>% 
  select(ATE, EYX1, EYX0, .chain:.draw) %>% 
  mutate(LRR = log(EYX1/EYX0)) %>% 
  summarise_draws()
  
