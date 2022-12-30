/* Pseudo Bayesian Inverse Probability of Treatment Weighted Estimator
* Author: A. Jordan Nafa; Stan Version 2.31.0; Last Revised 12-29-2022 */
functions {
  // Weighted Log PMF of the Bernoulli Pseudo-Likelihood
  real bernoulli_logit_ipw_lpmf(int y, real mu, real w_tilde) {
    real weighted_term;
    weighted_term = 0.00;
    weighted_term = weighted_term + w_tilde * (bernoulli_logit_lpmf(y | mu));
    return weighted_term;
  }
}

data {
  int<lower = 0> N; // Observations
  array[N] int Y; // Outcome Stage Response
  int<lower = 1> K; // Number of Population-Level Effects
  matrix[N, K] X; // Design Matrix for the Population-Level Effects
  int<lower = 1, upper = K> treat_pos; // Treatment Position
  
  // Statistics from the Design Stage Model
  vector<lower = 0>[N] lambda; // Mean of the Population-Level Weights
  vector<lower = 0>[N] delta; // Scale of the Population-Level Weights
  
  // Prior on the scale of the weights
  real<lower = 0> sd_prior_shape1;
  real<lower = 0> sd_prior_shape2;
}

transformed data {
  matrix[N, K] Xc;  // Centered version of X without an Intercept
  vector[K] means_X;  // Column Means of the Uncentered Design Matrix
  
  // Centering the design matrix
  for (i in 1:K) {
    means_X[i] = mean(X[, i]);
    Xc[, i] = X[, i] - means_X[i];
  }
  
  // Bootstrap Probabilities
  vector[N] boot_probs = rep_vector(1.0/N, N);
  
  // Matrices for the bootstrapped predictions
  matrix[N, K] Xa = X; 
  matrix[N, K] Xb = X;
    
  // Potential Outcome Y(X = 1, Z)
  Xa[, treat_pos] = ones_vector(N);
    
  // Potential Outcome Y(X = 0, Z)
  Xb[, treat_pos] = zeros_vector(N);
}

parameters {
  vector[K] b; // Population-Level Effects
  real Intercept; // Population-Level Intercept for the Centered Predictors
  vector<lower=0, upper=1>[N] delta_z; // Parameter for the IPT Weights
}

transformed parameters {
  // Compute the IPTM Weights
  vector[N] w_tilde; // IPTM Weights
  w_tilde = lambda + delta .* delta_z;
}

model {
  // Initialize the Linear Predictor
  vector[N] mu = Intercept + Xc * b;
  
  // Sampling the Weights Prior
  delta_z ~ beta(sd_prior_shape1, sd_prior_shape2);
  
  // Priors for the Model Parameters
  target += normal_lpdf(Intercept | 0, 1.5);
  target += normal_lpdf(b | 0, 1);
  
  // Weighted Likelihood
  for (n in 1:N) {
    target += bernoulli_logit_ipw_lpmf(Y[n] | mu[n], w_tilde[n]);
  }
}

generated quantities {

  // Actual population-level intercept
  real b_Intercept = Intercept - dot_product(means_X, b);
  
  // Row index to be sampled for bootstrap
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
  real EYX0 = mean(Y_X0); // E[Y | X = 0, Z]
}
