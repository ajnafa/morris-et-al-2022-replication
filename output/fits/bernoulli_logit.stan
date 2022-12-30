/* Bayesian Logit Model with Bootstrapped ATEs via the Bayesian g-formula
* Author: A. Jordan Nafa; Stan Version 2.31.0; Last Revised 12-29-2022 */

data {
  int<lower = 0> N; // Observations
  array[N] int Y; // Outcome Stage Response
  int<lower = 1> K; // Number of Population-Level Effects
  matrix[N, K] X; // Design Matrix for the Population-Level Effects
  int<lower = 1, upper = K> treat_pos; // Treatment Position
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
}

model {
  // Priors for the Model Parameters
  target += normal_lpdf(Intercept | 0, 1.5);
  target += normal_lpdf(b | 0, 1);
  
  // likelihood including constants
  target += bernoulli_logit_glm_lpmf(Y | Xc, Intercept, b);
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

