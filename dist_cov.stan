data {
  int<lower=1> n_obs;   // number of observed animals/groups
  int<lower=0> nz;      // number of zeros 
  int<lower=1> n_sites; //
  real<lower=0> B;      // max distance for study area
  int<lower=0> y[n_obs+nz]; // ones for observed animals
 // int<lower=1,upper=n_sites> s[n_obs+nz]; // site id
  real<lower=0,upper=B> r[n_obs];   // observed distances
  int<lower=1> K;   // number of env covariates
  //real<lower=0> Area;
  matrix[n_obs+nz, K] X;             // env design matrix 
  
}

transformed data{
  
}

parameters{
  real<lower=0> sigma;
  //real<lower=0,upper=1> psi;
  real<lower=0,upper=B> rsim[nz];
  real alpha;
  vector[K] beta;
}

transformed parameters{
  real<lower=0,upper=1> p[n_obs+nz];
  real<lower=0> sigma2 = sigma*sigma;
  
  for(i in 1:n_obs){
    p[i] = exp(-((r[i]^2)/2*sigma2)); 
  }
  for(j in (n_obs+1):(n_obs+nz)){
    p[j] = exp(-((rsim[j-n_obs]^2)/2*sigma2)); 
  }
  
}

model { 
  vector[n_obs+nz] psi;
  sigma ~ normal(0, 2);
  rsim ~ uniform(0, B);
  alpha ~ normal(0,2);
  beta ~ normal(0,1);
  
  for(n in 1:(n_obs+nz)){
      psi[n] = inv_logit(alpha + X[n,] * beta);
      if (y[n] == 0)
      target += log_sum_exp(bernoulli_lpmf(0 | psi[n]),
                            bernoulli_lpmf(1 | psi[n])
                              + bernoulli_lpmf(y[n] | p[n]));
    else
      target += bernoulli_lpmf(1 | psi[n])
                  + bernoulli_lpmf(y[n] | p[n]);
  }
}

// generated quantities{
//   real D = (n_obs+(nz*psi))/Area;
// }