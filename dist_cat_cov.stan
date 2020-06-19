data {
  int<lower=1> n_obs;   // number of observed animals/groups
  int<lower=0> nz;      // number of zeros 
  int<lower=1> n_sites; //
  real<lower=0> B;      // max distance for study area
  int<lower=0> y[n_obs+nz]; // ones for observed animals
  int<lower=1,upper=n_sites> site[n_obs]; // site id
  real<lower=0,upper=B> r[n_obs];   // observed distances
  int<lower=1> K;   // number of env covariates
  //real<lower=0> Area;
  matrix[n_sites, K] X;             // env design matrix 
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
    p[i] = exp(-r[i]^2/(2*sigma2)); 
  }
  for(j in (n_obs+1):(n_obs+nz)){
    p[j] = exp(-rsim[j-n_obs]^2/(2*sigma2));
  }
}

model { 
  vector[n_sites] lambda;
  
  sigma ~ normal(0, 1);
  rsim ~ uniform(0, B);
  alpha ~ normal(1,2);
  beta ~ normal(1,1);

  for(s in 1:n_sites){
    lambda[s] = exp(alpha + X[s,] * beta);
  }
  
  {
    vector[n_sites] site_prob = softmax(lambda);
    real psi= sum(lambda)/(n_obs + nz);
    vector[n_sites] lpsite = log(site_prob);
  
  for(n in 1: n_obs){
    target += bernoulli_lpmf(1 | psi) + bernoulli_lpmf(y[n] | p[n]);
    site[n] ~ categorical(site_prob);
  }
  
  for(i in 1:nz){
    vector[n_sites+1] lps = to_vector( append_row(bernoulli_lpmf(0 | psi), lpsite + bernoulli_lpmf(1 | psi) ) );
    for(s in 1:n_sites){
      lps[s+1] +=  bernoulli_lpmf(y[n_obs+i] | p[n_obs+i]);
    }
    target += log_sum_exp(lps);
  }
}
}

// generated quantities{
//   real D = (n_obs+(nz*psi))/Area;
// }