data {
  int<lower=1> n_obs;   // number of observed animals/groups
  int<lower=1> gr[n_obs]; // observed group size
  int<lower=0> nz;      // number of zeros 
  int<lower=1> n_sites; //
  real<lower=0> B;      // max distance for study area
  //  int<lower=0> y[n_obs+nz]; // ones for observed animals
  int<lower=1,upper=n_sites> site[n_obs]; // site id
  real<lower=0,upper=B> r[n_obs];   // observed distances
  int<lower=1> K;   // number of env covariates
  //real<lower=0> Area;
  matrix[n_sites, K] X;             // env design matrix 
  //int<lower=1> max_gr; // max group size
}

transformed data{
  int<lower=0> groupsize[n_obs];
  for(i in 1:n_obs) groupsize[i] = gr[i]-1;
}

parameters{
  //real<lower=0> sigma;
  //real<lower=0,upper=1> psi;
  real<lower=0,upper=B> rsim[nz];
  real<lower=0> ugz[nz];
  real alpha;
  real a_g;
  real b_g;
  vector[K] beta;
  real<lower=0> lambda_group;
}

transformed parameters{
  real<lower=0,upper=1> p[n_obs+nz];
  real<lower=0> sigma[n_obs+nz];
  vector[n_sites] lambda;
  real<lower=0,upper=1> psi;
  
  for(s in 1:n_sites){
    lambda[s] = exp(alpha + X[s,] * beta);
  }
  psi = sum(lambda)/(n_obs + nz);
  for(n in 1:n_obs){
    sigma[n] = exp(a_g + b_g * groupsize[n]);
    p[n] = exp(- r[n]^2/2*sigma[n]^2);
  }
  
  for(j in (n_obs+1):(n_obs+nz)){
    sigma[j] = exp(a_g + b_g * ugz[j-n_obs]);
    p[j] = exp(-rsim[j-n_obs]^2/2*sigma[j]^2);
  }
}

model { 

  rsim ~ uniform(0, B);
  ugz ~ normal(lambda_group, sqrt(lambda_group));
  alpha ~ normal(1,2);
  beta ~ normal(1,1);
  
  {
    vector[n_sites] site_prob = softmax(lambda);
    //real psi= sum(lambda)/(n_obs + nz);
    vector[n_sites * 2] lpsite =  to_vector(append_row(log(site_prob), log(site_prob)));
    
    for(n in 1: n_obs){
      target += bernoulli_lpmf(1 | psi) + bernoulli_lpmf(1 | p[n]);
      target += categorical_lpmf(site[n]|site_prob);
    }
    
    for(i in 1:nz){
      vector[n_sites * 2] lps = lpsite;
      for(s in 1:n_sites){
        lps[s] += bernoulli_lpmf(0 | psi);
        lps[s+n_sites] += bernoulli_lpmf(1 | psi) + bernoulli_lpmf(0 | p[n_obs+i]);
      }
      target += log_sum_exp(lps);
    }
  }
}
// generated quantities{
  //   real D = (n_obs+(nz*psi))/Area;
  // }
  