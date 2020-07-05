data {
  int<lower=1> n_obs;                     // number of observed animals/groups
  int<lower=1> gr[n_obs];                 // observed group size
  int<lower=0> nz;                        // number of zeros 
  int<lower=1> n_sites;                   //
  real<lower=0> B;                        // max distance for study area
  int<lower=1,upper=n_sites> site[n_obs]; // site id for observed animals
  real<lower=0,upper=B> r[n_obs];         // observed distances
  int<lower=1> K;                         // number of env covariates
  matrix[n_sites, K] X;                   // env design matrix 
  int<lower=1> max_gr;                    // max group size
}

transformed data{
  int<lower=0> groupsize[n_obs];
  int<lower=n_obs> m = n_obs + nz;
  for(i in 1:n_obs) groupsize[i] = gr[i]-1;
}

parameters{
  real<lower=0,upper=B> rsim[nz];   // unobserved distances
  //real<lower=0> ugz[nz];          // unobserved group size
  real alpha;
  real a_g;
  real b_g;
  vector[K] beta;
  real<lower=0> lambda_group;
}

transformed parameters{
  real logit_p[n_obs];
  matrix[nz, max_gr + 1] logit_pz;
  vector[n_sites] log_lambda;
  real<lower=0,upper=1> psi;
  real lp0;
  //vector[max_gr + 1] lpg;
    
  {
    real log_p[n_obs];
    matrix[nz, max_gr + 1] log_pz;
    
    //for(g in 1:(max_gr+1)) lpg[g] = poisson_lpmf(g-1|lambda_group);
    
    for(s in 1:n_sites){
      log_lambda[s] = alpha + X[s,] * beta;
    }
    
    psi = sum(exp(log_lambda))/(n_obs + nz);
    lp0 = bernoulli_lpmf(0 | psi) ;
    
    for(n in 1:n_obs){
      log_p[n] = - r[n]^2/(2*(exp(a_g + b_g * groupsize[n]))^2);
      logit_p[n] = log_p[n] - log1m_exp(log_p[n]);
    }
    
    for(j in 1:nz){
      for(g in 1:(max_gr+1)){
        log_pz[j,g] = -rsim[j]^2/(2* (exp(a_g + b_g * (g-1)) )^2);
        logit_pz[j,g] = log_pz[j,g] - log1m_exp(log_pz[j,g]);
      }
    }
  }
}

model { 
  
  rsim ~ uniform(0, B);
  lambda_group ~ normal(0,2);
  //ugz ~ normal(lambda_group, sqrt(lambda_group));
  a_g ~ normal(0,1);
  b_g ~ normal(0,1);
  alpha ~ normal(1,2);
  beta ~ normal(1,1);
  
  {
    //vector[n_sites] log_site_prob = log_softmax(log_lambda);
    vector[n_sites] site_prob = softmax(log_lambda);
    matrix[n_sites , max_gr + 1] lpgs;
    
    for(s in 1:n_sites){
      for(g in 1:(max_gr+1)){
        lpgs[s,g] = categorical_lpmf(s|site_prob)
        + poisson_lpmf(g-1|lambda_group) 
        + bernoulli_lpmf(1 | psi) ;
      }
    }
    
    for(n in 1: n_obs){
      target += bernoulli_lpmf(1 | psi) + bernoulli_logit_lpmf(1 | logit_p[n]);
      target += categorical_lpmf(site[n]|site_prob);
      target += poisson_lpmf(groupsize[n]|lambda_group);
    }
    

    for(i in 1:nz){
      matrix[n_sites , max_gr + 1] lps = lpgs;
      for(s in 1:n_sites){
        for(g in 1:(max_gr+1)){
          lps[s,g] += bernoulli_logit_lpmf(0 | logit_pz[i,g]);
        }
      }
      target += log_sum_exp(append_row(to_vector(lps), lp0));
    }
  }
}
