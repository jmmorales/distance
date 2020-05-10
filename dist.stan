data {
  int<lower=1> n_obs;
  int<lower=0> nz;
  real<lower=0> B;
  int<lower=0,upper=1> y[n_obs+nz];
  real<lower=0,upper=B> x[n_obs];
  real<lower=0> Area;
}

parameters{
  real<lower=0> sigma;
  real<lower=0,upper=1> psi;
  real<lower=0,upper=B> x[nz];
}

transformed parameters{
  real<lower=0,upper=1> p[n_obs+nz];
  real<lower=0> sigma2 = sigma*sigma;
  
  for(i in 1:n_obs){
    p[i] = exp(-((x[i]^2)/2*sigma2)); 
  }
  for(j in (n_obs+1):(n_obs+nz)){
    p[j] = exp(-((x[j-n_obs]^2)/2*sigma2)); 
  }
  
}

model { 
  sigma ~ normal(0,2);
  psi ~ beta(1,1);
  x ~ uniform(0, B);
  
  for(n in 1:(n_obs+nz)){
      if (y[n] == 0)
      target += log_sum_exp(bernoulli_lpmf(0 | psi),
                            bernoulli_lpmf(1 | psi)
                              + bernoulli_lpmf(y[n] | p[n]));
    else
      target += bernoulli_lpmf(1 | psi)
                  + bernoulli_lpmf(y[n] | p[n]);
  }
}

generated quantities{
  real D = (n_obs+(nz*psi))/Area;
}