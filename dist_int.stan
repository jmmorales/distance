functions{
  real hn(real x, // Function argument
  real xc, // Complement of function argument
  // on the domain (defined later)
  real[] theta, // parameters
  real[] x_r, // data (real)
  int[] x_i){ // data (integer)
  
  real sigma = theta[1];
  
  return exp(-x^2/(2*sigma^2));
  }
  
  int qpois(real q, real lambda, int max_x) {
    int x = 0;
    real res = poisson_cdf(x, lambda);
    
    while(res < q && x < max_x){
      x = x + 1;
      res = poisson_cdf(x, lambda);
    }
    return x; 
  } 
}

data {
  int<lower=1> n_obs;
  real<lower=0> B;
  real<lower=0,upper=B> x[n_obs];
  real<lower=0> Area;
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {
  real<lower = 0.0> sigma;
}

transformed parameters{
  real pbar = integrate_1d(hn, 0, B, { sigma }, x_r, x_i, 1e-8) / B;
}

model {
  sigma ~ normal(0, 1);
  for(i in 1:n_obs) target += (-x[i]^2/(2*sigma^2)) - log(pbar);
  
//   N = n_obs + qpois(0.9999, exp(log_lambda[n]) * (1 - p[sp[n]]), n_max[sp[n]]);
  
}

generated quantities{
  real D = n_obs/(Area * pbar);
}
