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
}
data {
  int N;
  real x[N];
  real<lower=0> B;
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
  for(i in 1:N) target += (-x[i]^2/(2*sigma^2)) - log(pbar);
}