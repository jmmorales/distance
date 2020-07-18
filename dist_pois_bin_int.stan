// https://github.com/stan-dev/example-models/blob/master/BPA/Ch.12/binmix.stan

functions { 
  
  /* compute the kronecker product
  * Args: 
  *   A,B: matrices 
  * Returns: 
  *   kronecker product of A and B
  */ 
  matrix kronecker(matrix A, matrix B) { 
    matrix[rows(A)*rows(B), cols(A)*cols(B)] kron; 
    for (i in 1:cols(A)) { 
      for (j in 1:rows(A)) { 
        kron[((j-1)*rows(B)+1):(j*rows(B)), ((i-1)*cols(B)+1):(i*cols(B))] = A[j,i] * B;
      } 
    } 
    return kron; 
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
  int<lower=1> n_obs;
  int<lower=1> n_sites;           // total number of observations (sites/segments)
  //int<lower=0> Y[n_sites];        // response variable (counts per site)
  int<lower=1> K;                 // number of sample-level predictors
  int<lower=1> n_s;               // num of species
  int<lower=1> n_t;               // num species level predictors (traits)
  int<lower=1,upper=n_s> sp[n_obs];   // species id 
  int<lower=1,upper=n_sites> site[n_obs];
  matrix[n_sites, K] X;                 // obs-level design matrix 
  matrix[n_s, n_t] TT;            // species-level traits
  matrix[n_s, n_s] C;             // phylogenetic correlation matrix
  vector[n_s] ones;               // vector on 1s
  int<lower=1> n_max[n_s];        // Upper bound of population size per spp
  real<lower=0> B;
  real<lower=0,upper=B> r[n_obs];
}

transformed data {
  real x_r[0]; // nothing for the integrand
  int x_i[0];  // nothing for the integrand
  
  int<lower=0> Y[n_sites, n_s]; // total spp by site
  
  for(i in 1:n_sites){
    for(j in 1:n_s){
      Y[i,j] = 0;
    }
  }
  
  for(i in 1:n_obs){
    Y[site[i], sp[i]] += 1;
  }
}

parameters {
  corr_matrix[K+1] Omega;           // correlation matrix for var-covar of betas
  vector<lower=0>[K+1] tau;         // scales for the variance covariance of betas
  vector[n_s * (K+1)] betas;
  real<lower=0,upper=1> rho;        // correlation between phylogeny and betas
  vector[n_t * (K+1)] z;            // coeffs for traits
  //real<lower=0,upper=1> p[n_s];   
}

transformed parameters { 
  matrix[K+1, K+1] Sigma = quad_form_diag(Omega, tau);
  matrix[n_s*(K+1), n_s*(K+1)] S = kronecker(Sigma, rho * C + (1-rho) * diag_matrix(ones));
  matrix[n_t, K+1] Z = to_matrix(z, n_t, K+1);    
  vector[n_s * (K+1)] m = to_vector(TT * Z);        // mean of coeffs
  matrix[n_s, K+1] b_m = to_matrix(betas, n_s, K+1);  // coeffs
  real pbar[n_s];
  
  for(i in 1:n_s) pbar[i] = integrate_1d(hn, 0, B, { exp(b_m[i,1]) }, x_r, x_i, 1e-8)/B;
} 

model {
  matrix[n_sites, n_s] log_lambda;
  int Ymax[n_sites, n_s];
  // priors
  // p ~ beta(2,2);
  Omega ~ lkj_corr(2);
  tau ~ student_t(3,0,10); // cauchy(0, 2.5); // lognormal()
  betas ~ multi_normal(m, S);
  //rho ~ beta(2,2);
  z ~ normal(0,2);
  
  // mix prior on rho
  //target += log_sum_exp(log(0.5) +  beta_lpdf(rho|1, 10), log(0.5) +  beta_lpdf(rho|2,2));
  
  for(i in 1: n_obs) target += (-r[i]^2/(2*exp(b_m[sp[i],1])^2)) - log(pbar[sp[i]]);
  
  for (n in 1:n_sites){
    for(s in 1:n_s){
      log_lambda[n,s] = dot_product( X[n,] , b_m[s,2:(K+1)]);
      Ymax[n,s] = Y[n,s] + qpois(0.9999, exp(log_lambda[n,s]) * (1 - pbar[s]), n_max[s]);
    }
  }
  
  for (n in 1:n_sites){
    for(s in 1:n_s){
      vector[Ymax[n,s] - Y[n,s] + 1] lp;
      for (j in 1:(Ymax[n,s]  - Y[n,s] + 1))
      lp[j] = poisson_log_lpmf(Y[n,s] + j - 1 | log_lambda[n,s])
      + binomial_lpmf(Y[n,s] | Y[n,s] + j - 1, pbar[s]);
      target += log_sum_exp(lp);
    }
  }
}
