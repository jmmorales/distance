functions { 
  
  /* compute the kronecker product
  * Copied from brms: Paul-Christian Bürkner (2018). 
  * Advanced Bayesian Multilevel Modeling with the R Package brms. 
  * The R Journal, 10(1), 395-411. <doi:10.32614/RJ-2018-017>
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
}

data {
  int<lower=1> n_obs;                     // number of observed animals/groups
  int<lower=0> nz;                        // number of zeros 
  int<lower=1> n_sites;                   //
  real<lower=0> B;                        // max distance for study area
  int<lower=1,upper=n_sites> site[n_obs]; // site id for observed animals
  real<lower=0,upper=B> r[n_obs];         // observed distances
  int<lower=1> K;                         // number of env covariates
  matrix[n_sites, K] X;                   // env design matrix 
  int<lower=1> J;                 // number of groups (species)
  int<lower=1> L;                 // number group level predictors
  matrix[J,L] TT;                 // group-level traits
  matrix[J,J] C;                  // phylogenetic correlation matrix
  vector[J] ones;                 // vector on 1s
  int<lower=1,upper=J> jj[n_obs+nz];     // group id 
}

transformed data{
  vector[J] nj = ones-1;
  int<lower=n_obs> m = n_obs + nz;
  int nzj = nz/J;
  
  for(i in 1:n_obs){
    nj[jj[i]] = nj[jj[i]] + 1;
  }
}

parameters{
  real<lower=0,upper=B> rsim[nz]; // unobserved distances
  real<lower=0> sigma[J];
  
  corr_matrix[K+2] Omega;       // correlation matrix for regression parameters
  vector<lower=0>[K+2] tau;     // variance for parameters
  vector[J * (K+2)] theta;      // regression coefficients
  real<lower=0,upper=1> rho;    // phylogenetic effect
  vector[L * (K+2)] z;          // coefficients for trait effects on regression pars
}

transformed parameters{
  matrix[K+2, K+2] Sigma = quad_form_diag(Omega, tau);
  matrix[J*(K+2), J*(K+2)] S = kronecker(Sigma, rho * C + (1-rho) * diag_matrix(ones));
  matrix[L, K+2] Z = to_matrix(z, L, K+2);
  vector[J * (K+2)] m_theta = to_vector(TT * Z); 
  matrix[J, K+2] b_m = to_matrix(theta, J, K+2);
  
  real logit_p[m];
  matrix[n_sites, J] log_lambda;
  real<lower=0,upper=1> psi[J];
  real lp0[J];
  
  {
    real log_p[m];
    real dist[m];
    
    for (i in 1:n_obs) {
      dist[i] = r[i];
    }
    for (i in 1:nz) {
      dist[n_obs + i] = rsim[i];
    }
    
    for(s in 1:n_sites){
      for(j in 1:J){
        log_lambda[s,j] = b_m[j,1] + X[s,1] * b_m[j,2];
      }
    }
    
    for(j in 1:J){
      psi[j] = sum(exp(log_lambda[,j]))/(nj[j] + nzj);
      lp0[j] = bernoulli_lpmf(0 | psi[j]) ;
    }
    
    for(n in 1:m){
      //log_p[n] = - dist[n]^2/(2*sigma[jj[n]]^2);
      log_p[n] = - dist[n]^2/(2*exp(b_m[jj[n],3])^2);
      logit_p[n] = log_p[n] - log1m_exp(log_p[n]);
    }
  }
}

model { 
  Omega ~ lkj_corr(2);
  tau ~ student_t(3,0,10); // cauchy(0, 2.5);
  theta ~ multi_normal(m_theta, S);
  //rho ~ beta(1,10);
  z ~ normal(0,1);
  rsim ~ uniform(0, B);
  //sigma ~ normal(0,1);
  
  {
    vector[J] site_prob[n_sites];
    for(j in 1:J) site_prob[j] = softmax(log_lambda[,j]);
    
    for(n in 1: n_obs){
      target += bernoulli_lpmf(1 | psi[jj[n]]) + bernoulli_logit_lpmf(1 | logit_p[n]);
      target += categorical_lpmf(site[n]|site_prob[jj[n]]);
    }
    
    for(i in 1:nz){
      vector[n_sites + 1] lps;
      lps[1] = lp0[jj[n_obs+i]];
      for(s in 1:n_sites){
        lps[s+1] = bernoulli_lpmf(1 | psi[jj[n_obs+i]]) 
        + bernoulli_logit_lpmf(0 | logit_p[n_obs+i]) 
        + categorical_lpmf(s|site_prob[jj[n_obs+i]]);
      }
      target += log_sum_exp(lps);
    }
  }
}
//generated quantities{
  //   real D = (n_obs+(nz*psi))/Area;
  // }
  