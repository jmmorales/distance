functions{
  
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


data{
  real<lower=0> B;
  int<lower=1> nsites;
  int<lower=1> nind;
  int<lower=1> site[nind];
  int<lower=1> nD; // number of distance bins
  real<lower=0> midpt[nD]; // midpoints of distances
  int N[nsites]; 
  real habitat[nsites];
  real wind[nsites];
  int<lower=1> dclass[nind];
  real<lower=0> delta;
  int<lower=0> max_x;
}

// transformed data {
//   int<lower=0> max_y[nsites];
// 
//   for (i in 1:nsites)
//     max_y[i] = max(N[i]);
// }

parameters{
  
  real alpha0;
  real alpha1;
  real beta0;
  real beta1;
  
}

model{
  real sigma[nsites];
  real log_lambda[nsites];
  //matrix[nsites,nD] fc;
  //matrix[nsites,nD] p;
  //matrix[nsites,nD] p_i;
  vector[nD] f[nsites];
  real pcap[nsites];
  int Nmax[nsites];
  
  alpha0 ~ normal(0,10); 
  alpha1 ~ normal(0,1);
  beta0 ~ normal(0,10);
  beta1 ~ normal(0,1);

  for(s in 1:nsites){
    log_lambda[s] = beta0 + beta1 * habitat[s]; // Linear model abundance 
    sigma[s] = exp(alpha0 + alpha1*wind[s] );   // Linear model detection
    // Construct cell probabilities for nD multinomial cells
    for(g in 1:nD){
      f[s,g] = exp(- (midpt[g] * midpt[g] / (2*sigma[s]*sigma[s]))) * delta / B;
      //p_i[s,g] = delta / B;
      //f[s,g] = p[s,g] * delta / B;
    }
    pcap[s] = sum(f[s]);
    //for(g in 1:nD) fc[s,g] = f[s,g] / pcap[s];
    Nmax[s] = N[s] + qpois(0.9999, exp(log_lambda[s]) * (1 - pcap[s]), max_x);
    //Nmax[s] = 30;
  }
  for(s in 1:nsites){  
    vector[Nmax[s] - N[s] + 1] lp;
    //Nmin = min(x + qpois(0.00001, exp(log_lambda[s]) * (1 - pcap[s])));
  
    for (j in 1:(Nmax[s] - N[s] + 1)){
      lp[j] = poisson_log_lpmf(N[s] + j - 1 | log_lambda[s])
             + binomial_lpmf(N[s] | N[s] + j - 1, pcap[s]);
    }
    target += log_sum_exp(lp);
  }
  
  for(i in 1:nind){
    dclass[i] ~ categorical(softmax(f[site[i]])); // Part 1 of HM
  }
}

generated quantities{
  int Ns[nsites];
  int totalN;
  real log_lambda[nsites];
  real D;

  for (i in 1:nsites){
    log_lambda[i] = beta0 + beta1 * habitat[i];
    Ns[i] = poisson_log_rng(log_lambda[i]);
  }
  totalN = sum(Ns);
  D = totalN/(nsites*1*2*B);
}
