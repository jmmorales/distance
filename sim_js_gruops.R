# Joint spp example with covariates in detection probability

library(ape)
library(mvtnorm)
set.seed(123)

n_sp = 20  # number of spp

n_env = 2  # environmental covariates
n_det = 2  # detection covariates
n_pars = n_env + n_det + 2   # number of parameters
n_t = 2   # number of traits

# define species traits
dgrass = rbeta(n_sp, 1, 1)  # some variable such as fraction of grass in diet
log_bm = rnorm(n_sp, 0, 1)  # log body mass

# define trait matrix includding ones for the intercept
TT = as.matrix(cbind(rep(1, n_sp), scale(dgrass), scale(log_bm)))

# simulate phylogeny 
tree = rtree(n = n_sp)    
CC = vcv(tree, corr = TRUE) # species correlation based on phylogeny

# sort species and re-arrange phylogenetic correlation matrix
tmp = dimnames(CC)
ids = as.numeric(as.factor(tmp[[1]]))

C = matrix(NA, ncol(CC), ncol(CC))
for(i in 1:ncol(CC)){
  for(j in 1:ncol(CC)){
    C[ids[i],ids[j]] = CC[i,j]
  }
}

# define parameters that link traits to expected betas
Z = matrix(rnorm((n_t + 1) * n_pars, 0 , 0.7),  (n_t + 1), n_pars)

# get expected betas
M = TT %*% Z

# define vcov for parameters
Sigma = diag(n_pars) * 0.6
rho = 0.5  # correlation with phylogeny

betas = rmvnorm(1, mean = as.vector(M), kronecker(Sigma, rho*C + (1-rho) * diag(n_sp)))
Beta = matrix(betas[1,], n_sp, n_pars)

# mean group size by spp (independent of traits at the moment)
lambda.group = exp(rnorm(n_sp, 0, 0.5))

# Now we define the sampling scheme. Here we follow the simulation sheme of package `AHMbook`.

n_sites = 100 # sampling units
# sample level predictor (environmental covariate)
X = cbind(rep(1, n_sites), matrix(rnorm(n_sites*n_env), n_sites, n_env))
B = 5 # max distance

N = matrix(NA, n_sites, n_sp)
for(i in 1: n_sp){
  N[,i] = rpois(n_sites, lambda = exp(X %*% Beta[i, (n_det+2):n_pars]))
}


data = NULL
for (i in 1:n_sites) {
  for(j in 1:n_sp){
    if(N[i,j] > 0){
      d = runif(N[i,j], 0, B)
      gs = rpois(N[i,j], lambda.group[j]) + 1
      sigma = exp(Beta[j,1] + gs * Beta[j,2] + TT[j,2] * Beta[j,3])
      p = exp(-d * d/(2 * (sigma^2)))
      y = rbinom(N[i,j], 1, p)
      d = d[y == 1]
      gs = gs[y == 1]
      y = y[y == 1]
      
    }
    if (sum(y) > 0){
      data = rbind(data, cbind(rep(i, sum(y)), rep(j, sum(y)), y, d, gs))
    } 
  }
}

colnames(data) = c("site","sp", "y", "d", "gs")
datos = as.data.frame(data)

#------------------------------------------------------------------------------
library("rstan")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

stan_dat <- list(
  n_obs = dim(datos)[1],
  n_sites = n_sites,
  gs = datos$gs,
  B = B,
  site = as.integer(datos$site), # (c(s, rep(1:n_sites, each = nzs ))),
  r = datos$d,
  K = 3, 
  X = as.matrix(cbind(numeric(length(x))+1, x)),
  n_max = rep(50, n_s),
  n_s = as.integer(n_s),
  n_t = 3,
  TT = TT,
  C = C,
  ones = numeric(n_s) + 1,
  sp = datos$sp
)

pars <- c("pbar", "b_m", "rho",  "Sigma", "z")

fit <- stan(file = 'dist_gr_pois_bin_int.stan',
            data = stan_dat,
            pars = pars,
            iter = 1000, thin = 1, chains = 3)
