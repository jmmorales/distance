

# simulate data
library(AHMbook)

set.seed(1234)
tmp <- simHDS(type="line", discard0=FALSE) 

datos = as.data.frame(tmp$data)

tm = which(datos$y==1)
y = datos$y[tm]
s = datos$site[tm] # site id
r = datos$d[tm]    # distance
x = as.matrix(tmp$habitat) # site covariates

n_obs = length(y)
nzs = 20
n_sites = dim(x)[1]

X = as.matrix(x)


library("rstan")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

nz = 100

stan_dat <- list(
  n_obs = n_obs,
  nz = nz,
  n_sites = n_sites,
  B = 3,
  y = as.integer( c(y, numeric(nz))), 
  site = as.integer(s), # (c(s, rep(1:n_sites, each = nzs ))),
  r = r,
  K = 1, 
  X = X 
)

pars <- c("alpha", "beta", "sigma")

fit <- stan(file = 'dist_cat_cov.stan',
            data = stan_dat,
            pars = pars,
            iter = 10000,
            thin = 5,
            chains = 3, control = list(adapt_delta = 0.99))
