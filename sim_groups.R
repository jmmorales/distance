

# simulate data
library(AHMbook)

#set.seed(1)
tmp <- simHDSg(type = "line")

datos = as.data.frame(tmp$data)

# tm = which(datos$y==1)
y = datos$y
gr = datos$gs
s = datos$site # site id
r = datos$d    # distance
x = as.matrix(tmp$habitat) # site covariates

n_obs = length(y)
nzs = 20
n_sites = dim(x)[1]

X = as.matrix(x)


library("rstan")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

nz = 300

stan_dat <- list(
  n_obs = n_obs,
  gr = gr,
  nz = nz,
  n_sites = n_sites,
  B = tmp$B,
  site = as.integer(s), # (c(s, rep(1:n_sites, each = nzs ))),
  r = r,
  K = 1, 
  X = X 
)

pars <- c("psi", "alpha", "beta", "sigma", "lambda_group", "a_g", "b_g")

fit <- stan(file = 'dist_groups_cov.stan',
            data = stan_dat,
            pars = pars,
            iter = 1000,
            thin = 1,
            chains = 3)


#, control = list(adapt_delta = 0.99))


psi = numeric(1000)
for(i in 1:1000){
  la = pars$alpha[i] + x * pars$beta[i]
  psi[i] = sum(exp(la))/(n_obs+nz)
}


mean.lambda = 2
beta.lam = 1
nsites = 100
habitat <- rnorm(nsites)
wind <- runif(nsites, -2, 2)
lambda <- exp(log(mean.lambda) + beta.lam * habitat)
N <- rpois(nsites, lambda)
