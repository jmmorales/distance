

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

pars <- c("psi", "alpha", "beta", "lambda_group", "a_g", "b_g")

fit <- stan(file = 'dist_groups_cov.stan',
            data = stan_dat,
            pars = pars,
            iter = 1000,
            thin = 1,
            chains = 3)

pos = extract(fit, pars = pars)
op = par(mfrow = c(3,2))
plot(density(pos$alpha))
abline(v = tmp$beta0)
plot(density(pos$beta))
abline(v = tmp$beta1)
plot(density(pos$a_g))
abline(v = tmp$alpha0)
plot(density(pos$b_g))
abline(v = tmp$alpha1)
plot(density(pos$lambda_group))
abline(v = tmp$lambda.group)
par(op)

