# Joint spp example with p_obs

library(ape)
library(mvtnorm)
set.seed(123)

n_sp = 20  # number of spp

n_env = 2  # environmental covariates

n_pars = n_env  + 1   # number of parameters
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
Z = matrix(rnorm((n_t + 1) * n_pars, 0 , 0.5),  (n_t + 1), n_pars)

# get expected betas
M = TT %*% Z

# define vcov for parameters
Sigma = diag(n_pars) * 0.3
rho = 0.5  # correlation with phylogeny

betas = rmvnorm(1, mean = as.vector(M), kronecker(Sigma, rho*C + (1-rho) * diag(n_sp)))
Beta = matrix(betas[1,], n_sp, n_pars)

# mean group size by spp (independent of traits at the moment)
# lambda.group = exp(rnorm(n_sp, 0, 0.5))

# Now we define the sampling scheme. Here we follow the simulation sheme of package `AHMbook`.

n_sites = 100 # sampling units
# sample level predictor (environmental covariate)
X = cbind(rep(1, n_sites), matrix(rnorm(n_sites*n_env), n_sites, n_env))
B = 5 # max distance

N = matrix(NA, n_sites, n_sp)
for(i in 1: n_sp){
  N[,i] = rpois(n_sites, lambda = exp(X %*% Beta[i, ]))
}

p = rbeta(n_sp, 5,3)

data = NULL
for (i in 1:n_sites) {
  for(j in 1:n_sp){
    if(N[i,j] > 0){
      #d = runif(N[i,j], 0, B)
      #gs = rpois(N[i,j], lambda.group[j]) + 1
      #sigma = exp(Beta[j,1] + gs * Beta[j,2] + TT[j,2] * Beta[j,3])
      #p = exp(-d * d/(2 * (sigma^2)))
      y = rbinom(N[i,j], 1, p[j])
      #d = d[y == 1]
      #gs = gs[y == 1]
      y = y[y == 1]
      if (sum(y) > 0){
        data = rbind(data, cbind(rep(i, sum(y)), rep(j, sum(y)), y))
      }
    }
  }
}

colnames(data) = c("site","sp", "y")
datos = as.data.frame(data)

#------------------------------------------------------------------------------
# library(cmdstanr)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

stan_dat <- list(
  n_obs = dim(datos)[1],
  area = rep(1.0, n_sites),
  n_sites = n_sites,
  site = as.integer(datos$site), # (c(s, rep(1:n_sites, each = nzs ))),
  K = dim(X)[2], 
  X = X,
  n_max = rep(100, n_sp),
  n_s = as.integer(n_sp),
  n_t = dim(TT)[2],
  TT = TT,
  C = C,
  ones = numeric(n_sp) + 1,
  sp = datos$sp,
  p_obs = p
)

pars <- c( "b_m", "rho",  "Sigma", "z", "D")

init_f <- function () list(b_m = matrix(0, n_sp, n_pars))

fit <- stan(file = 'poisson_binomial_pobs.stan',
            data = stan_dat,
            init = init_f,
            pars = pars,
            iter = 1000, thin = 1, chains = 3)


fit_summary <- summary(fit)$summary
op <- par(mfrow = c(1,2))
hist(fit_summary[,10], main = "R-hat")
hist(fit_summary[,9], main = "n-eff" )
par(op)

draws <- extract(fit, pars = "rho")

plot(density(draws$rho), main = "")
abline(v=rho)

# plot trait level parameters
zs <- fit_summary[grepl("z", rownames(fit_summary)),]
#plot(c(Z) - zs[,1])
df <- data.frame(x = 1:dim(zs)[1],
                 tz = c(Z),
                 fz = zs[,1],
                 L = zs[,4],
                 U = zs[,8])

ggplot(df, aes(x = x, y = tz)) +
  geom_point(size = 3, color="red") +
  geom_point(aes(y = fz), size = 2) +
  geom_linerange(aes(ymin = L, ymax = U)) +
  theme_classic()


# plot intercepts and slopes
bs <- fit_summary[grepl("b_m", rownames(fit_summary)),]

nf = layout(matrix(c(1,2,3,4,0,0),3,2,byrow=TRUE), widths=c(1,1), heights=c(1,1,0.1))
#layout.show(nf)
op <- par( mar = c(3, 3, 2, 2) + 0.1, mgp = c(3.5, 1, 0), las = 1, bty = "n", cex = 1.2)
plot(scale(dgrass), Beta[,2],  ylab = "", xlab = "", main = "intercept", ylim=c(-3,3))
points(scale(dgrass), bs[seq(1, (n_sp*3) ,by=3) + 1,1], pch = 19, col = 2)

plot(scale(dgrass), Beta[,3], ylab = "", xlab = "", main = "slope", ylim=c(-3,3) )
points(scale(dgrass), bs[seq(1, (n_sp*3) ,by=3) + 2,1], pch = 19, col = 2)

plot(scale(log_bm),Beta[,2], ylab = "", xlab = "", ylim=c(-3,3))
points(scale(log_bm), bs[seq(1, (n_sp*3) ,by=3) + 1,1], pch = 19, col = 2)

plot(scale(log_bm),Beta[,3], ylab = "", xlab = "", ylim=c(-3,3))
points(scale(log_bm),  bs[seq(1, (n_sp*3) ,by=3) + 2,1], pch = 19, col = 2)
mtext("         scaled grass           scaled log body mass", side = 1, line = -2, outer = TRUE, cex=1.3)
par(mfrow = c(1,1))

# another plot of coefficients and estimates
plot(c(Beta), 
     c(bs[seq(1, (n_sp*3) ,by=3) ,1], bs[seq(1, (n_sp*3) ,by=3) + 1,1], bs[seq(1, (n_sp*3) ,by=3) + 2,1]), xlab = "true value", ylab = "posterior mean")
abline(0,1)

# yet another one
df <- data.frame(x = 1:dim(bs)[1],
                 tb = c(Beta),
                 fb = c(bs[seq(1, (n_sp*3) ,by=3) ,1], bs[seq(1, (n_sp*3) ,by=3) + 1,1], bs[seq(1, (n_sp*3) ,by=3) + 2,1]),
                 L =  c(bs[seq(1, (n_sp*3) ,by=3) ,4], bs[seq(1, (n_sp*3) ,by=3) + 1,4], bs[seq(1, (n_sp*3) ,by=3) + 2,4]),
                 U =  c(bs[seq(1, (n_sp*3) ,by=3) ,8], bs[seq(1, (n_sp*3) ,by=3) + 1,8], bs[seq(1, (n_sp*3) ,by=3) + 2,8])
)

ggplot(df, aes(x = x, y = tb)) +
  geom_point(size = 2, color="red") +
  geom_point(aes(y = fb), size = 1) +
  geom_linerange(aes(ymin = L, ymax = U)) +
  theme_classic()

# plot density estimates
D <- fit_summary[grepl("D", rownames(fit_summary)),]

df <- data.frame(x = 1:dim(D)[1],
                 td = colSums(N)/n_sites,
                 fd = D[,1],
                 L =  D[,4],
                 U =  D[,8]
)

ggplot(df, aes(x = x, y = td)) +
  geom_point(size = 2, color="red") +
  geom_point(aes(y = fd), size = 1) +
  geom_linerange(aes(ymin = L, ymax = U)) +
  theme_classic()

