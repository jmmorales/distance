# Distance sampling using n-mixture
# here we discretize distances in bins if width delta 
# 
# from Kery and Royle page 452
# 8.5.3 BAYESIAN HDS USING THE THREE-PART CONDITIONAL MULTINOMIAL MODEL

library("rstan")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# we use this library to simulate data
library(AHMbook)

set.seed(1234)
tmp <- simHDS(type="line", discard0=FALSE) 
attach(tmp)

# Get number of individuals detected per site
# ncap = 1 plus number of detected individuals per site
ncap <- table(data[,1])
sites0 <- data[is.na(data[,2]),][,1] # sites where nothing detected
ncap[as.character(sites0)] <- 0      # Fill in 0 for sites with no detections
ncap <- as.vector(ncap)
# Prepare other data
site <- data[!is.na(data[,2]),1]     # site ID of each observation
delta <- 0.1                         # distance bin width for rect. approx.
midpt <- seq(delta/2, B, delta)      # make mid-points and chop up data
dclass <- data[,5] %/% delta + 1     # convert distances to cat. distances 
nD <- length(midpt)                  # Number of distance intervals
dclass <- dclass[!is.na(data[,2])]   # Observed categorical observations
nind <- length(dclass)               # Total number of individuals detected

# get total number of individuals observed per site 
s = unique(site)
N = numeric(nsites)
for( i in 1:length(s)){
  tem = which(site == s[i])
  N[s[i]] = length(tem)
}

# list data to pass to Stan
dat <- list(nsites=nsites, 
            nind=nind,
            B=B,
            N = N,
            nD=nD, 
            midpt=midpt,
            delta=delta, 
            ncap=ncap, 
            habitat=habitat, 
            wind=wind, 
            dclass=dclass, 
            site=site,
            max_x = 100)

pars = c("alpha0", "alpha1", "beta0", "beta1", "totalN", "D")

fit <- stan(file = 'n-mix.stan', 
            data = dat,
            pars = pars,
            iter = 1000, thin = 1, chains = 3)

print(fit)

samples = extract(fit)

op = par(mfrow=c(1,3))
plot(density(samples$totalN), main = "")
abline(v = sum(tmp$N.true))
plot(density(samples$alpha1), main = "")
abline(v=tmp$beta.sig)
plot(density(samples$beta1), main = "")
abline(v=tmp$beta.lam)
par(op)
