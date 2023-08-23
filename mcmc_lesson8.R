data("PlantGrowth")
?PlantGrowth
head(PlantGrowth)

boxplot(weight ~ group, data=PlantGrowth)

lmod = lm(weight ~ group, data=PlantGrowth)
summary(lmod)

anova(lmod)

library("rjags")

mod_string = " model {
    for (i in 1:length(y)) {
        y[i] ~ dnorm(mu[grp[i]], prec)
    }
    
    for (j in 1:3) {
        mu[j] ~ dnorm(0.0, 1.0/1.0e6)
    }
    
    prec ~ dgamma(5/2.0, 5*1.0/2.0)
    sig = sqrt( 1.0 / prec )
} "

set.seed(82)
str(PlantGrowth)
data_jags = list(y=PlantGrowth$weight, 
                 grp=as.numeric(PlantGrowth$group))

params = c("mu", "sig")

inits = function() {
  inits = list("mu"=rnorm(3,0.0,100.0), "prec"=rgamma(1,1.0,1.0))
}

mod = jags.model(textConnection(mod_string), data=data_jags, inits=inits, n.chains=3)
update(mod, 1e3)

mod_sim = coda.samples(model=mod,
                       variable.names=params,
                       n.iter=5e3)
mod_csim = as.mcmc(do.call(rbind, mod_sim)) # combined chains


plot(mod_sim)

gelman.diag(mod_sim)
autocorr.diag(mod_sim)
effectiveSize(mod_sim)

(pm_params = colMeans(mod_csim))

yhat = pm_params[1:3][data_jags$grp]
yhat
resid = data_jags$y - yhat
plot(resid)

plot(yhat, resid)

summary(mod_sim)

HPDinterval(mod_csim)

mean(mod_csim[,3] > mod_csim[,1])

mean(mod_csim[,3] > 1.1*mod_csim[,1])

HPDinterval(mod_csim[,3] - mod_csim[,1])

# 연습문제---------


mod1_string = " model {
    for (i in 1:length(y)) {
        y[i] ~ dnorm(mu[grp[i]], prec[grp[i]])
    }
    
    for (j in 1:3) {
        mu[j] ~ dnorm(0.0, 1.0/1.0e6)
    }
    
    for (k in 1:3) {
        prec[k] ~ dgamma(5/2.0, 5*1.0/2.0)
    }
    
    sig = sqrt( 1.0 / prec )
} "

set.seed(82)
str(PlantGrowth)
data_jags = list(y=PlantGrowth$weight, 
                 grp=as.numeric(PlantGrowth$group))

params = c("mu", "sig")

inits1 = function() {
  inits = list("mu"=rnorm(3,0.0,100.0), "prec"=rgamma(3,1.0,1.0))
}

mod1 = jags.model(textConnection(mod1_string), data=data_jags, inits=inits1, n.chains=3)
update(mod1, 1e3)

mod1_sim = coda.samples(model=mod1,
                       variable.names=params,
                       n.iter=5e3)
mod1_csim = as.mcmc(do.call(rbind, mod1_sim)) # combined chains


plot(mod1_sim)

gelman.diag(mod1_sim)
autocorr.diag(mod1_sim)
effectiveSize(mod1_sim)


(pm1_params = colMeans(mod1_csim))

yhat1 = pm1_params[1:3][data_jags$grp]
yhat1
resid1 = data_jags$y - yhat1
plot(resid1)

plot(yhat1, resid1)

summary(mod1_sim)
summary(mod_sim)

HPDinterval(mod1_csim)

mean(mod1_csim[,3] > mod1_csim[,1])

mean(mod1_csim[,3] > 1.1*mod1_csim[,1])

dic.samples(mod1, n.iter=5e3)
dic.samples(mod, n.iter=5e3)
62.99 - 66.92

mod_cm = lm(weight ~ -1 + group, data=PlantGrowth)
summary(mod_cm)
