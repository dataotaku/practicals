dum = dpois(0,1.5) + dpois(1, 1.5)
1-dum
ppois(1,1.5)
ppois(0,2)
ppois(3,2)
1-ppois(4,2)

# beta - binomial posterior
qbeta(c(0.025, 0.975), shape1=1+2586, shape2=1+2414)
credible_interval_app()
qbeta(c(0.025, 0.975), shape1=500+2586, shape2=500+2414)
qbeta(c(0.45, 0.55), shape1=5, shape2=200)
qbeta(c(0.025, 0.975), shape1=5+2586, shape2=200+2414)

qbeta(c(0.025, 0.975), shape1=1+3868, shape2=1+1132)

credible_interval_app()

qgamma(c(0.025, 0.975), shape = 4, rate = 1)
qgamma(c(0.025, 0.975), shape = 12, rate = 3)
qgamma(c(0.025, 0.975), shape = 120, rate = 30)

qgamma(c(0.05, 0.95), shape = 8119, rate = 5001)

qbeta(c(0.05, 0.95), shape1=2, shape2=5)
data(brfss)
head(brfss)
dim(brfss)
sum(brfss$fruit_per_day)

pbeta(0.2, shape1=4, shape2=22)
qbeta(0.5, shape1=1, shape2=3)
qbeta(0.206, shape1=3, shape2=3)

1-pbeta(0.5, shape1=40+103, shape2=60+97)
sqrt(4*50^2/(4+50^2))
(200*4 + 149.3*50^2)/(4+50^2)

1-pbeta(0.85, shape1=400, shape2=100)
# In case you are still looking a few months later:
#   
#   pPrior <- c(0.5, 0.5)  # prior probability that it is coin A or coin B (given)
# 
# pHeads <- c(0.75, 0.25)  # p=0.75 heads for coin A, p=0.25 heads for coin B (given)
# 
# likelihoodTwoTails <- (1-pHeads)**2  # vector will be (0.0625, 0.5625)
# 
# pPosterior <- pPrior*likelihoodTwoTails / sum(pPrior*likelihoodTwoTails)  # posterior probability that it is coin A or coin B (vector will be c(0.1 0.9))
# 
# sum(pHeads*pPosterior)  # probability of heads if coin times probability it is coin
# 
# There are more elegant ways to code it in R, but this may be easier to follow than the shortcuts
?pbinom
dbinom(6, 10, prob=0.8)
pbinom(6, 10, prob=0.8)
?dpois
dpois(5, lambda=10)
?Hypergeometric
??hypergeom
dhyper(x=4,m=6,n=14, k=5)
dhyper(7, 101, 95, 10) #101여성, 95남성, 10번추출, 7명여성
# 비복원추출 : 초기하 분포 (시행간이 독립이 아님, 확률이 변함)
dhyper(x=2,m=7,n=3, k=3)
?dexp
# 시간당 평균 1.6회 전화가옴. 1시간내에 전화가 올 확율은?
pexp(1, rate=1.6)
# 시간당 평균 1.6회 전화가옴. 5분에서 20분사이에 전화가 올 확율은?
pexp(20/60, rate=1.6) - pexp(5/60, rate=1.6)
# 사거리에 차량도착이 평균 12초 (분당 5대)
# 차가 도착하는 시간이 12초 이내
pexp(12/60, rate=5)
# 차가 도착하는 시간이 6초이내
pexp(6/60, rate=5)
# AS에 2시간 소요
# 한시간 이내로 AS가 될 확율
pexp(1, rate=0.5)
# 1시간 이상, 2시간 미만이 걸릴 확율
pexp(2,rate=0.5) - pexp(1, rate=0.5)
# 오후 1시에 고객센터에 전화한 고객이 오후 5시까지 수리가 안될 확율
1-pexp(4, rate=0.5)
# 평균 30분마다 한마리 낚음. 4마리 낚는데 2시간에서 4시간 걸릴 확율
# shape=4마리, rate=시간당 2마리, 
# rate나 scale 두가지로 확률 계산이 가능하다.
pgamma(4, shape=4, rate=2) - pgamma(2, shape=4, rate=2)
pgamma(4, shape=4, scale=1/2) - pgamma(2, shape=4, scale=1/2)

# 철판 배달 1분에 1.6개, 20개의 패널이 15분 이내에 배달될 확율은?
pgamma(15, shape=20, rate=1.6)

# install.packages("BAS")
# install.packages("rjags")
# install.packages("ggthemes")
library(PairedData)
library(tidyverse)
library(statsr)
library(tidyverse)
library(statsr)
library(BAS)
library(ggplot2)
library(dplyr)
library(BayesFactor)
library(knitr)
library(rjags)
library(coda) 
library(latex2exp)
library(foreign)
library(BHH2)
library(scales)
library(logspline)
library(cowplot)
library(ggthemes)

# Prior
m_0 = 35;  n_0 = 25;  s2_0 = 156.25; v_0 = n_0 - 1
# Data
data(tapwater); Y = tapwater$tthm
ybar = mean(Y); s2 = var(Y); n = length(Y)
# Posterior Hyper-paramters
n_n = n_0 + n
m_n = (n*ybar + n_0*m_0)/n_n
v_n = v_0 + n
s2_n = ((n-1)*s2 + v_0*s2_0 + n_0*n*(m_0 - ybar)^2/n_n)/v_n

set.seed(42)

phi = rgamma(10000, shape = v_n/2, rate=s2_n*v_n/2)

df = data.frame(phi = sort(phi))
df = mutate(df, 
            density = dgamma(phi, 
                             shape = v_n/2,
                             rate=s2_n*v_n/2))

ggplot(data=df, aes(x=phi)) + 
  geom_histogram(aes(x=phi, y=after_stat(density)), bins = 50) +
  geom_density(aes(phi, after_stat(density)), color="black", lwd=1.5,alpha=0.7) +
  geom_line(aes(x=phi, y=density), color="orange", lwd=1.5,alpha=0.5) +
  xlab(expression(phi)) + theme_tufte()

mean(phi)

quantile(phi, c(0.025, 0.975))

# mean  (v_n/2)/(v_n*s2_n/2)
1/s2_n

sigma = 1/sqrt(phi)
mean(sigma) # posterior mean of sigma

quantile(sigma, c(0.025, 0.975))

# 연습문제 풀이
df = data.frame(sigma = sort(sigma))

ggplot(data=df, aes(x=sigma)) + 
  geom_histogram(aes(x=sigma, y=after_stat(density)), bins = 50) +
  geom_density(aes(sigma, after_stat(density)), 
               color="black", lwd=1.5,alpha=0.7) +
  xlab(expression(sigma)) + theme_tufte()


m_0 = (60+10)/2; s2_0 = ((60-10)/4)^2;
n_0 = 2; v_0 = n_0 - 1
set.seed(1234)
S = 10000
phi = rgamma(S, v_0/2, s2_0*v_0/2)
sigma = 1/sqrt(phi)
mu = rnorm(S, mean=m_0, sd=sigma/(sqrt(n_0)))
Y = rnorm(S, mu, sigma)
quantile(Y, c(0.025,0.975))

# 적절한 prior predictive distribution을 시행착오를 거쳐 도출함.
# 2 -> 25로 늘림.

m_0 = (60+10)/2; s2_0 = ((60-10)/4)^2;
n_0 = 25; v_0 = n_0 - 1
set.seed(1234)
phi = rgamma(10000, v_0/2, s2_0*v_0/2)
sigma = 1/sqrt(phi)
mu = rnorm(10000, mean=m_0, sd=sigma/(sqrt(n_0)))
y = rnorm(10000, mu, sigma)
quantile(y, c(0.025,0.975))

sum(y < 0)/length(y)  # P(Y < 0) a priori

set.seed(1234)
phi = rgamma(10000, v_n/2, s2_n*v_n/2)
sigma = 1/sqrt(phi)
post_mu = rnorm(10000, mean=m_n, sd=sigma/(sqrt(n_n)))
pred_y =  rnorm(10000,post_mu, sigma)
quantile(pred_y, c(.025, .975))

sum(pred_y > 80)/length(pred_y)  # P(Y > 80 | data)

bayes_inference(y=tthm, data=tapwater, statistic="mean",
                mu_0 = 35, rscale=1, prior="JZS",
                type="ci", method="sim")

library(statsr)
bayes_inference(difference, data=zinc, statistic="mean", type="ht",
                prior="JZS", mu_0=0, method="theo", alt="twosided")
install.packages("dlookr")

unique(nc$premie)
nc_premature = filter(nc, premie == 'premie')
summary(nc_premature)
ggplot(data = nc_premature, aes(x = weight)) +
  geom_histogram(binwidth = 1)
weight_post = bayes_inference(y = weight, data = nc_premature, 
                              statistic = "mean", type = "ci",  
                              prior_family = "JZS", mu_0 = 7.7, 
                              rscale = 1,
                              method = "simulation",
                              cred_level = 0.95)
samples = as.data.frame(weight_post$samples)
nsim = nrow(samples)
samples = mutate(samples, y_pred = rnorm(nsim, mu, sqrt(sig2)))
ggplot(data = samples, aes(x = y_pred)) + 
  geom_histogram(aes(y = ..density..), bins = 100) +
  geom_density() + 
  xlab(expression(y[new]))
dplyr::select(samples, mu, y_pred) %>%
  map(quantile, probs=c(0.025, 0.50, 0.975))

(133*28 + 100*30)/(166)

# Week3 Quiz3 4번문제
m_0 = 30;  n_0 = 100;  s2_0 = 100; v_0 = n_0 - 1
# Data
ybar = 28; s2 = 13^2; n = 133
# Posterior Hyper-paramters
n_n = n_0 + n
m_n = (n*ybar + n_0*m_0)/n_n
v_n = v_0 + n
s2_n = ((n-1)*s2 + v_0*s2_0 + n_0*n*(m_0 - ybar)^2/n_n)/v_n

n_n
m_n
v_n
s2_n

# Week3 Quiz3 5번문제
n_0 = 20
ybar = 1; n=22; s=3.6; v = n-1
t = ybar/(s/sqrt(n))
t
bf_f1_f2 = sqrt((n+n_0)/n_0)*((t^2*(n_0/(n+n_0))+v)/(t^2+v))^((v+1)/2)
bf_f1_f2
