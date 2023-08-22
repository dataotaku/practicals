theta = rbeta(10000, shape1=5, shape2=3)
y = theta / (1 - theta)
mean(y)
mean( y > 1)

sample = rnorm(100000, mean=0, sd = 1)
quantile(sample, probs = .3)
qnorm(p=0.3)
sqrt(5.2/5000)




Q = matrix(c(0.0, 0.5, 0.0, 0.0, 0.5,
             0.5, 0.0, 0.5, 0.0, 0.0,
             0.0, 0.5, 0.0, 0.5, 0.0,
             0.0, 0.0, 0.5, 0.0, 0.5,
             0.5, 0.0, 0.0, 0.5, 0.0),
           nrow=5, byrow=T)
Q %*% Q
# if the secret number is currently 1, the probability that the number will be
# 3 two steps from now is 0.25.
(Q %*% Q)[1,3]

# Suppose we want to know the probability distribution of the your secret number,
# say, X_t+h given X_t where h is a large number.
Q5 = Q %*% Q %*% Q %*% Q %*% Q # h=5 steps in the future
Q5

Q10 = Q %*% Q %*% Q %*% Q %*% Q %*% Q %*% Q %*% Q  %*% Q %*% Q
Q10

Q30 = Q
for (i in 2:30) {
  Q30 = Q30 %*% Q
}
round(Q30, 3)

# The stationary distribution of a chain is the initial state distribution
# for which performing a transition will not change the probability of
# ending up in any given state.

c(0.2, 0.2, 0.2, 0.2, 0.2) %*% Q

# One consequence of this property is that once a chain reaches its stationary
# distribution will remain the distribution of the states there after.
Q = matrix(c(0.0, 0.5, 0.0, 0.0, 0.5,
             0.5, 0.0, 0.5, 0.0, 0.0,
             0.0, 0.5, 0.0, 0.5, 0.0,
             0.0, 0.0, 0.5, 0.0, 0.5,
             0.5, 0.0, 0.0, 0.5, 0.0),
           nrow=5, byrow=T)
Q
?sample.int

n= 5000
x = numeric(n)
x[1] = 1
for (i in 2:n) {
  # draws the next state from the integers 1 to 5 with probs from
  # the transition matrix Q, based on the previous value of X.
  x[i] = sample.int(5, size=1, prob = Q[x[i-1],])
  print(x[i])
}

table(x) / n

# Markov chains transition probs finding : very simple case. 
set.seed(34)
n=100
x=numeric(n)
head(x)
for (i in 2:n) {
  x[i] = rnorm(1, mean=x[i-1], sd=1.0)
}
plot.ts(x)

# Markov chains transition probs finding : continuous example 
set.seed(38)
n=1500
x=numeric(n)
phi = -0.6

for (i in 2:n) {
  x[i] = rnorm(1, mean=phi*x[i-1], sd=1.0)
}
plot.ts(x)

hist(x, freq=F)
curve(dnorm(x, mean=0, sd = sqrt(1/(1-phi^2))), col="red", add=T)
legend("topright", legend="Theoretical\ndistribution", col='red', lty=1, bty='n')


# 연습문제 풀이
Qe = matrix(c(0.0, 1.0, 0.3, 0.7), nrow=2, byrow=T)
Qe
c(1,0) %*% Qe %*% Qe %*% Qe

n=4
x = numeric(n)
x[1] = 0
Qr = Qe
for (i in 2:n) {
  # draws the next state from the integers 1 to 5 with probs from
  # the transition matrix Q, based on the previous value of X.
  Qr = Qr %*% Qe
}
Qr

n=100
Qe = matrix(c(0.0, 1.0, 0.3, 0.7), nrow=2, byrow=T)
Qe
x = numeric(n)
x[1] = 0
Qr = Qe
for (i in 2:n) {
  # draws the next state from the integers 1 to 5 with probs from
  # the transition matrix Q, based on the previous value of X.
  Qr = Qr %*% Qe
}
Qr

# metropolis-hastings algorithm
# install.packages("rjags")
library(rjags)
mod_string = " model {
  for (i in 1:n) {
    y[i] ~ dnorm(mu, 1.0/sig2)
  }
  mu ~ dt(0.0, 1.0/1.0, 1.0) # location, inverse scale, degrees of freedom
  sig2 = 1.0
} "


set.seed(50)
y = c(1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9)
n = length(y)

data_jags = list(y=y, n=n)
params = c("mu")

inits = function() {
  inits = list("mu"=0.0)
} # optional (and fixed)

mod = jags.model(textConnection(mod_string), data=data_jags, inits=inits)


update(mod, 500) # burn-in

mod_sim = coda.samples(model=mod,
                       variable.names=params,
                       n.iter=1000)


summary(mod_sim)


library("coda")
plot(mod_sim)


