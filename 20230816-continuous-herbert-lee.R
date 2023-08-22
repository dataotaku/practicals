# Suppose we are giving two students a multiple-choice exam with 40 questions,
# where each question has four choices. We don't know how much the students
# have studied for this exam, but we think that they will do better than just
# guessing randomly.
# 1) What are the parameters of interest?
# 2) What is our likelihood?
# 3) What prior should we use?
# 4) What is the prior probability P(theta>.25)? P(theta>.5)? P(theta>.8)?
# 5) Suppose the first student gets 33 questions right. What is the posterior
#    distribution for theta1? P(theta1>.25)? P(theta1>.5)? P(theta1>.8)?
#    What is a 95% posterior credible interval for theta1?
# 6) Suppose the second student gets 24 questions right. What is the posterior
#    distribution for theta2? P(theta2>.25)? P(theta2>.5)? P(theta2>.8)?
#    What is a 95% posterior credible interval for theta2?
# 7) What is the posterior probability that theta1>theta2, i.e., that the 
#    first student has a better chance of getting a question right than
#    the second student?

############
# Solutions:

# 1) Parameters of interest are theta1=true probability the first student
#    will answer a question correctly, and theta2=true probability the second
#    student will answer a question correctly.

# 2) Likelihood is Binomial(40, theta), if we assume that each question is 
#    independent and that the probability a student gets each question right 
#    is the same for all questions for that student.
?dbinom
# 3) The conjugate prior is a beta prior. Plot the density with dbeta.
theta=seq(from=0,to=1,by=.01)
# ?dbeta
plot(theta,dbeta(theta,1,1),type="l")
plot(theta,dbeta(theta,5,1),type="l")
plot(theta,dbeta(theta,8,4),type="l")

# 4) Find probabilities using the pbeta function.
1-pbeta(.25,5,1)
1-pbeta(.5,5,1)
1-pbeta(.8,5,1)

1-pbeta(.25,1,1)
1-pbeta(.5,1,1)
1-pbeta(.8,1,1)

# 5) Posterior is Beta(8+33,4+40-33) = Beta(41,11)
41/(41+11)  # posterior mean
33/40       # MLE

lines(theta,dbeta(theta,41,11))

# plot posterior first to get the right scale on the y-axis
plot(theta,dbeta(theta,41,11),type="l")
lines(theta,dbeta(theta,8,4),lty=2)
# plot likelihood
lines(theta,dbinom(33,size=40,p=theta),lty=3)
# plot scaled likelihood
lines(theta,44*dbinom(33,size=40,p=theta),lty=3)

# posterior probabilities
1-pbeta(.25,41,11)
1-pbeta(.5,41,11)
1-pbeta(.8,41,11)

# equal-tailed 95% credible interval
qbeta(.025,41,11)
qbeta(.975,41,11)

# 6) Posterior is Beta(8+24,4+40-24) = Beta(32,20)
32/(32+20)  # posterior mean
24/40       # MLE

plot(theta,dbeta(theta,32,20),type="l")
lines(theta,dbeta(theta,8,4),lty=2)
lines(theta,44*dbinom(24,size=40,p=theta),lty=3)

1-pbeta(.25,32,20)
1-pbeta(.5,32,20)
1-pbeta(.8,32,20)

qbeta(.025,32,20)
qbeta(.975,32,20)

# 7) Estimate by simulation: draw 1,000 samples from each and see how often 
#    we observe theta1>theta2

theta1=rbeta(1000,41,11)
theta2=rbeta(1000,32,20)
mean(theta1>theta2)


# Note for other distributions:
# dgamma,pgamma,qgamma,rgamma
# dnorm,pnorm,qnorm,rnorm

theta=seq(from=0,to=1,by=.01)
plot(theta,dbeta(theta,5,1),type="l")
mean(dbeta(theta,5,1))
pbeta(.5,1,5)

qbeta(.025,8,16)
qbeta(.975,8, 16)

pbeta(0.35, 20, 7)

theta=seq(from=0,to=30,by=.01)
# ?dgamma
plot(theta,dgamma(theta,shape = 8, rate=1),type="l")
lines(theta,dgamma(theta,shape = 67, rate=6))
67/6
qgamma(.05, 67,6)
?pgamma

4/13
5/14

# exponential distribution
sqrt(1/16 * (4/5)^2)
4/5
5/4
(1+5)/(20+12+15+8+13.5+25)
(20+12+15+8+13.5+25)
pgamma(0.1, 6, 93.5)

theta=seq(from=0,to=100,by=.01)
# ?dgamma
plot(theta,dgamma(theta,shape = 9, rate=390),type="l")
3+16+8+114+60+4+23+30+105
16+8+114+60+4+23+30+105

qgamma(0.025, 8, 420)
qgamma(0.975, 9, 390)

# normal distribution
mean(c(94.6, 95.4, 96.2, 94.9, 95.9))
qnorm(0.975, 96.17, sqrt(0.042))

pnorm(q=100, mean=96.17, sd=sqrt(0.042))

z <- rgamma(n=1000, shape=3, rate=200)
x <- 1/z
mean(x)

x_vec= c()
for (x in 1:1000) {
  z <- rgamma(n=1000, shape=3, rate=200)
  x <- 1/z
  x_vec = c(x_vec, mean(x))
}
mean(x_vec)
mu <- rnorm(1000, mean=500, sd=sqrt(x_vec))
mean(mu)

z <- rgamma(1000, shape=16.5, rate=6022.9)
sig2 <- 1/z
mu <- rnorm(1000, mean=609.3, sd=sqrt(sig2/27.1))
quantile(x=mu, probs=c(0.025, 0.975))

# muA와 muB의 simulation 사전 수행 필요 

# sum( muA > muB ) / 1000

mean(c(94.6, 95.4, 96.2, 94.9, 95.9))

?dbeta
theta=seq(from=0,to=1,by=.001)
plot(theta,dbeta(theta,shape1 = .5, shape2=.5),type="l")