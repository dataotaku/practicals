theta = rbeta(10000, shape1=5, shape2=3)
y = theta / (1 - theta)
mean(y)
mean( y > 1)

sample = rnorm(100000, mean=0, sd = 1)
quantile(sample, probs = .3)
qnorm(p=0.3)
sqrt(5.2/5000)


