# http://www.randomservices.org/random/data/Challenger2.txt
# 23 previous space shuttle launches before the Challenger disaster
# T is the temperature in Fahrenheit, I is the O-ring damage index

oring=read.table("./Challenger2.tsv",header=T, sep="\t")
# orings <-
#   structure(list(mission = 1:23, 
#                  temperature = c(53L, 57L, 58L, 
#                                                  63L, 66L, 67L, 67L, 67L, 68L, 69L, 70L, 70L, 70L, 70L, 72L, 73L, 
#                                                  75L, 75L, 76L, 76L, 78L, 79L, 81L), 
#                  damaged = c(5L, 1L, 1L, 1L, 
#                                                                                                  0L, 0L, 0L, 0L, 0L, 0L, 1L, 0L, 1L, 0L, 0L, 0L, 0L, 1L, 0L, 0L, 
#                                                                                               0L, 0L, 0L), 
#                  undamaged = c(1, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5, 6, 
#                                                                                                                             5, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6)), row.names = c(NA, -23L), 
#             class = c("tbl_df", "tbl", "data.frame"))
head(oring)
names(oring) = c("T","I")
attach(oring)
#note: masking T=TRUE

plot(T,I)

oring.lm=lm(I~T)
summary(oring.lm)

# add fitted line to scatterplot
lines(T,fitted(oring.lm))            

# 95% posterior interval for the slope
-0.24337 - 0.06349*qt(.975,21)
-0.24337 + 0.06349*qt(.975,21)

# note that these are the same as the frequentist confidence intervals

# the Challenger launch was at 31 degrees Fahrenheit
# how much o-ring damage would we predict?
# y-hat
18.36508-0.24337*31
coef(oring.lm)
coef(oring.lm)[1] + coef(oring.lm)[2]*31  

# posterior prediction interval (same as frequentist)
predict(oring.lm,data.frame(T=31),interval="predict")  
10.82052-2.102*qt(.975,21)*sqrt(1+1/23+((31-mean(T))^2/22/var(T)))

# posterior probability that damage index is greater than zero
1-pt((0-10.82052)/(2.102*sqrt(1+1/23+((31-mean(T))^2/22/var(T)))),21)


# http://www.randomservices.org/random/data/Galton.txt
# Galton's seminal data on predicting the height of children from the 
# heights of the parents, all in inches
# install.packages("UsingR")
library(UsingR)
data(galton)
data(GaltonFamilies)
heights = GaltonFamilies
head(heights)
summary(heights)
names(heights) = c("Family","Father","Mother","MidParentHeights","Kids","Child_num",
                   "Gender","Height")
# heights=read.table("http://www.randomservices.org/random/data/Galton.txt",header=T)
attach(heights)
names(heights)
unique(Kids)

pairs(heights)
summary(lm(Height~Father+Mother+Gender+Kids))
summary(lm(Height~Father+Mother+Gender))
heights.lm=lm(Height~Father+Mother+Gender)

summary(heights.lm)

# each extra inch taller a father is is correlated with 0.4 inch extra
#  height in the child
# each extra inch taller a mother is is correlated with 0.3 inch extra
#  height in the child
# a male child is on average 5.2 inches taller than a female child
# 95% posterior interval for the the difference in height by gender
5.215 - 0.142*qt(.975, 930)
5.215 + 0.142*qt(.975, 930)

# posterior prediction interval (same as frequentist)
predict(heights.lm,data.frame(Father=68,Mother=64,Gender="male"),interval="predict")
predict(heights.lm,data.frame(Father=68,Mother=64,Gender="female"),interval="predict")


# pga golf data
winner=read.table("./winner_data_pgalpga2008.dat",header=F)
names(winner) = c("yards","fairway","fm")
head(winner)
datF = subset(winner, fm==1, select=1:2)
head(datF)
datM = subset(winner, fm==2, select=1:2)
head(datM)

plot(datF)
plot(datM)

lm(fairway~yards, data = datF)
130.8933 -0.2565*260
predict(lm(fairway~yards, data = datF),
        data.frame(yards=260),interval="predict")

lm(fairway~yards + fm, data = winner)
winner$fm2 = ifelse(winner$fm == 2,1,0)
mod=lm(fairway~yards + fm2, data = winner)
summary(lm(fairway~yards + fm2, data = winner))

plot(fitted(mod), residuals(mod))
