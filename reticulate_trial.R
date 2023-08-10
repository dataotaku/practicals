# install.packages("tinytex")
# install.packages("formatR")
# 
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install(c("graph", "Rgraphviz"), dep=TRUE)
# 
# install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)
# 
# install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/testing"), dep=TRUE)
# 
# install.packages("BAS")
# 
# install.packages("readr")
# install.packages("ggfortify")
# 
install.packages("reticulate")

library(reticulate)
path_to_python <- install_python()
virtualenv_create("r-reticulate", python = path_to_python)

virtualenv_install("r-reticulate", "scipy")
virtualenv_install("r-reticulate", "seaborn")

?dchisq
?dnorm
