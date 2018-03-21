'''
This script aims to exchange the first two principle components of the features.
'''
# Load libraries
library(MVN)
library(gdata)

# Load data
data <- read.csv("csv/test_all.csv")
frame_start <- data[,1]
data <- data[,-1]
data <- data[,1:1024]
dim(data)
raw_data <- data[,-dim(data)[2]]

# PCA
data.pca <- prcomp(raw_data, scale = TRUE)
pca_data <- data.pca$x
npca_data <- t(t(pca_data) / data.pca$sdev)

# Perform rotation
npca_data_rotated <- npca_data
head(npca_data_rotated[,1:5],3)
npca_data_rotated[,1] <- npca_data[,2]
npca_data_rotated[,2] <- npca_data[,1]
head(npca_data_rotated[,1:5],3)

# Inverse-PCA
pca_data_new <- t(t(npca_data_rotated) * data.pca$sdev)
data_new <- t(t(pca_data_new %*% t(data.pca$rotation)) * data.pca$scale + data.pca$center)

write.csv(data_new, file = "output/test_all_new.csv")

