# makes the KNN submission

library(KNN)

print ("Reading train")
train <- read.csv("../data/train.csv", header=TRUE)
print ("done Reading train")
print ("Reading test")
test <- read.csv("../data/test.csv", header=TRUE)
print ("done Reading test")

labels <- train[,1]
train <- train[,-1]

print ("Training now")
results <- (0:9)[knn(train, test, labels, k = 1, algorithm="k-d_tree")]
print ("Done training")

print ("writing results")
write(results, file="knn_benchmark.csv", ncolumns=1) 
