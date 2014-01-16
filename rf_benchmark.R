# makes the random forest submission

library(randomForest)

print ("Reading train")
train <- read.csv("train.csv", header=TRUE)
print ("Done Reading train")
print ("Reading test")
test <- read.csv("test.csv", header=TRUE)
print ("Done Reading test")
labels <- as.factor(train[,1])
print ("Starting training now")
train <- train[,-1]

rf <- randomForest(train, labels, xtest=test, ntree=1)
predictions <- levels(labels)[rf$test$predicted]
print ("done training now")

print ("Writing predictions now")
write(predictions, file="rf_benchmark.csv", ncolumns=1) 
