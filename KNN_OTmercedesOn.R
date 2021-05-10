#This file's purpose is to run KNN modeling for...
#Mercedes, Microsoft, Samsung and Toyota OT data
#However, ultimately only Mercedes and Microsoft successfully ran through here


library(readxl)
library(caret)
library(class)
library(dplyr)
library(e1071)
library(FNN)
library(gmodels)
library(psych)
library(fpp2)
library(ggpubr)
library(MLmetrics)
library(writexl)
library(kknn)


#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
kmax_Likes = c()
kmax_Retweets = c()
distance_Likes = c()
distance_Retweets = c()
kernel_Likes = c()
kernel_Retweets = c()


#Read in train data:
train_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Train/MercedesBenz_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Test/MercedesBenz_test_OT.xlsx")


#Create binary 'has_link' variable in train/test data:
has_link = grepl("https://", train_data$Content)
train_data["has_link"] = has_link
train_data$has_link = ifelse(train_data$has_link == TRUE, 1, 0)

has_link2 = grepl("https://", test_data$Content)
test_data["has_link"] = has_link2
test_data$has_link = ifelse(test_data$has_link == TRUE, 1, 0)

#Create binary 'has_hash' variable in train/test data:
train_data["has_hash"] = ifelse(train_data$num_hashtags > 0, 1, 0)
test_data["has_hash"] = ifelse(test_data$num_hashtags > 0, 1, 0)

#One-hot encode topic, as it's a categorical variable
train_data$topic = factor(train_data$topic)
dmy = dummyVars(" ~ topic", data = train_data)
dmy2 = data.frame(predict(dmy, newdata = train_data))
train_data = cbind(train_data, dmy2)
#Drop one column of dummy variables:
train_data = within(train_data, rm(topic.0))

#Do the same for test data:
test_data$topic = factor(test_data$topic)
dmy = dummyVars(" ~ topic", data = test_data)
dmy2 = data.frame(predict(dmy, newdata = test_data))
test_data = cbind(test_data, dmy2)
#Drop one column of dummy variables:
test_data = within(test_data, rm(topic.0))


###############################################################################################################
#Model 1: Link
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

###############################################################################################################
#Model 2: Hashtag
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 3: Topic
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 4: (Link, Hashtag)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 5: (Link, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 6: (Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 7: (Link, Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)




#RETWEETS BELOW
########################################################################################################
############################################################################################################
############################################################################################################
#Model 1: Link
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

###############################################################################################################
#Model 2: Hashtag
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 3: Topic
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 4: (Link, Hashtag)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 5: (Link, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 6: (Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 7: (Link, Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


##########################################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                MAE_Retweets = Retweets_MAE)

df2 = data.frame(km_Likes = kmax_Likes,
                 dists_Likes = distance_Likes,
                 kernels_Likes = kernel_Likes,
                 km_Retweets = kmax_Retweets,
                 dists_Retweets = distance_Retweets,
                 kernels_Retweets = kernel_Retweets)

write_xlsx(df, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelingResults_IRT\\Storage\\KNN_Mercedes_OT.xlsx")
write_xlsx(df2, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelingResults_IRT\\Storage\\KNN_Mercedesparams_OT.xlsx")

#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
kmax_Likes = c()
kmax_Retweets = c()
distance_Likes = c()
distance_Retweets = c()
kernel_Likes = c()
kernel_Retweets = c()


#Read in train data:
train_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Train/Microsoft_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Test/Microsoft_test_OT.xlsx")


#Create binary 'has_link' variable in train/test data:
has_link = grepl("https://", train_data$Content)
train_data["has_link"] = has_link
train_data$has_link = ifelse(train_data$has_link == TRUE, 1, 0)

has_link2 = grepl("https://", test_data$Content)
test_data["has_link"] = has_link2
test_data$has_link = ifelse(test_data$has_link == TRUE, 1, 0)

#Create binary 'has_hash' variable in train/test data:
train_data["has_hash"] = ifelse(train_data$num_hashtags > 0, 1, 0)
test_data["has_hash"] = ifelse(test_data$num_hashtags > 0, 1, 0)

#One-hot encode topic, as it's a categorical variable
train_data$topic = factor(train_data$topic)
dmy = dummyVars(" ~ topic", data = train_data)
dmy2 = data.frame(predict(dmy, newdata = train_data))
train_data = cbind(train_data, dmy2)
#Drop one column of dummy variables:
train_data = within(train_data, rm(topic.0))

#Do the same for test data:
test_data$topic = factor(test_data$topic)
dmy = dummyVars(" ~ topic", data = test_data)
dmy2 = data.frame(predict(dmy, newdata = test_data))
test_data = cbind(test_data, dmy2)
#Drop one column of dummy variables:
test_data = within(test_data, rm(topic.0))


###############################################################################################################
#Model 1: Link
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

###############################################################################################################
#Model 2: Hashtag
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 3: Topic
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ topic.1 + topic.2, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 4: (Link, Hashtag)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 5: (Link, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link + topic.1 + topic.2, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 6: (Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_hash + topic.1 + topic.2, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 7: (Link, Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)




#RETWEETS BELOW
########################################################################################################
############################################################################################################
############################################################################################################
#Model 1: Link
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

###############################################################################################################
#Model 2: Hashtag
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 3: Topic
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ topic.1 + topic.2, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 4: (Link, Hashtag)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 5: (Link, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link + topic.1 + topic.2, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 6: (Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + topic.1 + topic.2, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 7: (Link, Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


##########################################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                MAE_Retweets = Retweets_MAE)

df2 = data.frame(km_Likes = kmax_Likes,
                 dists_Likes = distance_Likes,
                 kernels_Likes = kernel_Likes,
                 km_Retweets = kmax_Retweets,
                 dists_Retweets = distance_Retweets,
                 kernels_Retweets = kernel_Retweets)

write_xlsx(df, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelingResults_IRT\\Storage\\KNN_Microsoft_OT.xlsx")
write_xlsx(df2, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelingResults_IRT\\Storage\\KNN_Microsoftparams_OT.xlsx")

#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
kmax_Likes = c()
kmax_Retweets = c()
distance_Likes = c()
distance_Retweets = c()
kernel_Likes = c()
kernel_Retweets = c()


#Read in train data:
train_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Train/Samsung_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Test/Samsung_test_OT.xlsx")


#Create binary 'has_link' variable in train/test data:
has_link = grepl("https://", train_data$Content)
train_data["has_link"] = has_link
train_data$has_link = ifelse(train_data$has_link == TRUE, 1, 0)

has_link2 = grepl("https://", test_data$Content)
test_data["has_link"] = has_link2
test_data$has_link = ifelse(test_data$has_link == TRUE, 1, 0)

#Create binary 'has_hash' variable in train/test data:
train_data["has_hash"] = ifelse(train_data$num_hashtags > 0, 1, 0)
test_data["has_hash"] = ifelse(test_data$num_hashtags > 0, 1, 0)

#One-hot encode topic, as it's a categorical variable
train_data$topic = factor(train_data$topic)
dmy = dummyVars(" ~ topic", data = train_data)
dmy2 = data.frame(predict(dmy, newdata = train_data))
train_data = cbind(train_data, dmy2)
#Drop one column of dummy variables:
train_data = within(train_data, rm(topic.0))

#Do the same for test data:
test_data$topic = factor(test_data$topic)
dmy = dummyVars(" ~ topic", data = test_data)
dmy2 = data.frame(predict(dmy, newdata = test_data))
test_data = cbind(test_data, dmy2)
#Drop one column of dummy variables:
test_data = within(test_data, rm(topic.0))


###############################################################################################################
#Model 1: Link
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

###############################################################################################################
#Model 2: Hashtag
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 3: Topic
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ topic.1 + topic.2 + topic.3, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 4: (Link, Hashtag)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 5: (Link, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link + topic.1 + topic.2 + topic.3, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 6: (Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_hash + topic.1 + topic.2 + topic.3, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 7: (Link, Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2 + topic.3, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)




#RETWEETS BELOW
########################################################################################################
############################################################################################################
############################################################################################################
#Model 1: Link
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

###############################################################################################################
#Model 2: Hashtag
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 3: Topic
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ topic.1 + topic.2 + topic.3, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 4: (Link, Hashtag)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 5: (Link, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link + topic.1 + topic.2 + topic.3, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 6: (Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + topic.1 + topic.2 + topic.3, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 7: (Link, Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2 + topic.3, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


##########################################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                MAE_Retweets = Retweets_MAE)

df2 = data.frame(km_Likes = kmax_Likes,
                 dists_Likes = distance_Likes,
                 kernels_Likes = kernel_Likes,
                 km_Retweets = kmax_Retweets,
                 dists_Retweets = distance_Retweets,
                 kernels_Retweets = kernel_Retweets)

write_xlsx(df, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelingResults_IRT\\Storage\\KNN_Samsung_OT.xlsx")
write_xlsx(df2, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelingResults_IRT\\Storage\\KNN_Samsungparams_OT.xlsx")

#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
kmax_Likes = c()
kmax_Retweets = c()
distance_Likes = c()
distance_Retweets = c()
kernel_Likes = c()
kernel_Retweets = c()


#Read in train data:
train_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Train/Toyota_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Test/Toyota_test_OT.xlsx")


#Create binary 'has_link' variable in train/test data:
has_link = grepl("https://", train_data$Content)
train_data["has_link"] = has_link
train_data$has_link = ifelse(train_data$has_link == TRUE, 1, 0)

has_link2 = grepl("https://", test_data$Content)
test_data["has_link"] = has_link2
test_data$has_link = ifelse(test_data$has_link == TRUE, 1, 0)

#Create binary 'has_hash' variable in train/test data:
train_data["has_hash"] = ifelse(train_data$num_hashtags > 0, 1, 0)
test_data["has_hash"] = ifelse(test_data$num_hashtags > 0, 1, 0)

#One-hot encode topic, as it's a categorical variable
train_data$topic = factor(train_data$topic)
dmy = dummyVars(" ~ topic", data = train_data)
dmy2 = data.frame(predict(dmy, newdata = train_data))
train_data = cbind(train_data, dmy2)
#Drop one column of dummy variables:
train_data = within(train_data, rm(topic.0))

#Do the same for test data:
test_data$topic = factor(test_data$topic)
dmy = dummyVars(" ~ topic", data = test_data)
dmy2 = data.frame(predict(dmy, newdata = test_data))
test_data = cbind(test_data, dmy2)
#Drop one column of dummy variables:
test_data = within(test_data, rm(topic.0))


###############################################################################################################
#Model 1: Link
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

###############################################################################################################
#Model 2: Hashtag
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 3: Topic
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 4: (Link, Hashtag)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 5: (Link, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 6: (Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###############################################################################################################
#Model 7: (Link, Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Likes = c(kmax_Likes, km)
dist_ = model_$bestTune[,2]
distance_Likes = c(distance_Likes, dist_)
ker = model_$bestTune[,3]
kernel_Likes = c(kernel_Likes, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)




#RETWEETS BELOW
########################################################################################################
############################################################################################################
############################################################################################################
#Model 1: Link
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

###############################################################################################################
#Model 2: Hashtag
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 3: Topic
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 4: (Link, Hashtag)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 5: (Link, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 6: (Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###############################################################################################################
#Model 7: (Link, Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian', 'rectangular', 'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 +
                 topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
               trControl = control, tuneGrid = grid)
print(model_)

#Save hyperparams of best fitting model:
km = model_$bestTune[,1]
kmax_Retweets = c(kmax_Retweets, km)
dist_ = model_$bestTune[,2]
distance_Retweets = c(distance_Retweets, dist_)
ker = model_$bestTune[,3]
kernel_Retweets = c(kernel_Retweets, ker)


#Generate predictions:
preds_ = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


##########################################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                MAE_Retweets = Retweets_MAE)

df2 = data.frame(km_Likes = kmax_Likes,
                 dists_Likes = distance_Likes,
                 kernels_Likes = kernel_Likes,
                 km_Retweets = kmax_Retweets,
                 dists_Retweets = distance_Retweets,
                 kernels_Retweets = kernel_Retweets)

write_xlsx(df, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelingResults_IRT\\Storage\\KNN_Toyota_OT.xlsx")
write_xlsx(df2, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelingResults_IRT\\Storage\\KNN_Toyotaparams_OT.xlsx")

