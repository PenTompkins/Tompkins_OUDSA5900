#This file is being created to run all KNN OT2 files in a row

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
train_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Train/Amazon_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Test/Amazon_test_OT.xlsx")


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
#This particular subset of data contains no tweets belonging to topic 6
#Add that column in:
missingTop = rep(0, nrow(test_data))
test_data["topic.6"] = missingTop
test_data = within(test_data, rm(topic.0))

#Create categorical 'Sentiment' variable:
train_data["sent_cat"] = ifelse(train_data$Sentiment > 0, "Positive",
                                ifelse(train_data$Sentiment == 0, "Neutral","Negative"))

test_data["sent_cat"] = ifelse(test_data$Sentiment > 0, "Positive",
                               ifelse(test_data$Sentiment == 0, "Neutral","Negative"))

#One-hot encode sent_cat, as it's a categorical variable
train_data$sent_cat = factor(train_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = train_data)
dmy2 = data.frame(predict(dmy, newdata = train_data))
train_data = cbind(train_data, dmy2)
#Drop one column of dummy variables:
train_data = within(train_data, rm(sent_cat.Positive))

#Do the same for test data:
test_data$sent_cat = factor(test_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = test_data)
dmy2 = data.frame(predict(dmy, newdata = test_data))
test_data = cbind(test_data, dmy2)
#Drop one column of dummy variables:
test_data = within(test_data, rm(sent_cat.Positive))


###############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data, method = "kknn",
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
#############################################################################################################
#################################################################################################################
#################################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data, method = "kknn",
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

write_xlsx(df, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_Amazon_OT2.xlsx")
write_xlsx(df2, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_Amazon_OTparams2.xlsx")




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
train_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Train/BMW_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Test/BMW_test_OT.xlsx")


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

#Create categorical 'Sentiment' variable:
train_data["sent_cat"] = ifelse(train_data$Sentiment > 0, "Positive",
                                ifelse(train_data$Sentiment == 0, "Neutral","Negative"))

test_data["sent_cat"] = ifelse(test_data$Sentiment > 0, "Positive",
                               ifelse(test_data$Sentiment == 0, "Neutral","Negative"))

#One-hot encode sent_cat, as it's a categorical variable
train_data$sent_cat = factor(train_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = train_data)
dmy2 = data.frame(predict(dmy, newdata = train_data))
train_data = cbind(train_data, dmy2)
#Drop one column of dummy variables:
train_data = within(train_data, rm(sent_cat.Positive))

#Do the same for test data:
test_data$sent_cat = factor(test_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = test_data)
dmy2 = data.frame(predict(dmy, newdata = test_data))
test_data = cbind(test_data, dmy2)
#Drop one column of dummy variables:
test_data = within(test_data, rm(sent_cat.Positive))


###############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#############################################################################################################
#################################################################################################################
#################################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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

write_xlsx(df, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_BMW_OT2.xlsx")
write_xlsx(df2, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_BMW_OTparams2.xlsx")





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
train_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Train/Disney_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Test/Disney_test_OT.xlsx")


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

#Create categorical 'Sentiment' variable:
train_data["sent_cat"] = ifelse(train_data$Sentiment > 0, "Positive",
                                ifelse(train_data$Sentiment == 0, "Neutral","Negative"))

test_data["sent_cat"] = ifelse(test_data$Sentiment > 0, "Positive",
                               ifelse(test_data$Sentiment == 0, "Neutral","Negative"))

#One-hot encode sent_cat, as it's a categorical variable
train_data$sent_cat = factor(train_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = train_data)
dmy2 = data.frame(predict(dmy, newdata = train_data))
train_data = cbind(train_data, dmy2)
#Drop one column of dummy variables:
train_data = within(train_data, rm(sent_cat.Positive))

#Do the same for test data:
test_data$sent_cat = factor(test_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = test_data)
dmy2 = data.frame(predict(dmy, newdata = test_data))
test_data = cbind(test_data, dmy2)
#Drop one column of dummy variables:
test_data = within(test_data, rm(sent_cat.Positive))


###############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#############################################################################################################
#################################################################################################################
#################################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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

write_xlsx(df, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_Disney_OT2.xlsx")
write_xlsx(df2, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_Disney_OTparams2.xlsx")





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
train_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Train/MercedesBenz_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Test/MercedesBenz_test_OT.xlsx")


#Create binary 'has_link' variable in train/test data:
has_link = grepl("https://", train_data$Content)
train_data["has_link"] = has_link
train_data$has_link = ifelse(train_data$has_link == TRUE, 1, 0)
train_data$has_link = factor(train_data$has_link)
levels(train_data$has_link) = c(levels(train_data$has_link), "0")

has_link2 = grepl("https://", test_data$Content)
test_data["has_link"] = has_link2
test_data$has_link = ifelse(test_data$has_link == TRUE, 1, 0)
test_data$has_link = factor(test_data$has_link)
levels(test_data$has_link) = c(levels(test_data$has_link), "0")

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

#Create categorical 'Sentiment' variable:
train_data["sent_cat"] = ifelse(train_data$Sentiment > 0, "Positive",
                                ifelse(train_data$Sentiment == 0, "Neutral","Negative"))

test_data["sent_cat"] = ifelse(test_data$Sentiment > 0, "Positive",
                               ifelse(test_data$Sentiment == 0, "Neutral","Negative"))

#One-hot encode sent_cat, as it's a categorical variable
train_data$sent_cat = factor(train_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = train_data)
dmy2 = data.frame(predict(dmy, newdata = train_data))
train_data = cbind(train_data, dmy2)
#Drop one column of dummy variables:
train_data = within(train_data, rm(sent_cat.Positive))

#Do the same for test data:
test_data$sent_cat = factor(test_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = test_data)
dmy2 = data.frame(predict(dmy, newdata = test_data))
test_data = cbind(test_data, dmy2)
#Drop one column of dummy variables:
test_data = within(test_data, rm(sent_cat.Positive))


###############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
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
#############################################################################################################
#################################################################################################################
#################################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data, method = "kknn",
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

write_xlsx(df, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_Mercedes_OT2.xlsx")
write_xlsx(df2, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_Mercedes_OTparams2.xlsx")





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
train_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Train/Microsoft_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Test/Microsoft_test_OT.xlsx")


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

#Create categorical 'Sentiment' variable:
train_data["sent_cat"] = ifelse(train_data$Sentiment > 0, "Positive",
                                ifelse(train_data$Sentiment == 0, "Neutral","Negative"))

test_data["sent_cat"] = ifelse(test_data$Sentiment > 0, "Positive",
                               ifelse(test_data$Sentiment == 0, "Neutral","Negative"))

#One-hot encode sent_cat, as it's a categorical variable
train_data$sent_cat = factor(train_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = train_data)
dmy2 = data.frame(predict(dmy, newdata = train_data))
train_data = cbind(train_data, dmy2)
#Drop one column of dummy variables:
train_data = within(train_data, rm(sent_cat.Positive))

#Do the same for test data:
test_data$sent_cat = factor(test_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = test_data)
dmy2 = data.frame(predict(dmy, newdata = test_data))
test_data = cbind(test_data, dmy2)
#Drop one column of dummy variables:
test_data = within(test_data, rm(sent_cat.Positive))


###############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#############################################################################################################
#################################################################################################################
#################################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2, data = train_data, method = "kknn",
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

write_xlsx(df, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_Microsoft_OT2.xlsx")
write_xlsx(df2, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_Microsoft_OTparams2.xlsx")





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
train_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Train/Samsung_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Test/Samsung_test_OT.xlsx")


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

#Create categorical 'Sentiment' variable:
train_data["sent_cat"] = ifelse(train_data$Sentiment > 0, "Positive",
                                ifelse(train_data$Sentiment == 0, "Neutral","Negative"))

test_data["sent_cat"] = ifelse(test_data$Sentiment > 0, "Positive",
                               ifelse(test_data$Sentiment == 0, "Neutral","Negative"))

#One-hot encode sent_cat, as it's a categorical variable
train_data$sent_cat = factor(train_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = train_data)
dmy2 = data.frame(predict(dmy, newdata = train_data))
train_data = cbind(train_data, dmy2)
#Drop one column of dummy variables:
train_data = within(train_data, rm(sent_cat.Positive))

#Do the same for test data:
test_data$sent_cat = factor(test_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = test_data)
dmy2 = data.frame(predict(dmy, newdata = test_data))
test_data = cbind(test_data, dmy2)
#Drop one column of dummy variables:
test_data = within(test_data, rm(sent_cat.Positive))


###############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3, data = train_data, method = "kknn",
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
#############################################################################################################
#################################################################################################################
#################################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3, data = train_data, method = "kknn",
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

write_xlsx(df, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_Samsung_OT2.xlsx")
write_xlsx(df2, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_Samsung_OTparams2.xlsx")





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
train_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Train/Toyota_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Test/Toyota_test_OT.xlsx")


#Create binary 'has_link' variable in train/test data:
has_link = grepl("https://", train_data$Content)
train_data["has_link"] = has_link
train_data$has_link = ifelse(train_data$has_link == TRUE, 1, 0)
train_data$has_link = factor(train_data$has_link)
levels(train_data$has_link) = c(levels(train_data$has_link), "0")

has_link2 = grepl("https://", test_data$Content)
test_data["has_link"] = has_link2
test_data$has_link = ifelse(test_data$has_link == TRUE, 1, 0)
test_data$has_link = factor(test_data$has_link)
levels(test_data$has_link) = c(levels(test_data$has_link), "0")

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

#Create categorical 'Sentiment' variable:
train_data["sent_cat"] = ifelse(train_data$Sentiment > 0, "Positive",
                                ifelse(train_data$Sentiment == 0, "Neutral","Negative"))

test_data["sent_cat"] = ifelse(test_data$Sentiment > 0, "Positive",
                               ifelse(test_data$Sentiment == 0, "Neutral","Negative"))

#One-hot encode sent_cat, as it's a categorical variable
train_data$sent_cat = factor(train_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = train_data)
dmy2 = data.frame(predict(dmy, newdata = train_data))
train_data = cbind(train_data, dmy2)
#Drop one column of dummy variables:
train_data = within(train_data, rm(sent_cat.Positive))

#Do the same for test data:
test_data$sent_cat = factor(test_data$sent_cat)
dmy = dummyVars(" ~ sent_cat", data = test_data)
dmy2 = data.frame(predict(dmy, newdata = test_data))
test_data = cbind(test_data, dmy2)
#Drop one column of dummy variables:
test_data = within(test_data, rm(sent_cat.Positive))


###############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
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
#############################################################################################################
#################################################################################################################
#################################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
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
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "kknn",
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
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
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
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
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
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, method = "kknn",
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

write_xlsx(df, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_Toyota_OT2.xlsx")
write_xlsx(df2, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\KNN_Toyota_OTparams2.xlsx")


