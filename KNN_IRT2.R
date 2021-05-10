#This program is being created to run all KNN IRT2 files in a row
#Meaning num_words has been added in

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
train_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Train/Amazon_train_IRT2.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Test/Amazon_test_IRT2.xlsx")


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
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#############################################################################################################
#############################################################################################################

###############################################################################################################
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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

write_xlsx(df, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_Amazon_IRT2.xlsx")
write_xlsx(df2, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_AmazonParams_IRT2.xlsx")




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
train_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Train/BMW_train_IRT2.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Test/BMW_test_IRT2.xlsx")


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
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#############################################################################################################
#############################################################################################################
###############################################################################################################
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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

write_xlsx(df, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_BMW_IRT2.xlsx")
write_xlsx(df2, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_BMWParams_IRT2.xlsx")


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
train_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Train/CocaCola_train_IRT2.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Test/CocaCola_test_IRT2.xlsx")


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
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#############################################################################################################
#############################################################################################################
###############################################################################################################
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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

write_xlsx(df, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_CocaCola_IRT2.xlsx")
write_xlsx(df2, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_CocaColaParams_IRT2.xlsx")



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
train_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Train/Google_train_IRT2.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Test/Google_test_IRT2.xlsx")


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
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#############################################################################################################
#############################################################################################################

###############################################################################################################
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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

write_xlsx(df, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_Google_IRT2.xlsx")
write_xlsx(df2, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_GoogleParams_IRT2.xlsx")





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
train_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Train/McDonalds_train_IRT2.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Test/McDonalds_test_IRT2.xlsx")


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
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#############################################################################################################
#############################################################################################################
###############################################################################################################
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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

write_xlsx(df, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_McDonalds_IRT2.xlsx")
write_xlsx(df2, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_McDonaldsParams_IRT2.xlsx")





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
train_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Train/MercedesBenz_train_IRT2.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Test/MercedesBenz_test_IRT2.xlsx")


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
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#############################################################################################################
#############################################################################################################

###############################################################################################################
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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

write_xlsx(df, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_Mercedes_IRT2.xlsx")
write_xlsx(df2, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_MercedesParams_IRT2.xlsx")





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
train_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Train/Microsoft_train_IRT2.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_IRT/Test/Microsoft_test_IRT2.xlsx")


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
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#############################################################################################################
#############################################################################################################
###############################################################################################################
#Model 0: num_words
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 1: (Link, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 2: (Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 3: (Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 4: (Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 5: (Link, Hashtag, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 6: (Link, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 7: (Link, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 8: (Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 9: (Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 10: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 11: (Link, Hashtag, Sentiment, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 12: (Link, Hashtag, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 13: (Link, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 14: (Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2)
grid = expand.grid(kmax = 20:50,
                   distance = 1:5,
                   kernel = c('gaussian',  'rectangular',  'optimal'))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16 + topic.17 + topic.18 + topic.19 + num_words, data = train_data, method = "kknn",
               trControl = control, tuneLength = 50)
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

write_xlsx(df, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_Microsoft_IRT2.xlsx")
write_xlsx(df2, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\KNN_MicrosoftParams_IRT2.xlsx")



