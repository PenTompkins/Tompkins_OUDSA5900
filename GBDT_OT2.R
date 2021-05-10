#This file is being created to run GBDT for all OT data
#with sentiment added in

library(readxl)
library(caret)
library(tidyverse)
library(xgboost)
library(MLmetrics)
library(plyr)
library(writexl)


#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
nrounds_Likes = c()
nrounds_Retweets = c()
maxdepth_Likes = c()
maxdepth_Retweets = c()
eta_Likes = c()
eta_Retweets = c()
gamma_Likes = c()
gamma_Retweets = c()
colsamp_Likes = c()
colsamp_Retweets = c()
minchildweight_Likes = c()
minchildweight_Retweets = c()
subsample_Likes = c()
subsample_Retweets = c()

#Read in train data:
train_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Train/Amazon_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Test/Amazon_test_OT.xlsx")


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


##############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13,
                           topic.14 = train_data$topic.14,
                           topic.15 = train_data$topic.15,
                           topic.16 = train_data$topic.16)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13,
                         topic.14 = test_data$topic.14,
                         topic.15 = test_data$topic.15,
                         topic.16 = test_data$topic.16) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13,
                           topic.14 = train_data$topic.14,
                           topic.15 = train_data$topic.15,
                           topic.16 = train_data$topic.16)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13,
                         topic.14 = test_data$topic.14,
                         topic.15 = test_data$topic.15,
                         topic.16 = test_data$topic.16) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13,
                           topic.14 = train_data$topic.14,
                           topic.15 = train_data$topic.15,
                           topic.16 = train_data$topic.16)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13,
                         topic.14 = test_data$topic.14,
                         topic.15 = test_data$topic.15,
                         topic.16 = test_data$topic.16) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13,
                           topic.14 = train_data$topic.14,
                           topic.15 = train_data$topic.15,
                           topic.16 = train_data$topic.16)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13,
                         topic.14 = test_data$topic.14,
                         topic.15 = test_data$topic.15,
                         topic.16 = test_data$topic.16) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



#RETWEETS BELOW
##################################################################################################################
##################################################################################################################
##################################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13,
                           topic.14 = train_data$topic.14,
                           topic.15 = train_data$topic.15,
                           topic.16 = train_data$topic.16)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13,
                         topic.14 = test_data$topic.14,
                         topic.15 = test_data$topic.15,
                         topic.16 = test_data$topic.16) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13,
                           topic.14 = train_data$topic.14,
                           topic.15 = train_data$topic.15,
                           topic.16 = train_data$topic.16)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13,
                         topic.14 = test_data$topic.14,
                         topic.15 = test_data$topic.15,
                         topic.16 = test_data$topic.16) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13,
                           topic.14 = train_data$topic.14,
                           topic.15 = train_data$topic.15,
                           topic.16 = train_data$topic.16)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13,
                         topic.14 = test_data$topic.14,
                         topic.15 = test_data$topic.15,
                         topic.16 = test_data$topic.16) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13,
                           topic.14 = train_data$topic.14,
                           topic.15 = train_data$topic.15,
                           topic.16 = train_data$topic.16)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13,
                         topic.14 = test_data$topic.14,
                         topic.15 = test_data$topic.15,
                         topic.16 = test_data$topic.16) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                nrounds_Likes = nrounds_Likes,
                maxdepth_Likes = maxdepth_Likes,
                eta_Likes = eta_Likes,
                gamma_Likes = gamma_Likes,
                colsamp_Likes = colsamp_Likes,
                minchildweight_Likes = minchildweight_Likes,
                subsample_Likes = subsample_Likes,
                MAE_Retweets = Retweets_MAE,
                nrounds_Retweets = nrounds_Retweets,
                maxdepth_Retweets = maxdepth_Retweets,
                eta_Retweets = eta_Retweets,
                gamma_Retweets = gamma_Retweets,
                colsamp_Retweets = colsamp_Retweets,
                minchildweight_Retweets = minchildweight_Retweets,
                subsample_Retweets = subsample_Retweets)



write_xlsx(df, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelRes_OT\\GBDT_Amazon_OT2.xlsx")



#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
nrounds_Likes = c()
nrounds_Retweets = c()
maxdepth_Likes = c()
maxdepth_Retweets = c()
eta_Likes = c()
eta_Retweets = c()
gamma_Likes = c()
gamma_Retweets = c()
colsamp_Likes = c()
colsamp_Retweets = c()
minchildweight_Likes = c()
minchildweight_Retweets = c()
subsample_Likes = c()
subsample_Retweets = c()

#Read in train data:
train_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Train/BMW_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Test/BMW_test_OT.xlsx")


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


##############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



#RETWEETS BELOW
############################################################################################################
############################################################################################################
############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                nrounds_Likes = nrounds_Likes,
                maxdepth_Likes = maxdepth_Likes,
                eta_Likes = eta_Likes,
                gamma_Likes = gamma_Likes,
                colsamp_Likes = colsamp_Likes,
                minchildweight_Likes = minchildweight_Likes,
                subsample_Likes = subsample_Likes,
                MAE_Retweets = Retweets_MAE,
                nrounds_Retweets = nrounds_Retweets,
                maxdepth_Retweets = maxdepth_Retweets,
                eta_Retweets = eta_Retweets,
                gamma_Retweets = gamma_Retweets,
                colsamp_Retweets = colsamp_Retweets,
                minchildweight_Retweets = minchildweight_Retweets,
                subsample_Retweets = subsample_Retweets)



write_xlsx(df, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelRes_OT\\GBDT_BMW_OT2.xlsx")




#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
nrounds_Likes = c()
nrounds_Retweets = c()
maxdepth_Likes = c()
maxdepth_Retweets = c()
eta_Likes = c()
eta_Retweets = c()
gamma_Likes = c()
gamma_Retweets = c()
colsamp_Likes = c()
colsamp_Retweets = c()
minchildweight_Likes = c()
minchildweight_Retweets = c()
subsample_Likes = c()
subsample_Retweets = c()

#Read in train data:
train_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Train/Disney_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Test/Disney_test_OT.xlsx")


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


##############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



#RETWEETS BELOW
############################################################################################################
############################################################################################################
############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                nrounds_Likes = nrounds_Likes,
                maxdepth_Likes = maxdepth_Likes,
                eta_Likes = eta_Likes,
                gamma_Likes = gamma_Likes,
                colsamp_Likes = colsamp_Likes,
                minchildweight_Likes = minchildweight_Likes,
                subsample_Likes = subsample_Likes,
                MAE_Retweets = Retweets_MAE,
                nrounds_Retweets = nrounds_Retweets,
                maxdepth_Retweets = maxdepth_Retweets,
                eta_Retweets = eta_Retweets,
                gamma_Retweets = gamma_Retweets,
                colsamp_Retweets = colsamp_Retweets,
                minchildweight_Retweets = minchildweight_Retweets,
                subsample_Retweets = subsample_Retweets)



write_xlsx(df, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelRes_OT\\GBDT_Disney_OT2.xlsx")




#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
nrounds_Likes = c()
nrounds_Retweets = c()
maxdepth_Likes = c()
maxdepth_Retweets = c()
eta_Likes = c()
eta_Retweets = c()
gamma_Likes = c()
gamma_Retweets = c()
colsamp_Likes = c()
colsamp_Retweets = c()
minchildweight_Likes = c()
minchildweight_Retweets = c()
subsample_Likes = c()
subsample_Retweets = c()

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


##############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



#RETWEETS BELOW
###############################################################################################################
###############################################################################################################
###############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                nrounds_Likes = nrounds_Likes,
                maxdepth_Likes = maxdepth_Likes,
                eta_Likes = eta_Likes,
                gamma_Likes = gamma_Likes,
                colsamp_Likes = colsamp_Likes,
                minchildweight_Likes = minchildweight_Likes,
                subsample_Likes = subsample_Likes,
                MAE_Retweets = Retweets_MAE,
                nrounds_Retweets = nrounds_Retweets,
                maxdepth_Retweets = maxdepth_Retweets,
                eta_Retweets = eta_Retweets,
                gamma_Retweets = gamma_Retweets,
                colsamp_Retweets = colsamp_Retweets,
                minchildweight_Retweets = minchildweight_Retweets,
                subsample_Retweets = subsample_Retweets)



write_xlsx(df, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelRes_OT\\GBDT_Mercedes_OT2.xlsx")




#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
nrounds_Likes = c()
nrounds_Retweets = c()
maxdepth_Likes = c()
maxdepth_Retweets = c()
eta_Likes = c()
eta_Retweets = c()
gamma_Likes = c()
gamma_Retweets = c()
colsamp_Likes = c()
colsamp_Retweets = c()
minchildweight_Likes = c()
minchildweight_Retweets = c()
subsample_Likes = c()
subsample_Retweets = c()

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


##############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



#RETWEETS BELOW
############################################################################################################
############################################################################################################
############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                nrounds_Likes = nrounds_Likes,
                maxdepth_Likes = maxdepth_Likes,
                eta_Likes = eta_Likes,
                gamma_Likes = gamma_Likes,
                colsamp_Likes = colsamp_Likes,
                minchildweight_Likes = minchildweight_Likes,
                subsample_Likes = subsample_Likes,
                MAE_Retweets = Retweets_MAE,
                nrounds_Retweets = nrounds_Retweets,
                maxdepth_Retweets = maxdepth_Retweets,
                eta_Retweets = eta_Retweets,
                gamma_Retweets = gamma_Retweets,
                colsamp_Retweets = colsamp_Retweets,
                minchildweight_Retweets = minchildweight_Retweets,
                subsample_Retweets = subsample_Retweets)



write_xlsx(df, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelRes_OT\\GBDT_Microsoft_OT2.xlsx")




#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
nrounds_Likes = c()
nrounds_Retweets = c()
maxdepth_Likes = c()
maxdepth_Retweets = c()
eta_Likes = c()
eta_Retweets = c()
gamma_Likes = c()
gamma_Retweets = c()
colsamp_Likes = c()
colsamp_Retweets = c()
minchildweight_Likes = c()
minchildweight_Retweets = c()
subsample_Likes = c()
subsample_Retweets = c()

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


##############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



#RETWEETS BELOW
###############################################################################################################
###############################################################################################################
###############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                nrounds_Likes = nrounds_Likes,
                maxdepth_Likes = maxdepth_Likes,
                eta_Likes = eta_Likes,
                gamma_Likes = gamma_Likes,
                colsamp_Likes = colsamp_Likes,
                minchildweight_Likes = minchildweight_Likes,
                subsample_Likes = subsample_Likes,
                MAE_Retweets = Retweets_MAE,
                nrounds_Retweets = nrounds_Retweets,
                maxdepth_Retweets = maxdepth_Retweets,
                eta_Retweets = eta_Retweets,
                gamma_Retweets = gamma_Retweets,
                colsamp_Retweets = colsamp_Retweets,
                minchildweight_Retweets = minchildweight_Retweets,
                subsample_Retweets = subsample_Retweets)



write_xlsx(df, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelRes_OT\\GBDT_Samsung_OT2.xlsx")




#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
nrounds_Likes = c()
nrounds_Retweets = c()
maxdepth_Likes = c()
maxdepth_Retweets = c()
eta_Likes = c()
eta_Retweets = c()
gamma_Likes = c()
gamma_Retweets = c()
colsamp_Likes = c()
colsamp_Retweets = c()
minchildweight_Likes = c()
minchildweight_Retweets = c()
subsample_Likes = c()
subsample_Retweets = c()

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


##############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparams of best model:
nrounds_Likes = c(nrounds_Likes, bestmodel[,1])
maxdepth_Likes = c(maxdepth_Likes, bestmodel[,2])
eta_Likes = c(eta_Likes, bestmodel[,3])
gamma_Likes = c(gamma_Likes, bestmodel[,4])
colsamp_Likes = c(colsamp_Likes, bestmodel[,5])
minchildweight_Likes = c(minchildweight_Likes, bestmodel[,6])
subsample_Likes = c(subsample_Likes, bestmodel[,7])


#RETWEETS BELOW
##################################################################################################################
##################################################################################################################
##################################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])

##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 5, repeats = 2, search = "grid")
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3 = train_data$topic.3,
                           topic.4 = train_data$topic.4,
                           topic.5 = train_data$topic.5,
                           topic.6 = train_data$topic.6,
                           topic.7 = train_data$topic.7,
                           topic.8 = train_data$topic.8,
                           topic.9 = train_data$topic.9,
                           topic.10 = train_data$topic.10,
                           topic.11 = train_data$topic.11,
                           topic.12 = train_data$topic.12,
                           topic.13 = train_data$topic.13)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data_xy, method = "xgbTree",
               trControl = control, tuneLength = 5)

#Find the best model out of all trained during tuning:
bestmodel = model_$bestTune

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3 = test_data$topic.3,
                         topic.4 = test_data$topic.4,
                         topic.5 = test_data$topic.5,
                         topic.6 = test_data$topic.6,
                         topic.7 = test_data$topic.7,
                         topic.8 = test_data$topic.8,
                         topic.9 = test_data$topic.9,
                         topic.10 = test_data$topic.10,
                         topic.11 = test_data$topic.11,
                         topic.12 = test_data$topic.12,
                         topic.13 = test_data$topic.13) #format test input(s) properly
preds_ = predict(model_, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparams of best model:
nrounds_Retweets = c(nrounds_Retweets, bestmodel[,1])
maxdepth_Retweets = c(maxdepth_Retweets, bestmodel[,2])
eta_Retweets = c(eta_Retweets, bestmodel[,3])
gamma_Retweets = c(gamma_Retweets, bestmodel[,4])
colsamp_Retweets = c(colsamp_Retweets, bestmodel[,5])
minchildweight_Retweets = c(minchildweight_Retweets, bestmodel[,6])
subsample_Retweets = c(subsample_Retweets, bestmodel[,7])


##########################################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                nrounds_Likes = nrounds_Likes,
                maxdepth_Likes = maxdepth_Likes,
                eta_Likes = eta_Likes,
                gamma_Likes = gamma_Likes,
                colsamp_Likes = colsamp_Likes,
                minchildweight_Likes = minchildweight_Likes,
                subsample_Likes = subsample_Likes,
                MAE_Retweets = Retweets_MAE,
                nrounds_Retweets = nrounds_Retweets,
                maxdepth_Retweets = maxdepth_Retweets,
                eta_Retweets = eta_Retweets,
                gamma_Retweets = gamma_Retweets,
                colsamp_Retweets = colsamp_Retweets,
                minchildweight_Retweets = minchildweight_Retweets,
                subsample_Retweets = subsample_Retweets)



write_xlsx(df, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelRes_OT\\GBDT_Toyota_OT2.xlsx")
