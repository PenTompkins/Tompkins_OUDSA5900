#This program is being created to generate SVR predictions for Amazon OT data
#with sentiment added in
#
#Overall, this is program: 100

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
Likes_cost = c()
Likes_gamma = c()
Likes_epsilon = c()
Retweets_cost = c()
Retweets_gamma = c()
Retweets_epsilon = c()



#Read in train data:
train_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_OT/Train/Amazon_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/mtism/Documents/TwitterProject/Analysis_Data_OT/Test/Amazon_test_OT.xlsx")


#Create binary 'has_link' variable in train/test data:
has_link = grepl("https://", train_data$Content)
train_data["has_link"] = has_link
train_data$has_link = ifelse(train_data$has_link == TRUE, 1, 0)
#Variable must be considered a factor for this model type:
train_data$has_link = factor(train_data$has_link)

has_link2 = grepl("https://", test_data$Content)
test_data["has_link"] = has_link2
test_data$has_link = ifelse(test_data$has_link == TRUE, 1, 0)
#Variable must be considered a factor for this model type:
test_data$has_link = factor(test_data$has_link)

#Create binary 'has_hash' variable in train/test data:
train_data["has_hash"] = ifelse(train_data$num_hashtags > 0, 1, 0)
test_data["has_hash"] = ifelse(test_data$num_hashtags > 0, 1, 0)
#Ensure the variable is considered a factor:
train_data$has_hash = factor(train_data$has_hash)
test_data$has_hash = factor(test_data$has_hash)

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


###########################################################################################################
#Model 1: Sentiment
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = tune(svm, `Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparameters of best model:
Likes_cost = c(Likes_cost, bestmodel$cost)
Likes_gamma = c(Likes_gamma, bestmodel$gamma)
Likes_epsilon = c(Likes_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = tune(svm, `Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparameters of best model:
Likes_cost = c(Likes_cost, bestmodel$cost)
Likes_gamma = c(Likes_gamma, bestmodel$gamma)
Likes_epsilon = c(Likes_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = tune(svm, `Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparameters of best model:
Likes_cost = c(Likes_cost, bestmodel$cost)
Likes_gamma = c(Likes_gamma, bestmodel$gamma)
Likes_epsilon = c(Likes_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3= train_data$topic.3,
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
model_ = tune(svm, `Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3= test_data$topic.3,
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
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparameters of best model:
Likes_cost = c(Likes_cost, bestmodel$cost)
Likes_gamma = c(Likes_gamma, bestmodel$gamma)
Likes_epsilon = c(Likes_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Likes"
model_ = tune(svm, `Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparameters of best model:
Likes_cost = c(Likes_cost, bestmodel$cost)
Likes_gamma = c(Likes_gamma, bestmodel$gamma)
Likes_epsilon = c(Likes_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3= train_data$topic.3,
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
model_ = tune(svm, `Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3= test_data$topic.3,
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
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparameters of best model:
Likes_cost = c(Likes_cost, bestmodel$cost)
Likes_gamma = c(Likes_gamma, bestmodel$gamma)
Likes_epsilon = c(Likes_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3= train_data$topic.3,
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
model_ = tune(svm, `Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3= test_data$topic.3,
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
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparameters of best model:
Likes_cost = c(Likes_cost, bestmodel$cost)
Likes_gamma = c(Likes_gamma, bestmodel$gamma)
Likes_epsilon = c(Likes_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Likes` = train_data$`Number of Likes`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3= train_data$topic.3,
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
model_ = tune(svm, `Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3= test_data$topic.3,
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
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

#Store hyperparameters of best model:
Likes_cost = c(Likes_cost, bestmodel$cost)
Likes_gamma = c(Likes_gamma, bestmodel$gamma)
Likes_epsilon = c(Likes_epsilon, bestmodel$epsilon)


#RETWEETS BELOW
##############################################################################################################
##############################################################################################################
##############################################################################################################
#Model 1: Sentiment
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = tune(svm, `Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparameters of best model:
Retweets_cost = c(Retweets_cost, bestmodel$cost)
Retweets_gamma = c(Retweets_gamma, bestmodel$gamma)
Retweets_epsilon = c(Retweets_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = tune(svm, `Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparameters of best model:
Retweets_cost = c(Retweets_cost, bestmodel$cost)
Retweets_gamma = c(Retweets_gamma, bestmodel$gamma)
Retweets_epsilon = c(Retweets_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = tune(svm, `Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparameters of best model:
Retweets_cost = c(Retweets_cost, bestmodel$cost)
Retweets_gamma = c(Retweets_gamma, bestmodel$gamma)
Retweets_epsilon = c(Retweets_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3= train_data$topic.3,
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
model_ = tune(svm, `Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3= test_data$topic.3,
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
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparameters of best model:
Retweets_cost = c(Retweets_cost, bestmodel$cost)
Retweets_gamma = c(Retweets_gamma, bestmodel$gamma)
Retweets_epsilon = c(Retweets_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative)
colnames(train_data_xy)[1] = "Number of Retweets"
model_ = tune(svm, `Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative) #format test input(s) properly
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparameters of best model:
Retweets_cost = c(Retweets_cost, bestmodel$cost)
Retweets_gamma = c(Retweets_gamma, bestmodel$gamma)
Retweets_epsilon = c(Retweets_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3= train_data$topic.3,
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
model_ = tune(svm, `Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3= test_data$topic.3,
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
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparameters of best model:
Retweets_cost = c(Retweets_cost, bestmodel$cost)
Retweets_gamma = c(Retweets_gamma, bestmodel$gamma)
Retweets_epsilon = c(Retweets_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3= train_data$topic.3,
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
model_ = tune(svm, `Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3= test_data$topic.3,
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
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparameters of best model:
Retweets_cost = c(Retweets_cost, bestmodel$cost)
Retweets_gamma = c(Retweets_gamma, bestmodel$gamma)
Retweets_epsilon = c(Retweets_epsilon, bestmodel$epsilon)


###########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)

#Tune 2000 models to determine optimal hyperparameters
train_data_xy = data.frame(`Number of Retweets` = train_data$`Number of Retweets`,
                           has_link = train_data$has_link,
                           has_hash = train_data$has_hash,
                           sent_cat.Neutral = train_data$sent_cat.Neutral,
                           sent_cat.Negative = train_data$sent_cat.Negative,
                           topic.1 = train_data$topic.1,
                           topic.2 = train_data$topic.2,
                           topic.3= train_data$topic.3,
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
model_ = tune(svm, `Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13 + topic.14 + topic.15 + topic.16, data = train_data_xy,
              ranges = list(epsilon=seq(0.1,1,0.1), cost=c(1:100), gamma = c(0.001, 0.1)))


#Find the best model out of all trained during tuning:
bestmodel = model_$best.model

#Generate predictions:
test_data_x = data.frame(has_link = test_data$has_link,
                         has_hash = test_data$has_hash,
                         sent_cat.Neutral = test_data$sent_cat.Neutral,
                         sent_cat.Negative = test_data$sent_cat.Negative,
                         topic.1 = test_data$topic.1,
                         topic.2 = test_data$topic.2,
                         topic.3= test_data$topic.3,
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
preds_ = predict(bestmodel, test_data_x)

#Calculate MAE:
mae_ = MAE(preds_, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

#Store hyperparameters of best model:
Retweets_cost = c(Retweets_cost, bestmodel$cost)
Retweets_gamma = c(Retweets_gamma, bestmodel$gamma)
Retweets_epsilon = c(Retweets_epsilon, bestmodel$epsilon)


#######################
#Save results
df = data.frame(MAE_Likes = Likes_MAE,
                cost_Likes = Likes_cost,
                eps_Likes = Likes_epsilon,
                gamma_Likes = Likes_gamma,
                MAE_Retweets = Retweets_MAE,
                cost_Retweets = Retweets_cost,
                eps_Retweets = Retweets_epsilon,
                gamma_Retweets = Retweets_gamma)

write_xlsx(df, "C:\\Users\\mtism\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\SVR_Amazon_OT2.xlsx")
