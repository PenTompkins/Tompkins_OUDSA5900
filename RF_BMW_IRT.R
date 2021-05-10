#This program is being created to generate random forest predictions for BMW IRT data
#
#Overall, this is program: 88

library(randomForest)
library(readxl)
library(fpp2)
library(ggpubr)
library(MLmetrics)
library(writexl)
library(caret)

#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
Likes_mtry = c()
Retweets_mtry = c()

#Read in train data:
train_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_IRT/Train/BMW_train_IRT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_IRT/Test/BMW_test_IRT.xlsx")


#Create binary 'has_link' variable in train/test data:
has_link = grepl("https://", train_data$Content)
train_data["has_link"] = has_link
train_data$has_link = ifelse(train_data$has_link == TRUE, 1, 0)

has_link2 = grepl("https://", test_data$Content)
test_data["has_link"] = has_link2
test_data$has_link = ifelse(test_data$has_link == TRUE, 1, 0)

#Create binary 'has_hash' (hashtag) variable in train/test data:
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


#############################################################################################################
#Model 1: Link
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1))
model_ = train(`Number of Likes` ~ has_link, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ has_link, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 2: Hashtag
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1))
model_ = train(`Number of Likes` ~ has_hash, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ has_hash, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 3: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:2))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 4: Topic
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:2))
model_ = train(`Number of Likes` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])

####################################################################################################################
#Model 5: Link, Hashtag
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:2))
model_ = train(`Number of Likes` ~ has_link + has_hash, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ has_link + has_hash, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 6: Link, Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 7: (Link, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Likes` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 8: Hashtag, Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 9: (Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Likes` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 10: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 11: Link, Hashtag, Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 12: (Link, Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 13: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 14: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 15: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Likes` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)
Likes_mtry = c(Likes_mtry, model_$bestTune[1,])




#RETWEETS BELOW
##############################################################################################################
##############################################################################################################
##############################################################################################################
#Model 1: Link
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1))
model_ = train(`Number of Retweets` ~ has_link, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ has_link, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 2: Hashtag
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1))
model_ = train(`Number of Retweets` ~ has_hash, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ has_hash, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 3: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:2))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 4: Topic
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:2))
model_ = train(`Number of Retweets` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])

####################################################################################################################
#Model 5: Link, Hashtag
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:2))
model_ = train(`Number of Retweets` ~ has_link + has_hash, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ has_link + has_hash, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 6: Link, Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 7: (Link, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Retweets` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ has_link + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 8: Hashtag, Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 9: (Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Retweets` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 10: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 11: Link, Hashtag, Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 12: (Link, Hashtag, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ has_link + has_hash + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 13: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 14: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


####################################################################################################################
#Model 15: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(mtry=c(1:3))
model_ = train(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7,
               data = train_data, method = "rf",
               tuneGrid = grid, importance = TRUE)

print(model_)

#Generate predictions:
preds = predict(model_$finalModel, newdata = model.matrix(`Number of Retweets` ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7, test_data))

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)
Retweets_mtry = c(Retweets_mtry, model_$bestTune[1,])


#######################
#Save results
df = data.frame(MAE_Likes = Likes_MAE,
                MAE_Retweets = Retweets_MAE,
                mtry_Likes = Likes_mtry,
                mtry_Retweets = Retweets_mtry)

write_xlsx(df, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\RF_BMW_IRT.xlsx")

