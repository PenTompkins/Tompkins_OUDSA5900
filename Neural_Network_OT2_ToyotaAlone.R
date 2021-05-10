#This file is created to perform NN for Toyota OT data alone
#with sentiment added in

library(readxl)
library(fpp2)
library(ggpubr)
library(MLmetrics)
library(writexl)
library(tidyverse)
library(GGally)
library(neuralnet)
library(caret)


#Never hurts:
set.seed(1)

#For saving results:
Likes_MAE = c()
Retweets_MAE = c()
hiddenlayers_Likes = c()
hiddenlayers_Retweets = c()
actfuncs_Likes = c()
actfuncs_Retweets = c()

#Read in train data:
train_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Train/Toyota_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Owner/Documents/Twitter Project/Analysis_Data_OT/Test/Toyota_test_OT.xlsx")

#Scale 'Number of Likes' variable in train data:
max_train = max(train_data$`Number of Likes`)
min_train = min(train_data$`Number of Likes`)
train_data$scaledLikes = scale(train_data$`Number of Likes`, center = min_train, scale = max_train - min_train)

#Scale 'Number of Likes' variable in test data:
max_test = max(test_data$`Number of Likes`)
min_test = min(test_data$`Number of Likes`)
test_data$scaledLikes = scale(test_data$`Number of Likes`, center = min_test, scale = max_test - min_test)

#Create binary 'has_link' variable in train/test data:
has_link = grepl("https://", train_data$Content)
train_data["has_link"] = has_link
train_data$has_link = ifelse(train_data$has_link == TRUE, 1, 0)
#train_data$has_link = factor(train_data$has_link)
#levels(train_data$has_link) = c(levels(train_data$has_link), "0")

has_link2 = grepl("https://", test_data$Content)
test_data["has_link"] = has_link2
test_data$has_link = ifelse(test_data$has_link == TRUE, 1, 0)
#test_data$has_link = factor(test_data$has_link)
#levels(test_data$has_link) = c(levels(test_data$has_link), "0")

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
#Drop column of dummy variables:
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
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(2), layer2=c(0:1), layer3=c(0:1))
nn = train(scaledLikes ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledLikes, 'scaled:scale') + attr(train_data$scaledLikes, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Likes`)
MAE_

######################################
#Model 1t: Sentiment using tanh activation function:

#First, need to rescale data on [-1, 1] instead of [0,1] for tanh:
scale11 = function(x){
  (2 * ((x - min(x))/(max(x)-min(x)))) - 1
}

train_data["scaledLikes2"] = scale11(train_data$`Number of Likes`)
test_data["scaledLikes2"] = scale11(test_data$`Number of Likes`)

set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(2), layer2=c(0:1), layer3=c(0:1))
nnt = train(scaledLikes2 ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
#First, set diffx and minx to values from transforming train_data
diffx = max(train_data$`Number of Likes`) - min(train_data$`Number of Likes`)
minx = min(train_data$`Number of Likes`)
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Likes`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "Default") #save that the default activation function was best
  Likes_MAE = c(Likes_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "tanh") #save that the tanh activation function was best
  Likes_MAE = c(Likes_MAE, MAE_t) #save the resulting MAE
}



##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(3), layer2=c(0:2), layer3=c(0:1))
nn = train(scaledLikes ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledLikes, 'scaled:scale') + attr(train_data$scaledLikes, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Likes`)
MAE_

######################################
#Model 2t: (Link, Sentiment) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(3), layer2=c(0:2), layer3=c(0:1))
nnt = train(scaledLikes2 ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Likes`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "Default") #save that the default activation function was best
  Likes_MAE = c(Likes_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "tanh") #save that the tanh activation function was best
  Likes_MAE = c(Likes_MAE, MAE_t) #save the resulting MAE
}


##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(3), layer2=c(0:2), layer3=c(0:1))
nn = train(scaledLikes ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledLikes, 'scaled:scale') + attr(train_data$scaledLikes, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Likes`)
MAE_

######################################
#Model 3t: (Hashtag, Sentiment) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(3), layer2=c(0:2), layer3=c(0:1))
nnt = train(scaledLikes2 ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Likes`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "Default") #save that the default activation function was best
  Likes_MAE = c(Likes_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "tanh") #save that the tanh activation function was best
  Likes_MAE = c(Likes_MAE, MAE_t) #save the resulting MAE
}



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(15), layer2=c(4:8), layer3=c(0:3))
nn = train(scaledLikes ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledLikes, 'scaled:scale') + attr(train_data$scaledLikes, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Likes`)
MAE_

######################################
#Model 4t: (Sentiment, Topic) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(15), layer2=c(4:8), layer3=c(0:3))
nnt = train(scaledLikes2 ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Likes`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "Default") #save that the default activation function was best
  Likes_MAE = c(Likes_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "tanh") #save that the tanh activation function was best
  Likes_MAE = c(Likes_MAE, MAE_t) #save the resulting MAE
}



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(4), layer2=c(0:3), layer3=c(0:2))
nn = train(scaledLikes ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledLikes, 'scaled:scale') + attr(train_data$scaledLikes, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Likes`)
MAE_

######################################
#Model 5t: (Link, Hashtag, Sentiment) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(4), layer2=c(0:3), layer3=c(0:2))
nnt = train(scaledLikes2 ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Likes`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "Default") #save that the default activation function was best
  Likes_MAE = c(Likes_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "tanh") #save that the tanh activation function was best
  Likes_MAE = c(Likes_MAE, MAE_t) #save the resulting MAE
}



##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(16), layer2=c(4:8), layer3=c(0:3))
nn = train(scaledLikes ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledLikes, 'scaled:scale') + attr(train_data$scaledLikes, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Likes`)
MAE_

######################################
#Model 6t: (Link, Sentiment, Topic) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(16), layer2=c(4:8), layer3=c(0:3))
nnt = train(scaledLikes2 ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Likes`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "Default") #save that the default activation function was best
  Likes_MAE = c(Likes_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "tanh") #save that the tanh activation function was best
  Likes_MAE = c(Likes_MAE, MAE_t) #save the resulting MAE
}



##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(16), layer2=c(4:8), layer3=c(0:3))
nn = train(scaledLikes ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledLikes, 'scaled:scale') + attr(train_data$scaledLikes, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Likes`)
MAE_

######################################
#Model 7t: (Hashtag, Sentiment, Topic) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(16), layer2=c(4:8), layer3=c(0:3))
nnt = train(scaledLikes2 ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Likes`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "Default") #save that the default activation function was best
  Likes_MAE = c(Likes_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "tanh") #save that the tanh activation function was best
  Likes_MAE = c(Likes_MAE, MAE_t) #save the resulting MAE
}



##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(17), layer2=c(4:8), layer3=c(0:4))
nn = train(scaledLikes ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledLikes, 'scaled:scale') + attr(train_data$scaledLikes, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Likes`)
MAE_

######################################
#Model 8t: (Link, Hashtag, Sentiment, Topic) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(17), layer2=c(4:8), layer3=c(0:4))
nnt = train(scaledLikes2 ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Likes`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "Default") #save that the default activation function was best
  Likes_MAE = c(Likes_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Likes = c(hiddenlayers_Likes, hc) #save the hidden layer configurations
  actfuncs_Likes = c(actfuncs_Likes, "tanh") #save that the tanh activation function was best
  Likes_MAE = c(Likes_MAE, MAE_t) #save the resulting MAE
}



#RETWEETS BELOW
#################################################################################################################
#################################################################################################################
#################################################################################################################
#Scale 'Number of Retweets' variable in train data:
max_train = max(train_data$`Number of Retweets`)
min_train = min(train_data$`Number of Retweets`)
train_data$scaledRetweets = scale(train_data$`Number of Retweets`, center = min_train, scale = max_train - min_train)

#Scale 'Number of Retweets' variable in test data:
max_test = max(test_data$`Number of Retweets`)
min_test = min(test_data$`Number of Retweets`)
test_data$scaledRetweets = scale(test_data$`Number of Retweets`, center = min_test, scale = max_test - min_test)

###############################################################################################################
#Model 1: Sentiment
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(2), layer2=c(0:1), layer3=c(0:1))
nn = train(scaledRetweets ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledRetweets, 'scaled:scale') + attr(train_data$scaledRetweets, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Retweets`)
MAE_

######################################
#Model 1t: Sentiment using tanh activation function:

#First, need to rescale data on [-1, 1] instead of [0,1] for tanh:
scale11 = function(x){
  (2 * ((x - min(x))/(max(x)-min(x)))) - 1
}

train_data["scaledRetweets2"] = scale11(train_data$`Number of Retweets`)
test_data["scaledRetweets2"] = scale11(test_data$`Number of Retweets`)

set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(2), layer2=c(0:1), layer3=c(0:1))
nnt = train(scaledRetweets2 ~ sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
#First, set diffx and minx to values from transforming train_data
diffx = max(train_data$`Number of Retweets`) - min(train_data$`Number of Retweets`)
minx = min(train_data$`Number of Retweets`)
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Retweets`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "Default") #save that the default activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "tanh") #save that the tanh activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_t) #save the resulting MAE
}



##########################################################################################################
#Model 2: (Link, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(3), layer2=c(0:2), layer3=c(0:1))
nn = train(scaledRetweets ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledRetweets, 'scaled:scale') + attr(train_data$scaledRetweets, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Retweets`)
MAE_

######################################
#Model 2t: (Link, Sentiment) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(3), layer2=c(0:2), layer3=c(0:1))
nnt = train(scaledRetweets2 ~ has_link + sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Retweets`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "Default") #save that the default activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "tanh") #save that the tanh activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_t) #save the resulting MAE
}


##########################################################################################################
#Model 3: (Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(3), layer2=c(0:2), layer3=c(0:1))
nn = train(scaledRetweets ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledRetweets, 'scaled:scale') + attr(train_data$scaledRetweets, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Retweets`)
MAE_

######################################
#Model 3t: (Hashtag, Sentiment) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(3), layer2=c(0:2), layer3=c(0:1))
nnt = train(scaledRetweets2 ~ has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Retweets`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "Default") #save that the default activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "tanh") #save that the tanh activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_t) #save the resulting MAE
}



##########################################################################################################
#Model 4: (Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(15), layer2=c(4:8), layer3=c(0:3))
nn = train(scaledRetweets ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledRetweets, 'scaled:scale') + attr(train_data$scaledRetweets, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Retweets`)
MAE_

######################################
#Model 4t: (Sentiment, Topic) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(15), layer2=c(4:8), layer3=c(0:3))
nnt = train(scaledRetweets2 ~ sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Retweets`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "Default") #save that the default activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "tanh") #save that the tanh activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_t) #save the resulting MAE
}



##########################################################################################################
#Model 5: (Link, Hashtag, Sentiment)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(4), layer2=c(0:3), layer3=c(0:2))
nn = train(scaledRetweets ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledRetweets, 'scaled:scale') + attr(train_data$scaledRetweets, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Retweets`)
MAE_

######################################
#Model 5t: (Link, Hashtag, Sentiment) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(4), layer2=c(0:3), layer3=c(0:2))
nnt = train(scaledRetweets2 ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Retweets`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "Default") #save that the default activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "tanh") #save that the tanh activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_t) #save the resulting MAE
}



##########################################################################################################
#Model 6: (Link, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(16), layer2=c(4:8), layer3=c(0:3))
nn = train(scaledRetweets ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledRetweets, 'scaled:scale') + attr(train_data$scaledRetweets, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Retweets`)
MAE_

######################################
#Model 6t: (Link, Sentiment, Topic) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(16), layer2=c(4:8), layer3=c(0:3))
nnt = train(scaledRetweets2 ~ has_link + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Retweets`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "Default") #save that the default activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "tanh") #save that the tanh activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_t) #save the resulting MAE
}



##########################################################################################################
#Model 7: (Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(16), layer2=c(4:8), layer3=c(0:3))
nn = train(scaledRetweets ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledRetweets, 'scaled:scale') + attr(train_data$scaledRetweets, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Retweets`)
MAE_

######################################
#Model 7t: (Hashtag, Sentiment, Topic) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(16), layer2=c(4:8), layer3=c(0:3))
nnt = train(scaledRetweets2 ~ has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Retweets`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "Default") #save that the default activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "tanh") #save that the tanh activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_t) #save the resulting MAE
}



##########################################################################################################
#Model 8: (Link, Hashtag, Sentiment, Topic)
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(17), layer2=c(4:8), layer3=c(0:4))
nn = train(scaledRetweets ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
           tuneGrid = grid)
print(nn)
#Generate predictions:
preds_nn = compute(nn$finalModel, test_data)$net.result

#Back transform predictions:
bt_p = preds_nn * attr(train_data$scaledRetweets, 'scaled:scale') + attr(train_data$scaledRetweets, 'scaled:center')
bt_p

#Compute MAE:
MAE_ = MAE(bt_p, test_data$`Number of Retweets`)
MAE_

######################################
#Model 8t: (Link, Hashtag, Sentiment, Topic) using tanh activation function:
set.seed(1)
control = trainControl(method = "repeatedcv", number = 10, repeats = 3)
grid = expand.grid(layer1=c(17), layer2=c(4:8), layer3=c(0:4))
nnt = train(scaledRetweets2 ~ has_link + has_hash + sent_cat.Neutral + sent_cat.Negative + topic.1 + topic.2 + topic.3 + topic.4 + topic.5 + topic.6 + topic.7 + topic.8 + topic.9 + topic.10 + topic.11 + topic.12 + topic.13, data = train_data, trControl=control, method="neuralnet",
            tuneGrid = grid, act.fct = "tanh")
print(nnt)

#Generate predictions:
preds_nnt = compute(nnt$finalModel, test_data)$net.result

#Back transform predictions:
bt_pt = (preds_nnt + 1)/2 * (diffx) + minx
MAE_t = MAE(bt_pt, test_data$`Number of Retweets`)
MAE_t 


#Determine which model was best:
if (MAE_ < MAE_t){ #if using default activation function was better than tanh
  hlayers = nn$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "Default") #save that the default activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_) #save the resulting MAE
}

if (MAE_ > MAE_t){ #if using tanh activation function was better than default
  hlayers = nnt$bestTune
  h1 = hlayers[,1]
  h2 = hlayers[,2]
  h3 = hlayers[,3]
  hc = c(h1,h2,h3)
  hiddenlayers_Retweets = c(hiddenlayers_Retweets, hc) #save the hidden layer configurations
  actfuncs_Retweets = c(actfuncs_Retweets, "tanh") #save that the tanh activation function was best
  Retweets_MAE = c(Retweets_MAE, MAE_t) #save the resulting MAE
}


#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                MAE_Retweets = Retweets_MAE,
                Act.Funcs_Likes = actfuncs_Likes,
                Act.Funcs_Retweets = actfuncs_Retweets)

df2 = data.frame(hiddenlayers_Likes = hiddenlayers_Likes,
                 hiddenlayers_rt = hiddenlayers_Retweets)

write_xlsx(df, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelingResults_IRT\\Storage\\NN_Toyota_OT2.xlsx")
write_xlsx(df2, "C:\\Users\\Owner\\Documents\\Twitter Project\\ModelingResults_IRT\\Storage\\NN_Toyota_OTlayers2.xlsx")
