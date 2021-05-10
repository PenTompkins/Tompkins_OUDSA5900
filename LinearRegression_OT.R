#This program is being created to generate linear regression predictions using OT data
#
#Overall, this is program: 85

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

#Read in train data:
train_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Train/Toyota_train_OT.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_OT/Test/Toyota_test_OT.xlsx")


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

#Ensure that 'topic' is considered a categorical variable:
train_data$topic = factor(train_data$topic)
test_data$topic = factor(test_data$topic)

###########################################################################################################
#Model 1: Link

#Create the model:
model_ = lm(`Number of Likes` ~ has_link, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 2: Hashtag

#Create the model:
model_ = lm(`Number of Likes` ~ has_hash, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

###########################################################################################################
#Model 3: Topic

#Create the model:
model_ = lm(`Number of Likes` ~ topic, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

###########################################################################################################
#Model 4: (Link, Hashtag)

#Create the model:
model_ = lm(`Number of Likes` ~ has_link + has_hash, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

###########################################################################################################
#Model 5: (Link, Topic)

#Create the model:
model_ = lm(`Number of Likes` ~ has_link + topic, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

###########################################################################################################
#Model 6: (Hashtag, Topic)

#Create the model:
model_ = lm(`Number of Likes` ~ has_hash + topic, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 7: (Link, Hashtag, Topic)

#Create the model:
model_ = lm(`Number of Likes` ~ has_link + has_hash + topic, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)



#RETWEETS BELOW
#######################################################################################################
#############################################################################################################
#########################################################################################################
#Model 1: Link

#Create the model:
model_ = lm(`Number of Retweets` ~ has_link, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 2: Hashtag

#Create the model:
model_ = lm(`Number of Retweets` ~ has_hash, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

###########################################################################################################
#Model 3: Topic

#Create the model:
model_ = lm(`Number of Retweets` ~ topic, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

###########################################################################################################
#Model 4: (Link, Hashtag)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_link + has_hash, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

###########################################################################################################
#Model 5: (Link, Topic)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_link + topic, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

###########################################################################################################
#Model 6: (Hashtag, Topic)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_hash + topic, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 7: (Link, Hashtag, Topic)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_link + has_hash + topic, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


#################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                MAE_Retweets = Retweets_MAE)

write_xlsx(df, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_OT\\Storage\\LR.xlsx")



































