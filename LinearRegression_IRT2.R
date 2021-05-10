#This program is being created to generate linear regression predictions using IRT2 data
#Meaning, 'num_words' has been added in
#
#Overall, this is program: 105

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
train_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_IRT/Train/Microsoft_train_IRT2.xlsx")

#Read in test data:
test_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Analysis_Data_IRT/Test/Microsoft_test_IRT2.xlsx")


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

#Create categorical 'Sentiment' variable:
train_data["sent_cat"] = ifelse(train_data$Sentiment > 0, "Positive",
                                ifelse(train_data$Sentiment == 0, "Neutral","Negative"))

test_data["sent_cat"] = ifelse(test_data$Sentiment > 0, "Positive",
                               ifelse(test_data$Sentiment == 0, "Neutral","Negative"))

#Ensure this is considered a categorical variable:
train_data$sent_cat = factor(train_data$sent_cat)
test_data$sent_cat = factor(test_data$sent_cat)

###########################################################################################################
#Model 0: num_words

#Create the model:
model_ = lm(`Number of Likes` ~ num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)



###########################################################################################################
#Model 1: (Link, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ has_link + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 2: (Hashtag, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ has_hash + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 3: (Sentiment, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ sent_cat + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 4: (Topic, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 5: (Link, Hashtag, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ has_link + has_hash + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 6: (Link, Sentiment, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ has_link + sent_cat + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 7: (Link, Topic, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ has_link + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 8: (Hashtag, Sentiment, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ has_hash + sent_cat + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 9: (Hashtag, Topic, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ has_hash + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)

###########################################################################################################
#Model 10: (Sentiment, Topic, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ sent_cat + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 11: (Link, Hashtag, Sentiment, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ has_link + has_hash + sent_cat + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 12: (Link, Hashtag, Topic, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ has_link + has_hash + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 13: (Link, Sentiment, Topic, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ has_link + sent_cat + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 14: (Hashtag, Sentiment, Topic, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ has_hash + sent_cat + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)


###########################################################################################################
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)

#Create the model:
model_ = lm(`Number of Likes` ~ has_link + has_hash + sent_cat + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Likes`)
mae_

#Store MAE for the model:
Likes_MAE = c(Likes_MAE, mae_)



#RETWEETS BELOW
##############################################################################################################
###############################################################################################################
##############################################################################################################

###########################################################################################################
#Model 0: num_words

#Create the model:
model_ = lm(`Number of Retweets` ~ num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###################################################################################################################
#Model 1: (Link, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_link + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 2: (Hashtag, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_hash + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 3: (Sentiment, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ sent_cat + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 4: (Topic, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 5: (Link, Hashtag, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_link + has_hash + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 6: (Link, Sentiment, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_link + sent_cat + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 7: (Link, Topic, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_link + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 8: (Hashtag, Sentiment, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_hash + sent_cat + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 9: (Hashtag, Topic, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_hash + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)

###########################################################################################################
#Model 10: (Sentiment, Topic, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ sent_cat + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 11: (Link, Hashtag, Sentiment, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_link + has_hash + sent_cat + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 12: (Link, Hashtag, Topic, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_link + has_hash + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 13: (Link, Sentiment, Topic, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_link + sent_cat + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 14: (Hashtag, Sentiment, Topic, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_hash + sent_cat + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


###########################################################################################################
#Model 15: (Link, Hashtag, Sentiment, Topic, num_words)

#Create the model:
model_ = lm(`Number of Retweets` ~ has_link + has_hash + sent_cat + topic + num_words, data = train_data)

#Generate predictions:
preds = predict(model_, newdata = test_data)

#Calculate MAE:
mae_ = MAE(preds, test_data$`Number of Retweets`)
mae_

#Store MAE for the model:
Retweets_MAE = c(Retweets_MAE, mae_)


##################################
#Save results:
df = data.frame(MAE_Likes = Likes_MAE,
                MAE_Retweets = Retweets_MAE)

write_xlsx(df, "C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\ModelingResults_IRT\\Storage\\LR2.xlsx")



