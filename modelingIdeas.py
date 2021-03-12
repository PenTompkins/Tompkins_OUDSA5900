#First attempt at Twitter predictions, just trying to get a plan put together for the proposal.
#Overall, this is program: 2

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from scipy.stats import multivariate_normal as mvn
import nltk
import os
import random
import string

import os, sys, email,re

#Read in the Amazon data: Below line of code will need to be reconfigured for your filepath
amazon_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Amazon_Dec_1_2020.xlsx')
#print(amazon_data.shape)

#Remove retweets from the Amazon account, as they aren't technically Amazon account tweets
patternDel = "^RT @"
filter1 = amazon_data["Content"].str.contains(patternDel)
#print(filter1)

amazon_tweets = amazon_data[~filter1].copy()
#print(amazon_tweets.shape) #3174, just like in R


#Creating the 'contains link' binary variable
#with_link = []
patternDel2 = "https://"
filter2 = amazon_tweets["Content"].str.contains(patternDel2)

#Add the boolean variable 'with_link' to the data:
amazon_tweets["with_link"] = filter2
#print(amazon_tweets)

#Adding the boolean variable 'is_IRT' to indicate whether the tweet is in response to someone
patternDel3 = "^@"
filter3 = amazon_tweets["Content"].str.contains(patternDel3)
amazon_tweets["is_IRT"] = filter3
#print(amazon_tweets)

#Need to check and make sure I grouped correctly (check mean values)
with_links = amazon_tweets[amazon_tweets["with_link"] == True]
without_links = amazon_tweets[amazon_tweets["with_link"] == False]

#print(with_links.shape)

#print("With_links mean likes:")
#print(np.mean(with_links["Number of Likes"]))
#print("With_links avg retweets:")
#print(np.mean(with_links["Number of Retweets"]))
#print("Without_links mean likes:")
#print(np.mean(without_links["Number of Likes"]))
#print("Without_links avg retweets:")
#print(np.mean(without_links["Number of Retweets"]))
#Seems to match up with R's results


#Attempt 1 at using with_link, is_IRT, and text data to predict likes:

#Now, I need to split the data into test/train
from sklearn.model_selection import train_test_split
x_trainData, x_testData, y_trainData, y_testData = train_test_split(amazon_tweets, amazon_tweets["Number of Likes"], test_size = 0.25, random_state=1)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor

#Extract tweets from training data for vectorization
train_tweets = x_trainData["Content"]

vect = TfidfVectorizer(stop_words={'english'})

#Perform tfidf on training data tweets
numericTweets = vect.fit_transform(train_tweets)

#Extract features from training data that will actually be used
x_train = pd.DataFrame(x_trainData[["with_link","is_IRT"]])

#include the tfidf scores
x_train["content_score"] = numericTweets

#Extract the feature of interest, likes, for y_train:
y_train = y_trainData


#Change boolean 'True' and 'False' values to 0's and 1's
x_train[["with_link", "is_IRT"]] *= 1


#Train random forest model
#Can't figure out how to get tfidf results incorporated in this model at this time, two boolean variables will be used only
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state=1)
rf.fit(x_train[["with_link", "is_IRT"]], y_train)

#print("It was able to train")

#Format the test data
x_testData[["with_link", "is_IRT"]] *= 1
x_test = x_testData[["with_link", "is_IRT"]]

#Test how well it did predicting likes on with_link and is_IRT alone 
preds = rf.predict(x_test)

y_test = y_testData

errors = abs(preds - y_test)

print("MAE for likes using only with_Link and is_IRT: ", round(np.mean(errors), 4)) #MAE = 29.9382

#################################################################################################################################################################

#Might as well also see how this performs in predicting retweets:
x_trainData2, x_testData2, y_trainData2, y_testData2 = train_test_split(amazon_tweets, amazon_tweets["Number of Retweets"], test_size = 0.25, random_state=2)

#Extract features from training data that will actually be used
x_train2 = pd.DataFrame(x_trainData2[["with_link","is_IRT"]])

#Format these variables to be numerically binary:
x_train2[["with_link", "is_IRT"]] *= 1

#Train random forest model to predict retweets based on with_link and is_IRT
rf2 = RandomForestRegressor(n_estimators = 1000, random_state=2)
rf2.fit(x_train2[["with_link", "is_IRT"]], y_trainData2)

#Format the test data
x_testData2[["with_link", "is_IRT"]] *= 1
x_test2 = x_testData2[["with_link", "is_IRT"]]

#Test how well it did predicting retweets on with_link and is_IRT alone 
preds2 = rf2.predict(x_test2)

errors2 = abs(preds2 - y_testData2)

print("MAE for retweets using only with_Link and is_IRT: ", round(np.mean(errors2), 4)) #MAE = 4.4733


##################################################################################################################################################################
#Might as well also examine if the number of hashtags improves predictions:
#I've already tested and this yields the same MAE as version 1 when you don't include num_hashtags
#Also, I'm doing this after AMZN_hashtags.py, I'm just going back and adding on to this
x_trainData3, x_testData3, y_trainData3, y_testData3 = train_test_split(amazon_tweets, amazon_tweets["Number of Likes"], test_size = 0.25, random_state=1)

#Extract features from training data that will actually be used
x_train3 = pd.DataFrame(x_trainData3[["with_link","is_IRT", "num_hashtags"]])

#Format these variables to be numerically binary:
x_train3[["with_link", "is_IRT"]] *= 1

#Train random forest model to predict likes based on with_link, is_IRT, and num_hashtags
rf3 = RandomForestRegressor(n_estimators = 1000, random_state=1)
rf3.fit(x_train3[["with_link", "is_IRT", "num_hashtags"]], y_trainData3)

#Format the test data
x_testData3[["with_link", "is_IRT"]] *= 1
x_test3 = x_testData3[["with_link", "is_IRT", "num_hashtags"]]

#Test how well it did predicting retweets on with_link and is_IRT alone 
preds3 = rf3.predict(x_test3)

errors3 = abs(preds3 - y_testData3)

print("MAE for likes using with_Link, is_IRT, and num_hashtags: ", round(np.mean(errors3), 4)) #MAE = 31.18, slightly worsened predictions for likes

##################################################################################################################################################################
#Testing to see how including num_hashtags effects performance in predictions of retweets:
#I've already checked to make sure that this yields the same results as model2 when not including num_hashtags
x_trainData4, x_testData4, y_trainData4, y_testData4 = train_test_split(amazon_tweets, amazon_tweets["Number of Retweets"], test_size = 0.25, random_state=2)

#Extract features from training data that will actually be used
x_train4 = pd.DataFrame(x_trainData4[["with_link","is_IRT", "num_hashtags"]])

#Format these variables to be numerically binary:
x_train4[["with_link", "is_IRT"]] *= 1

#Train random forest model to predict retweets based on with_link and is_IRT
rf4 = RandomForestRegressor(n_estimators = 1000, random_state=2)
rf4.fit(x_train4[["with_link", "is_IRT", "num_hashtags"]], y_trainData4)

#Format the test data
x_testData4[["with_link", "is_IRT"]] *= 1
x_test4 = x_testData4[["with_link", "is_IRT", "num_hashtags"]]

#Test how well it did predicting retweets on with_link and is_IRT alone 
preds4 = rf4.predict(x_test4)

errors4 = abs(preds4 - y_testData4)

print("MAE for retweets using, with_Link, is_IRT, and num_hashtags: ", round(np.mean(errors4), 4)) #MAE = 4.4733












