# -*- coding: utf-8 -*-
#This file is being created to create a standardized dataset consisting of tweets from all companies
#
#Overall, this is program: 36

import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras
import math
from scipy import stats

#Below line of code will need to be reconfigured to match the beginning of filepath to your data
path = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\'
commonEnd = '_Dec_1_2020.xlsx'
files = ['BMW', 'CocaCola', 'Disney', 'Google', 'McDonalds', 'MercedesBenz', 'Microsoft', 'Samsung', 'Toyota']

#Begin by reading in Amazon data: Below line of code will need to be reconfigured for your filepath
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Amazon_Dec_1_2020.xlsx')
company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]

#Remove retweets from the company account, as they aren't technically company account tweets
patternDel = "^RT @"
filter1 = company_data["Content"].str.contains(patternDel)
company_tweets = company_data[~filter1].copy()

#Save tweet category (IRT vs. OT):
#Perform initial separation based on "^@" regex:
initIRT = [bool(re.search("^@", i)) for i in company_tweets["Content"]]
initOT = [not elem for elem in initIRT]
#print(initOT)

#Create IRT and OT variables in the data:
company_tweets["IRT"] = initIRT
company_tweets["OT"] = initOT

#print(company_tweets["IRT"])
#print(company_tweets["OT"])

#Fill in NAs under the 'In Reply To' field with "OT":
company_tweets["In Reply To"] = company_tweets["In Reply To"].replace(np.nan, "OT", regex=True)
#print(company_tweets["In Reply To"].head(5))
                

#Clean up initial OT/IRT separation:
def cleanSplit(tweets, text1, text2, text3, text4, text5):
    for i in range(len(tweets[text1])):
        if tweets.iloc[i, tweets.columns.get_loc(text2)] == True: #if the tweet was marked IRT initially
            if tweets.iloc[i, tweets.columns.get_loc(text3)] == tweets.iloc[i, tweets.columns.get_loc(text4)]: #but it's in reply to the company
                j = i #then index our current position so that we may examine the 'next' (technically previous) tweet
                while tweets.iloc[j, tweets.columns.get_loc(text3)] == tweets.iloc[j, tweets.columns.get_loc(text4)]: #while this continues to be true
                    j = j + 1 #keep following the chain
                if tweets.iloc[j, tweets.columns.get_loc(text3)] == "OT": #if an official tweet is at the end of the chain
                    tweets.iat[i, 19] = False #then the original tweet is part of an official thread, not true IRT
                    tweets.iat[i, 20] = True
                    #print(tweets.iloc[i, tweets.columns.get_loc(text1)])
    return tweets

company_tweets2 = cleanSplit(company_tweets, "Content", "IRT", "In Reply To", "Author", "OT")

#Remove tweets not interpretable to the English reader, as those audiences may behave differently
English_tweets = company_tweets2[company_tweets2["Language"] == "en"]
und_tweets = company_tweets2[company_tweets2["Language"] == "und"]
all_eligible = pd.concat([English_tweets, und_tweets], axis=0)

#Function to standardize tweet performance metrics:
def standardize_metrics(df, metric1, metric2):
    z1 = stats.zscore(df[metric1])
    z2 = stats.zscore(df[metric2])
    df1 = df.copy()
    df1[metric1] = z1
    df1[metric2] = z2
    return df1

std_amzn = standardize_metrics(all_eligible, "Number of Likes", "Number of Retweets")
print("Standardized Amazon has %s tweets" % len(std_amzn))
std_data = std_amzn.copy()
#########################
for file in files:
    fpath = path + file + commonEnd
    company_data = pd.read_excel(fpath)
    company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]
    #Remove retweets:
    patternDel = "^RT @"
    filter1 = company_data["Content"].str.contains(patternDel)
    company_tweets = company_data[~filter1].copy()
    #Save tweet category:
    initIRT = [bool(re.search("^@", i)) for i in company_tweets["Content"]]
    initOT = [not elem for elem in initIRT]
    
    #Create IRT and OT variables in the data:
    company_tweets["IRT"] = initIRT
    company_tweets["OT"] = initOT
    
    #Fill in NAs under the 'In Reply To' field with "OT":
    company_tweets["In Reply To"] = company_tweets["In Reply To"].replace(np.nan, "OT", regex=True)
    
    #Finalize OT vs. IRT split categories:
    company_tweets2 = cleanSplit(company_tweets, "Content", "IRT", "In Reply To", "Author", "OT")
    
    
    #Remove tweets not interpretable to an English reader, as those markets may behave differently
    English_tweets = company_tweets2[company_tweets2["Language"] == "en"]
    und_tweets = company_tweets2[company_tweets2["Language"] == "und"]
    all_eligible = pd.concat([English_tweets, und_tweets], axis=0)
    
    #Standardize tweet performance metrics:
    std_comp = standardize_metrics(all_eligible, "Number of Likes", "Number of Retweets")
    print("Standardized %s has %s tweets" % (company_name, len(std_comp)))
    #std_data.append(std_comp)
    std_data = pd.concat([std_data, std_comp], axis=0)
    
#Save the resulting file:
std_data.to_excel("std_Data.xlsx")
print("All done")
                                    















