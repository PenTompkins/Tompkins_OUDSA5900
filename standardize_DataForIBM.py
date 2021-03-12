# -*- coding: utf-8 -*-
#This file is being created to create a standardized dataset consisting of tweets from all companies
#However, I only want to ultimately keep the tweets used in 'IBM_Sentiment.py', for all datasets
#
#Overall, this is program: 36.2

import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras
import math
from scipy import stats

#Below line of code will need to be reconfigured to match the beginning of your filepath to the data
path = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\'
commonEnd = '_Dec_1_2020.xlsx'
files = ['BMW', 'CocaCola', 'Disney', 'Google', 'McDonalds', 'MercedesBenz', 'Microsoft', 'Samsung', 'Toyota']

#Begin by reading in Amazon data: Below line of code will need to be reconfigured to match your filepath
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
#English_tweets = company_tweets2[company_tweets2["Language"] == "en"]
#und_tweets = company_tweets2[company_tweets2["Language"] == "und"]
#all_eligible = pd.concat([English_tweets, und_tweets], axis=0)

def treatOuts(df, text_field):
    z = np.abs(stats.zscore(df[text_field]))
    df1 = df[(z < 4)]
    return df1

#Remove tweets which were considered extreme outliers in 'IBM_Sentiment.py'
IRT_initial = company_tweets2[company_tweets2["IRT"] == True].copy()
OT_initial = company_tweets2[company_tweets2["OT"] == True].copy()
IRT_noOuts = treatOuts(IRT_initial, "Number of Likes")
OT_noOuts = treatOuts(OT_initial, "Number of Likes")
print(company_name)
print("Initially, there were %s IRT tweets" % len(IRT_initial))
print("After outlier treatment, there are %s IRT tweets" % len(IRT_noOuts))
print("Initially, there were %s official tweets" % len(OT_initial))
print("After outlier treatment, there are %s official tweets" % len(OT_noOuts))
noOut_tweets = pd.concat([OT_noOuts, IRT_noOuts], axis=0)

#Function to standardize tweet performance metrics:
def standardize_metrics(df, metric1, metric2):
    z1 = stats.zscore(df[metric1])
    z2 = stats.zscore(df[metric2])
    df1 = df.copy()
    df1[metric1] = z1
    df1[metric2] = z2
    return df1

#Extract only 'en' and 'und' tweets:
English_tweets = noOut_tweets[noOut_tweets["Language"] == "en"]
und_tweets = noOut_tweets[noOut_tweets["Language"] == "und"]
all_eligible = pd.concat([English_tweets, und_tweets], axis=0)

#Standardize the remaining tweets (tweets ultimately utilized during 'IBM_Sentiment.py')
std_amzn = standardize_metrics(all_eligible, "Number of Likes", "Number of Retweets")
print("Standardized Amazon has %s tweets" % len(std_amzn))
rel_data = std_amzn.copy()
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
    
    #Remove tweets which were considered extreme outliers in 'IBM_Sentiment.py'
    IRT_initial = company_tweets2[company_tweets2["IRT"] == True].copy()
    OT_initial = company_tweets2[company_tweets2["OT"] == True].copy()
    IRT_noOuts = treatOuts(IRT_initial, "Number of Likes")
    OT_noOuts = treatOuts(OT_initial, "Number of Likes")
    print("\n")
    print(company_name)
    print("Initially, there were %s IRT tweets" % len(IRT_initial))
    print("After outlier treatment, there are %s IRT tweets" % len(IRT_noOuts))
    print("Initially, there were %s official tweets" % len(OT_initial))
    print("After outlier treatment, there are %s official tweets" % len(OT_noOuts))
    noOut_tweets = pd.concat([OT_noOuts, IRT_noOuts], axis=0)    
    
    #Extract 'en' and 'und' tweets:
    English_tweets = noOut_tweets[noOut_tweets["Language"] == "en"]
    und_tweets = noOut_tweets[noOut_tweets["Language"] == "und"]
    all_eligible = pd.concat([English_tweets, und_tweets], axis=0)
    
    #Standardize tweet performance metrics:
    std_comp = standardize_metrics(all_eligible, "Number of Likes", "Number of Retweets")
    print("Standardized %s has %s tweets" % (company_name, len(std_comp)))
    #std_data.append(std_comp)
    rel_data = pd.concat([rel_data, std_comp], axis=0)
    
#Save the resulting file:
rel_data.to_excel("rel_Data.xlsx")
print("All done")
                                    
