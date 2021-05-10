#This file's purpose is to remove extreme outliers from data:
#Outliers will be calculated separately for likes and retweets, and also within OT vs. IRT
#
#Overall, this is program: 52


import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras
import math
from scipy import stats


#Begin by reading in data: Below line of code will need to be reconfigured for your filepath
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Toyota_Dec_1_2020.xlsx')
company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]

#Remove retweets from the company account, as they aren't technically company account tweets
patternDel = "^RT @"
filter1 = company_data["Content"].str.contains(patternDel)
company_tweets2 = company_data[~filter1].copy()
print("After removing retweets, %s has %s tweets" % (company_name, len(company_tweets2)))

#########################################################################
#Before doing anything else, designate tweets as 'OT' or 'IRT':
#Perform initial separation based on "^@" regex:
initIRT = [bool(re.search("^@", i)) for i in company_tweets2["Content"]]
initOT = [not elem for elem in initIRT]
#print(initOT)

#Create IRT and OT variables in the data:
company_tweets2["IRT"] = initIRT
company_tweets2["OT"] = initOT

#print(company_tweets["IRT"])
#print(company_tweets["OT"])

#Fill in NAs under the 'In Reply To' field with "OT":
company_tweets2["In Reply To"] = company_tweets2["In Reply To"].replace(np.nan, "OT", regex=True)
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

company_tweets2 = cleanSplit(company_tweets2, "Content", "IRT", "In Reply To", "Author", "OT")

#Create official and IRT datasets:
#official_tweets = company_tweets2[company_tweets2["OT"] == True].copy()
#IRT_tweets = company_tweets2[company_tweets2["IRT"] == True].copy()

#Creating function to remove outliers from data:
#text1 = 'Number of Likes', text2 = 'Number of Retweets'
def remove_outliers(df, text1, text2):
    
    t3 = 'stdLikes'
    t4 = 'stdRT'
    z1 = stats.zscore(df[text1]) #create z-scores for Likes
    z2 = stats.zscore(df[text2]) #create z-scores for Retweets
    df[t3] = z1 #create variable representing standardized likes
    df[t4] = z2 #create variable representing standardized retweets
    df1 = df[df[t3] < 5].copy() #keep tweets receiving less than 5 standard deviations above mean likes
    df2 = df1[df1[t4] < 5].copy() #Of those, keep tweets receiving less than 5 standard deviations above mean retweets
    return df2


#Remove extreme outliers from official tweets:
#official_tweets2 = remove_outliers(official_tweets, "Number of Likes", "Number of Retweets")

#Remove outliers from IRT tweets:
#IRT_tweets2 = remove_outliers(IRT_tweets, "Number of Likes", "Number of Retweets")



##########################################################################

#Remove tweets not meant for the English audience, as those markets may behave differently
English_tweets = company_tweets2[company_tweets2["Language"] == 'en'].copy()
und_tweets = company_tweets2[company_tweets2["Language"] == 'und'].copy()
company_tweets3 = pd.concat([English_tweets, und_tweets], axis=0)
print("For %s, we're beginning with %s English audience tweets" % (company_name, len(company_tweets3)))


#Create official and IRT datasets:
official_tweets = company_tweets3[company_tweets3["OT"] == True].copy()
IRT_tweets = company_tweets3[company_tweets3["IRT"] == True].copy()

#Remove extreme outliers from English audience official tweets:
official_tweets2 = remove_outliers(official_tweets, "Number of Likes", "Number of Retweets")

#Remove outliers from English audience IRT tweets:
IRT_tweets2 = remove_outliers(IRT_tweets, "Number of Likes", "Number of Retweets")


#Create dataset consisting of tweets meant for the English audience, free of extreme outliers (likes or RTs 5+ stdevs above mean, within category)
company_tweets = pd.concat([official_tweets2, IRT_tweets2], axis=0)
print("After removing all outliers, there are %s tweets remaining" % len(company_tweets))

figpath = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\OutlierFree_data\\'
ending = '_outsRemoved.xlsx'
figpath2 = figpath + str(company_name) + ending
company_tweets.to_excel(figpath2)