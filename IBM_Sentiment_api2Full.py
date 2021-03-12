# -*- coding: utf-8 -*-
#This file is being created to perform the 'IBM_Sentiment.py' process for the full, standardized dataset
#The code will be copied/pasted from 'IBM_Sentiment_api2.py', however some portions will be left out
#For example, tweets in this dataset are already labeled 'OT' or 'IRT', no need to do that again
#
#Overall, this is program: 35.3

import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras
import math
from scipy import stats

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, SentimentOptions, CategoriesOptions

##!!!Note: You will need to create a free IBM cloud account (lite version), create a 'Natural Language Understanding' resource, then..
##..generate your own API key and URL (copy/paste in place of mine), and finally change 'version' below to the date in which your account was created
#Credential info
version = '2021-03-09'
apiKey = 'QTAWA_sL2x3jBJVf6WbtdZ3IGpQyBPfj8F-ZfJxqVrLX'
URL = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/797ad017-f578-4f5a-aab4-8ec764455c5c'

#Creating the IBM instance
nlu = NaturalLanguageUnderstandingV1(
   version=version,
   iam_apikey=apiKey,
   url=URL)

print("It worked")
pd.set_option('display.max_colwidth', -1)

#Read in the data: Below line of code will need to be reconfigured for your filepath
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Python Files\\rel_Data.xlsx')

#Remove retweets from the company account, as they aren't technically company account tweets
patternDel = "^RT @"
filter1 = company_data["Content"].str.contains(patternDel)
company_tweets = company_data[~filter1].copy()
#print(company_tweets.shape)

#Function to handle extreme outliers
##Shouldn't need to use this since outliers were handled in the creation of datafile
def treatOuts(df, text_field):
    z = np.abs(stats.zscore(df[text_field]))
    df1 = df[(z < 4)]
    return df1

#Separate OT and IRT categories:
#IRT_initial = company_tweets2[company_tweets2["IRT"] == True].copy()
#OT_initial = company_tweets2[company_tweets2["OT"] == True].copy()
#IRT_noOuts = treatOuts(IRT_initial, "Number of Likes")
#OT_noOuts = treatOuts(OT_initial, "Number of Likes")
#print("Initially, there were %s IRT tweets" % len(IRT_initial))
#print("After outlier treatment, there are %s IRT tweets" % len(IRT_noOuts))
#print("Initially, there were %s official tweets" % len(OT_initial))
#print("After outlier treatment, there are %s official tweets" % len(OT_noOuts))
#company_tweets = pd.concat([OT_noOuts, IRT_noOuts], axis=0)
#print(company_tweets.shape)

##Replace emojis with representative text:
import emoji
#Function to replace emojis within an individual tweet
def conv_emojis(tweet):
    return emoji.demojize(str(tweet))

#Function which calls 'conv_emojis' function for all tweets
def replace_emojis(dframe, text):
    for i in range(len(dframe[text])):
        dframe.iat[i, 2] = conv_emojis(str(dframe.iloc[i, dframe.columns.get_loc(text)]))
    return dframe

company_tweets = replace_emojis(company_tweets, "Content")

#Remove/replace 'smart' apostrophes and quotation marks with standard keyboard equivalents:
company_tweets["Content"] = company_tweets["Content"].str.replace(r"’", "'") #replace closing smart apostrophes with regular apostrophe
company_tweets["Content"] = company_tweets["Content"].str.replace(r"‘", "'") #replace opening smart apostrophes with regular apostrophe
company_tweets["Content"] = company_tweets["Content"].str.replace(r"“", "\"") #replace opening smart quotes with regular quotes
company_tweets["Content"] = company_tweets["Content"].str.replace(r"”", "\"") #replace closing smart quotes with regular quotes

##Replace abbreviations with full representation
appos = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not"
}

#Need to convert to lowercase to ensure matches with above dictionary
company_tweets["Content"] = company_tweets["Content"].str.lower()

#Function to replace abbreviations in single tweet:
def conv_abbr(tweet):
    words = tweet.split()
    reworded = [appos[word] if word in appos else word for word in words]
    reworded = " ".join(reworded)
    return reworded

#Function which calls function above to replace abbreviations in all tweets:
def replace_abbr(df, text_field):
    for i in range(len(df[text_field])):
        df.iat[i, 2] = conv_abbr(str(df.iloc[i, df.columns.get_loc(text_field)]))
    return df

#Replace all abbreviations with full form:
company_tweets = replace_abbr(company_tweets, "Content")

#Perform standardization on the textual contents of the company's tweets:
#Removing underscore from set of acceptable chars since emojis are represented with underscores
#Changing set of acceptable characters to just be the set of English letters
def standardize_text_forSentiment(df, text_field):
    #Don't remove/replace periods for sentiment analysis, don't want to convert acronyms to real words with sentiment scores
    #df[text_field] = df[text_field].str.replace(r".", "") #remove/replace periods w/ nothing. Should now count acronyms as one word
    df[text_field] = df[text_field].str.replace(r"&", "and") #replace ampersands with 'and'
    df[text_field] = df[text_field].str.replace(r"http\S+", "") #remove links and replace w/ nothing
    df[text_field] = df[text_field].str.replace(r"http", "") #ensure all links have been removed
    df[text_field] = df[text_field].str.replace(r"@\S+", "") #remove @username mentions and replace with nothing
    df[text_field] = df[text_field].str.replace(r"%", " percent") #replace '%' with ' percent'
    df[text_field] = df[text_field].str.replace(r"#[A-Za-z0-9]+", "") #remove/replace hashtags with nothing for sentiment analysis
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z]", " ") #Remove/replace any char that's not a letter with a space
    df[text_field] = df[text_field].str.replace(r"@", "at") #replace any remaining '@' symbols with 'at'
    df[text_field] = df[text_field].str.lower() #convert all remaining text to lowercase
    df[text_field] = df[text_field].str.replace(r"\s+", " ") #replace 2+ spaces with single space
    df[text_field] = df[text_field].str.replace(r",(?=\d{3})", "")#remove commas followed by 3 numbers and replace w/ nothing    
    return df

textual_tweets = standardize_text_forSentiment(company_tweets, "Content")

#This dataset only consists of tweets marked 'en' or 'und', no need to extract those out here

#Removing rows with no text left inside them
filter1 = textual_tweets["Content"] != ""
cleanGlish_tweets = textual_tweets[filter1]
#print("cleanGlish just created")
#print(cleanGlish_tweets.shape)

#Truly make sure that only textual contents remain:
cleanGlish_tweets["Content"] = cleanGlish_tweets["Content"].str.replace(r"\s+", " ")
filter2 = cleanGlish_tweets["Content"] != ""
cleanGlish_tweets = cleanGlish_tweets[filter2]
print("After removing non-text/empty rows, there are this many tweets:")
print(cleanGlish_tweets.shape)
filter3 = cleanGlish_tweets["Content"] != " "
cleanGlish_tweets = cleanGlish_tweets[filter3]
print("After removing rows consisting of one space, there are:")
print(cleanGlish_tweets.shape)
print(cleanGlish_tweets["Content"].head(5))

#Record run-time for sentiment portion:
import time

sentiments = []
eye = 0
start_time = time.time()
for tweet in cleanGlish_tweets["Content"]:
    if tweet != "":
        response = nlu.analyze(text=tweet, features=Features(sentiment=SentimentOptions()), language='en').get_result()
        score = response.get('sentiment').get('document').get('score')
        sentiments.append(score)
    else:
        print("Encountered an empty tweet")
        sentiments.append(1000)
        print(eye)
    #print(eye)
    eye = eye + 1

end_time = time.time()
run_time = end_time - start_time
print("Sentiment run time: %s" % run_time)
cleanGlish_tweets["Sentiments"] = sentiments
print(cleanGlish_tweets["Sentiments"].head(5))

#Establish baseline:
print("Total number of tweets: %s" % len(cleanGlish_tweets))
print("Across all tweets, the average sentiment is: %s" % np.mean(cleanGlish_tweets["Sentiments"]))
print("Across all tweets, average number of likes is: %s" % np.mean(cleanGlish_tweets["Number of Likes"]))
print("Across all tweets, average number of RTs is: %s" % np.mean(cleanGlish_tweets["Number of Retweets"]))

#Examine differences in negative, neutral, and positive categories:
#Examine sentiment distribution if scores < 0 are negative, scores = 0 are neutral, and scores > 0 are positive:
neg_tweets = cleanGlish_tweets[cleanGlish_tweets["Sentiments"] < 0].copy()
neutral_tweets = cleanGlish_tweets[cleanGlish_tweets["Sentiments"] == 0].copy()
pos_tweets = cleanGlish_tweets[cleanGlish_tweets["Sentiments"] > 0].copy()
#print("Negative Tweets:")
#print(neg_tweets.shape)
print("\n")
print("Total number of negative tweets: %s" % len(neg_tweets))
#print(neg_tweets["Content"].head(5))
#print(neg_tweets["Sentiments"].head(5))
print("Average negative sentiment: %s" % np.mean(neg_tweets["Sentiments"]))
print("Negative tweets average %s likes" % np.mean(neg_tweets["Number of Likes"]))
print("Negative tweets average %s RTs:" % np.mean(neg_tweets["Number of Retweets"]))

#print("Neutral Tweets:")
#print(neutral_tweets.shape)
print("\n")
print("Total number of neutral tweets: %s" % len(neutral_tweets))
#print(neutral_tweets["Content"].head(5))
#print(neutral_tweets["Sentiments"].head(5))
print("Average neutral sentiment: %s" % np.mean(neutral_tweets["Sentiments"]))
print("Neutral tweets average %s likes" % np.mean(neutral_tweets["Number of Likes"]))
print("Neutral tweets average %s RTs:" % np.mean(neutral_tweets["Number of Retweets"]))

#print("Positive Tweets:")
#print(pos_tweets.shape)
print("\n")
print("Total number of positive tweets: %s" % len(pos_tweets))
#print(pos_tweets["Content"].head(5))
#print(pos_tweets["Sentiments"].head(5))
print("Average positive sentiment: %s" % np.mean(pos_tweets["Sentiments"]))
print("Positive tweets average %s likes" % np.mean(pos_tweets["Number of Likes"]))
print("Positive tweets average %s RTs:" % np.mean(pos_tweets["Number of Retweets"]))

###Perform split analysis:
official_tweets = cleanGlish_tweets[cleanGlish_tweets["OT"] == True].copy()
IRT_tweets = cleanGlish_tweets[cleanGlish_tweets["IRT"] == True].copy()

#Examine proportion of official vs. IRT tweets:
numAll = len(cleanGlish_tweets["Content"])
numOT = len(official_tweets["Content"])
numIRT = len(IRT_tweets["Content"])

propOT = (numOT/numAll)
propIRT = (numIRT/numAll)
print("Proportion of official tweets: %f" % propOT)
print("Proportion of IRT tweets: %f" % propIRT)

#Set the minimum threshold for proportion of tweets in either category:
#Changing minThresh to 0 since we'll definitely have ample tweets in both categories here
minThresh = 0

if propOT >= minThresh and propIRT >= minThresh:
    print("There are enough tweets for full analysis")
    
    #Official Tweet Analysis:
    print("\n")
    print("Total number of official tweets: %s" % len(official_tweets["Content"]))
    print("Average sentiment of official tweets: %s" % np.mean(official_tweets["Sentiments"]))
    print("Average likes across all official tweets: %s" %np.mean(official_tweets["Number of Likes"]))
    print("Average RTs across all official tweets: %s" %np.mean(official_tweets["Number of Retweets"]))
    
    neg_OT = official_tweets[official_tweets["Sentiments"] < 0].copy()
    neutral_OT = official_tweets[official_tweets["Sentiments"] == 0].copy()
    pos_OT = official_tweets[official_tweets["Sentiments"] > 0].copy()
    
    print("Total number of negative official tweets: %s" % len(neg_OT["Content"]))
    print("Average sentiment of negative official tweets: %s" % np.mean(neg_OT["Sentiments"]))
    print("Average likes across negative official tweets: %s" % np.mean(neg_OT["Number of Likes"]))
    print("Average RTs across negative official tweets: %s" % np.mean(neg_OT["Number of Retweets"]))
    
    print("Total number of neutral official tweets: %s" % len(neutral_OT["Content"]))
    print("Average sentiment of neutral official tweets: %s" % np.mean(neutral_OT["Sentiments"]))
    print("Average likes across neutral official tweets: %s" % np.mean(neutral_OT["Number of Likes"]))
    print("Average RTs across neutral official tweets: %s" % np.mean(neutral_OT["Number of Retweets"]))
    
    print("Total number of positive official tweets: %s" % len(pos_OT["Content"]))
    print("Average sentiment of positive official tweets: %s" % np.mean(pos_OT["Sentiments"]))
    print("Average likes across positive official tweets: %s" % np.mean(pos_OT["Number of Likes"]))
    print("Average RTs across positive official tweets: %s" % np.mean(pos_OT["Number of Retweets"]))
    
    #Analyze sentiments across top, median, and bottom categories of official tweets:
    #Specify number of tweets to take as top 25%
    nr = math.ceil(0.25 * len(official_tweets))
    
    #Sort the data based on likes:
    official_tweets = official_tweets.sort_values(by = "Number of Likes", ascending=False)
    
    #Extract the top 25% performing official tweets:
    top_official = official_tweets.head(nr).copy()
    
    print("Total number of top official tweets: %s" % len(top_official["Content"]))
    print("Average sentiment of top official tweets: %s" % np.mean(top_official["Sentiments"]))
    print("Average likes across top official tweets: %s" % np.mean(top_official["Number of Likes"]))
    print("Average RTs across top official tweets: %s" % np.mean(top_official["Number of Retweets"]))
    
    #Extract median 50% performing official tweets:
    desired_num = int(round((0.5 * len(official_tweets))))
    nr2 = nr + 1
    endPoint = int(nr + desired_num)
    if len(official_tweets) % 4 == 0:
        endPoint = endPoint + 1
    mid_official = official_tweets.iloc[nr2:endPoint]
    
    print("Total number of median official tweets: %s" % len(mid_official["Content"]))
    print("Average sentiment of median official tweets: %s" % np.mean(mid_official["Sentiments"]))
    print("Average likes across median official tweets: %s" % np.mean(mid_official["Number of Likes"]))
    print("Average RTs across median official tweets: %s" % np.mean(mid_official["Number of Retweets"]))
    
    #Extract bottom 25% performing official tweets:
    bottom_official = official_tweets.tail(nr).copy()
    
    print("Total number of bottom official tweets: %s" % len(bottom_official["Content"]))
    print("Average sentiment of bottom official tweets: %s" % np.mean(bottom_official["Sentiments"]))
    print("Average likes across bottom official tweets: %s" % np.mean(bottom_official["Number of Likes"]))
    print("Average RTs across bottom official tweets: %s" % np.mean(bottom_official["Number of Retweets"]))
    
    #########################Perform same analysis over IRT tweets:
    #IRT Tweet Analysis:
    print("\n")
    print("Total number of IRT tweets: %s" % len(IRT_tweets["Content"]))
    print("Average sentiment of IRT tweets: %s" % np.mean(IRT_tweets["Sentiments"]))
    print("Average likes across all IRT tweets: %s" %np.mean(IRT_tweets["Number of Likes"]))
    print("Average RTs across all IRT tweets: %s" %np.mean(IRT_tweets["Number of Retweets"]))
    
    neg_IRT = IRT_tweets[IRT_tweets["Sentiments"] < 0].copy()
    neutral_IRT = IRT_tweets[IRT_tweets["Sentiments"] == 0].copy()
    pos_IRT = IRT_tweets[IRT_tweets["Sentiments"] > 0].copy()
    
    print("Total number of negative IRT tweets: %s" % len(neg_IRT["Content"]))
    print("Average sentiment of negative IRT tweets: %s" % np.mean(neg_IRT["Sentiments"]))
    print("Average likes across negative IRT tweets: %s" % np.mean(neg_IRT["Number of Likes"]))
    print("Average RTs across negative IRT tweets: %s" % np.mean(neg_IRT["Number of Retweets"]))
    
    print("Total number of neutral IRT tweets: %s" % len(neutral_IRT["Content"]))
    print("Average sentiment of neutral IRT tweets: %s" % np.mean(neutral_IRT["Sentiments"]))
    print("Average likes across neutral IRT tweets: %s" % np.mean(neutral_IRT["Number of Likes"]))
    print("Average RTs across neutral IRT tweets: %s" % np.mean(neutral_IRT["Number of Retweets"]))
    
    print("Total number of positive IRT tweets: %s" % len(pos_IRT["Content"]))
    print("Average sentiment of positive IRT tweets: %s" % np.mean(pos_IRT["Sentiments"]))
    print("Average likes across positive IRT tweets: %s" % np.mean(pos_IRT["Number of Likes"]))
    print("Average RTs across positive IRT tweets: %s" % np.mean(pos_IRT["Number of Retweets"]))
    
    #Analyze sentiments across top, median, and bottom categories of IRT tweets:
    #Specify number of tweets to take as top 25%
    nr = math.ceil(0.25 * len(IRT_tweets))
    
    #Sort the data based on likes:
    IRT_tweets = IRT_tweets.sort_values(by = "Number of Likes", ascending=False)
    
    #Extract the top 25% performing IRT tweets:
    top_IRT = IRT_tweets.head(nr).copy()
    
    print("Total number of top IRT tweets: %s" % len(top_IRT["Content"]))
    print("Average sentiment of top IRT tweets: %s" % np.mean(top_IRT["Sentiments"]))
    print("Average likes across top IRT tweets: %s" % np.mean(top_IRT["Number of Likes"]))
    print("Average RTs across top IRT tweets: %s" % np.mean(top_IRT["Number of Retweets"]))
    
    #Extract median 50% performing IRT tweets:
    desired_num = int(round((0.5 * len(IRT_tweets))))
    nr2 = nr + 1
    endPoint = int(nr + desired_num)
    if len(IRT_tweets) % 4 == 0:
        endPoint = endPoint + 1
    mid_IRT = IRT_tweets.iloc[nr2:endPoint]
    
    print("Total number of median IRT tweets: %s" % len(mid_IRT["Content"]))
    print("Average sentiment of median IRT tweets: %s" % np.mean(mid_IRT["Sentiments"]))
    print("Average likes across median IRT tweets: %s" % np.mean(mid_IRT["Number of Likes"]))
    print("Average RTs across median IRT tweets: %s" % np.mean(mid_IRT["Number of Retweets"]))
    
    #Extract bottom 25% performing IRT tweets:
    bottom_IRT = IRT_tweets.tail(nr).copy()
    
    print("Total number of bottom IRT tweets: %s" % len(bottom_IRT["Content"]))
    print("Average sentiment of bottom IRT tweets: %s" % np.mean(bottom_IRT["Sentiments"]))
    print("Average likes across bottom IRT tweets: %s" % np.mean(bottom_IRT["Number of Likes"]))
    print("Average RTs across bottom IRT tweets: %s" % np.mean(bottom_IRT["Number of Retweets"]))

cleanGlish_tweets.to_excel("sent_fullResults.xlsx")
print("All done")