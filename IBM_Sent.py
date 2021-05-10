# -*- coding: utf-8 -*-
#This file is being created to perform sentiment analysis again (forgot to save all results) using 'IBM-Watson analyzer'
#Code developed from: https://heartbeat.fritz.ai/evaluating-unsupervised-sentiment-analysis-tools-using-labeled-data-8d4bb1336cee
#I'll be mostly creating this file by copying/pasting code from 'IBM_Sentiment.py'
#
#Overall, this is program: 60


import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras
import math
from scipy import stats
import random

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, SentimentOptions, CategoriesOptions

#Don't think it's needed, but never hurts to be seeded
random.seed(1)
np.random.seed(2)

##!!!Note: You will need to create a free IBM cloud account (lite version), create a 'Natural Language Understanding' resource, then..
##..generate your own API key and URL (copy/paste in place of mine), and finally change 'version' below to the date in which your account was created
#Credential info
version = '2021-03-06'
apiKey = 'aJ3KmWyOUiN8T9PzNtsThX83yZQ5Xa0WcNSqapophMJA'
URL = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/32787a42-fd35-4b51-9b2c-f91eef6a7ee9'

#Creating the IBM instance
nlu = NaturalLanguageUnderstandingV1(
   version=version,
   iam_apikey=apiKey,
   url=URL)

print("It worked")

#Variables below will need to be reconfigured to match your filepath as well
path = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Pictures\\SentimentResults\\'
commonEnd = '_sentResults.xlsx'

#Read in the data: Below line of code will need to be reconfigured for your filepath
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Amazon_Dec_1_2020.xlsx')
company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]
print(company_name)

#Remove retweets from the company account, as they aren't technically company account tweets
patternDel = "^RT @"
filter1 = company_data["Content"].str.contains(patternDel)
company_tweets = company_data[~filter1].copy()


#Examine first 5 tweets before any alterations are made
pd.set_option('display.max_colwidth', -1)
#print(company_tweets["Content"].head(5))

##Replace emojis with representative text:
import emoji
#Function to replace emojis within an individual tweet
def conv_emojis(tweet):
    return emoji.demojize(str(tweet))

#Function which calls 'conv_emojis' function for all tweets
def replace_emojis(dframe, text):
    for i in range(len(dframe[text])):
        dframe.iat[i, 1] = conv_emojis(str(dframe.iloc[i, dframe.columns.get_loc(text)]))
    return dframe

company_tweets = replace_emojis(company_tweets, "Content")
#print(company_tweets["Content"].head(5))


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
        df.iat[i, 1] = conv_abbr(str(df.iloc[i, df.columns.get_loc(text_field)]))
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


#Removing tweets that weren't originally in English, or UND (consisting of combo of emojis, links, mentions)
English_tweets = textual_tweets[textual_tweets["Language"] == "en"]
#print(English_tweets.shape)
und_tweets = textual_tweets[textual_tweets["Language"] == "und"]
#print(und_tweets.shape)
all_tweets = pd.concat([English_tweets, und_tweets], axis=0)
#print(all_tweets.shape)

#Removing rows with no text left inside them
filter1 = all_tweets["Content"] != ""
cleanGlish_tweets = all_tweets[filter1]
#print(cleanGlish_tweets.shape)

###Spellchecking code is ultimately not used, as it's slow and brand company twitters seem to proof-read tweets beforehand
from spellchecker import SpellChecker

spell = SpellChecker(language='en')

#Function to correct spelling mistakes within a single tweet
def fix_spelling(tweet):
    words = tweet.split()
    fixed_words = [spell.correction(word) for word in words]
    fixed_words = " ".join(fixed_words)
    return fixed_words

#Function which calls function to correct spelling mistakes for all tweets
def replace_spelling(df, text_field):
    for i in range(len(df[text_field])):
        df.iat[i, 1] = fix_spelling(df.iloc[i, df.columns.get_loc(text_field)])
    return df

#Correct spelling mistakes found in any tweet: !!!Commenting out
#cleanGlish_tweets = replace_spelling(cleanGlish_tweets, "Content")
##Seems that commenting this out causes punctuation to remain, but I don't think that's necessarily a problem

#Examine tweets after standardization has been performed:
#print(cleanGlish_tweets["Content"].head(5))

##What I have above for spelling correction works, it just takes a decent amount of time and doesn't really seem too useful
##Since these tweets originate from official brand company twitter accounts, they don't seem to contain spelling errors


###Perform sentiment analysis:

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
cleanGlish_tweets["Sentiment"] = sentiments
print(cleanGlish_tweets["Sentiment"].head(5))

#Save results:
fpath = path + str(company_name) + commonEnd
cleanGlish_tweets.to_excel(fpath)

