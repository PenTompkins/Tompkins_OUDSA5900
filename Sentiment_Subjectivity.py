# -*- coding: utf-8 -*-
#This file is being created to perform sentiment/subjectivity analysis separate from topic modeling
#I'm beginning by copying/pasting the sentiment portion of code from the beginning of 'Topic_Sentiment.py'
#The sentiment/subjectivity code was derived from: https://towardsdatascience.com/text-analysis-basics-in-python-443282942ec5
#!Note: This file has been retired in favor of 'IBM_Sentiment.py', no real analysis is performed here anymore
#
#Overall, this is program: 34

import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras



#Read in the data
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Amazon_Dec_1_2020.xlsx')
company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]

#Remove retweets from the company account, as they aren't technically company account tweets
patternDel = "^RT @"
filter1 = company_data["Content"].str.contains(patternDel)
company_tweets = company_data[~filter1].copy()
company_tweets2 = company_data[~filter1].copy()
#print(company_tweets.shape)

#Examine first 5 tweets before any alterations are made
pd.set_option('display.max_colwidth', -1)
print(company_tweets["Content"].head(5))

#Remove/replace 'smart' apostrophes and quotation marks with standard keyboard equivalents:
company_tweets["Content"] = company_tweets["Content"].str.replace(r"’", "'") #replace closing smart apostrophes with regular apostrophe
company_tweets["Content"] = company_tweets["Content"].str.replace(r"‘", "'") #replace opening smart apostrophes with regular apostrophe
company_tweets["Content"] = company_tweets["Content"].str.replace(r"“", "\"") #replace opening smart quotes with regular quotes
company_tweets["Content"] = company_tweets["Content"].str.replace(r"”", "\"") #replace closing smart quotes with regular quotes

#Examine tweets after removing/replacing 'smart' apostrophes and quotes:
print(company_tweets["Content"].head(5))

##!!For sentiment analysis, let's keep apostrophes in
#Remove apostrophes followed by 's' and replace with nothing (Disney's becomes Disney):
#company_tweets["Content"] = company_tweets["Content"].str.replace(r"'s", "")

#Remove remaining apostrophes and replace with nothing (convert I'm to Im and such):
#company_tweets["Content"] = company_tweets["Content"].str.replace(r"'", "")

#Perform standardization on the textual contents of the company's tweets:
#Slightly different than before, commenting out line to remove/replace periods with nothing, adding periods to the set of acceptable chars to keep
#Also replacing double spaces with single spaces, replacing '%' with ' percent', replacing newline chars with spaces
def standardize_text_forSentiment(df, text_field):
    #Don't remove/replace periods for sentiment analysis, don't want to convert acronyms to real words with sentiment scores
    #df[text_field] = df[text_field].str.replace(r".", "") #remove/replace periods w/ nothing. Should now count acronyms as one word
    df[text_field] = df[text_field].str.replace(r"&", "and") #replace ampersands with 'and'
    df[text_field] = df[text_field].str.replace(r"http\S+", "") #remove links and replace w/ nothing
    df[text_field] = df[text_field].str.replace(r"http", "") #ensure all links have been removed
    df[text_field] = df[text_field].str.replace(r"@\S+", "") #remove @username mentions and replace with nothing
    #Replace '%' with ' percent'
    df[text_field] = df[text_field].str.replace(r"%", " percent")
    #add periods to the set of acceptable chars below, remove newline character
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\.]", " ")#Remove/replace anything that's not capital/lowercase letter, number, parentheses, comma, or any of the following symbols with a space
    df[text_field] = df[text_field].str.replace(r"@", "at") #replace any remaining '@' symbols with 'at'
    df[text_field] = df[text_field].str.lower() #convert all remaining text to lowercase
    #Replace double spaces created above with single spaces
    df[text_field] = df[text_field].str.replace(r"\s+", " ") #replace 2+ spaces with single space
    df[text_field] = df[text_field].str.replace(r",(?=\d{3})", "")#remove commas followed by 3 numbers and replace w/ nothing    
    return df

textual_tweets = standardize_text_forSentiment(company_tweets, "Content")

#Examine tweets after standardization has been performed:
print(textual_tweets["Content"].head(5))

#Removing tweets that weren't originally in English
English_tweets = textual_tweets[textual_tweets["Language"] == "en"]

#Removing rows with no text left inside them
filter1 = English_tweets["Content"] != ""
cleanGlish_tweets = English_tweets[filter1]

##Perform first attempt at sentiment analysis:
from textblob import TextBlob

#Create polarity and subjectivity scores:
cleanGlish_tweets["Polarity"] = cleanGlish_tweets["Content"].apply(lambda x:TextBlob(x).polarity)
cleanGlish_tweets["Subjective"] = cleanGlish_tweets["Content"].apply(lambda x:TextBlob(x).subjectivity)

#Examine results:
print("Tweets and polarity, then subjectivity scores:")
print(cleanGlish_tweets["Content"].head(5))
print(cleanGlish_tweets["Polarity"].head(5))
print(cleanGlish_tweets["Subjective"].head(5))

print("Average polarity across all %s company tweets: %s" % (company_name, np.mean(cleanGlish_tweets["Polarity"])))
print("Average subjectivity across all %s company tweets: %s" % (company_name, np.mean(cleanGlish_tweets["Subjective"])))

#If I define 'negative' to be polarity scores between -1 and -1/3, 'neutral' to be between -1/3 and 1/3, then 'positive' to be between 1/3 and 1
#I wonder how many tweets will fall into each category?
print("Before splitting, there are %s tweets" % len(cleanGlish_tweets["Content"]))

neg_tweets = cleanGlish_tweets[cleanGlish_tweets["Polarity"] < -1/3]
#neutral_tweets = cleanGlish_tweets[(cleanGlish_tweets["Polarity"] >= -1/3) and (cleanGlish_tweets["Polarity"] <= 1/3)]
neutral_tweets = cleanGlish_tweets[cleanGlish_tweets["Polarity"].between(-1/3, 1/3)] #.between is inclusive by default
pos_tweets = cleanGlish_tweets[cleanGlish_tweets["Polarity"] > 1/3]

print("There are %s negative tweets, with an average polarity of %s" % (len(neg_tweets["Content"]), np.mean(neg_tweets["Polarity"])))
print("There are %s neutral tweets, with an average polarity of %s" % (len(neutral_tweets["Content"]), np.mean(neutral_tweets["Polarity"])))
print("There are %s positive tweets, with an average polarity of %s" % (len(pos_tweets["Content"]), np.mean(pos_tweets["Polarity"])))

print("Negative tweets: Average likes = %s, Average RT = %s" % (np.mean(neg_tweets["Number of Likes"]), np.mean(neg_tweets["Number of Retweets"])))
print("Neutral tweets: Average likes = %s, Average RT = %s" % (np.mean(neutral_tweets["Number of Likes"]), np.mean(neutral_tweets["Number of Retweets"])))
print("Positive tweets: Average likes = %s, Average RT = %s" % (np.mean(pos_tweets["Number of Likes"]), np.mean(pos_tweets["Number of Retweets"])))

#I wonder what this is considering negative tweets to be:
#print(neg_tweets["Content"].head(5)) #honestly, 4/5 of these tweets seemed pretty negative (apologetic, really)


##Define tweets with subjectivity scores <= 0.5 to be objective, tweets w/ subjectivity scores > 0.5 to be subjective
leans_obj = cleanGlish_tweets[cleanGlish_tweets["Subjective"] <= 0.5]
leans_subj = cleanGlish_tweets[cleanGlish_tweets["Subjective"] > 0.5]

#Examine subjectivity:
print("%s objective tweets. Average likes = %s, average RT = %s" % (len(leans_obj["Content"]), np.mean(leans_obj["Number of Likes"]), np.mean(leans_obj["Number of Retweets"])))
print("%s subjective tweets. Average likes = %s, average RT = %s" % (len(leans_subj["Content"]), np.mean(leans_subj["Number of Likes"]), np.mean(leans_subj["Number of Retweets"])))


##Determine whether results remain the same, generally, or change after splitting into OT vs. IRT tweet categories:
#reset the data
company_tweets = company_tweets2.copy()

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

company_tweets = cleanSplit(company_tweets, "Content", "IRT", "In Reply To", "Author", "OT")
#Create official and IRT datasets:
official_tweets = company_tweets[company_tweets["OT"] == True].copy()
IRT_tweets = company_tweets[company_tweets["IRT"] == True].copy()
print(official_tweets.shape)
print(IRT_tweets.shape)
                               
#Examine proportion of official vs. IRT tweets:
numAll = len(company_tweets["Content"])
numOT = len(official_tweets["Content"])
numIRT = len(IRT_tweets["Content"])

propOT = (numOT/numAll)
propIRT = (numIRT/numAll)
print(company_name)
print("Proportion of official tweets: %f" % propOT)
print("Proportion of IRT tweets: %f" % propIRT)

#Of the 6 datasets not containing enough of one or the other, the highest proportion is propOT = 0.0416 for Google data
#Coca Cola, Disney, McDonalds, Samsung, and Toyota each have less than 0.01 for either propOT or propIRT
#Thus, I believe it's fair to consider all tweets being part of one group in the case that propOT or propIRT is less than 0.05
minThresh = 0.05

###If there are enough tweets in both categories, analyze sentiment/subjectivity separately:
if propOT >= minThresh and propIRT >= minThresh:
    print("%s has enough tweets in both categories for further analysis" % company_name)
    
    #Remove/replace 'smart' apostrophes and quotation marks with standard keyboard equivalents:
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"’", "'") #replace closing smart apostrophes with regular apostrophe
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"‘", "'") #replace opening smart apostrophes with regular apostrophe
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"“", "\"") #replace opening smart quotes with regular quotes
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"”", "\"") #replace closing smart quotes with regular quotes
    
    #Examine tweets after removing/replacing 'smart' apostrophes and quotes:
    #print(official_tweets["Content"].head(5))
    
    ##Not removing apostrophes for sentiment analysis
    #Remove apostrophes followed by 's' and replace with nothing (Disney's becomes Disney):
    #official_tweets["Content"] = official_tweets["Content"].str.replace(r"'s", "")
    
    #Remove remaining apostrophes and replace with nothing (convert I'm to Im and such):
    #official_tweets["Content"] = official_tweets["Content"].str.replace(r"'", "")    
    
    #Official tweets: Remove links, @username mentions, non alphanumeric characters, replace @ with at, convert all text to lowercase
    official_tweets2 = standardize_text_forSentiment(official_tweets, "Content")
    
    #Removing tweets that weren't originally in English
    English_official = official_tweets2[official_tweets2["Language"] == "en"]
    
    #Removing rows with no text left inside them
    filter3 = English_official["Content"] != ""
    official_clean = English_official[filter3]
    
    #Create polarity and subjectivity scores for official tweets only:
    official_clean["Polarity"] = official_clean["Content"].apply(lambda x:TextBlob(x).polarity)
    official_clean["Subjective"] = official_clean["Content"].apply(lambda x:TextBlob(x).subjectivity)
    
    
    #Examine results:
    print("\n")
    print("Official Tweets and polarity, then subjectivity scores:")
    print(official_clean["Content"].head(5))
    print(official_clean["Polarity"].head(5))
    print(official_clean["Subjective"].head(5))
    
    print("Average polarity across all %s official tweets: %s" % (company_name, np.mean(official_clean["Polarity"])))
    print("Average subjectivity across all %s official tweets: %s" % (company_name, np.mean(official_clean["Subjective"])))
    
    print("Before splitting, there are %s official tweets" % len(official_clean["Content"]))
    
    neg_official = official_clean[official_clean["Polarity"] < -1/3]
    neutral_official = official_clean[official_clean["Polarity"].between(-1/3, 1/3)]
    pos_official = official_clean[official_clean["Polarity"] > 1/3]
    
    print("There are %s negative OT tweets, with an average polarity of %s" % (len(neg_official["Content"]), np.mean(neg_official["Polarity"])))
    print("There are %s neutral OT tweets, with an average polarity of %s" % (len(neutral_official["Content"]), np.mean(neutral_official["Polarity"])))
    print("There are %s positive OT tweets, with an average polarity of %s" % (len(pos_official["Content"]), np.mean(pos_official["Polarity"])))
    
    print("Negative OT tweets: Average likes = %s, Average RT = %s" % (np.mean(neg_official["Number of Likes"]), np.mean(neg_official["Number of Retweets"])))
    print("Neutral OT tweets: Average likes = %s, Average RT = %s" % (np.mean(neutral_official["Number of Likes"]), np.mean(neutral_official["Number of Retweets"])))
    print("Positive OT tweets: Average likes = %s, Average RT = %s" % (np.mean(pos_official["Number of Likes"]), np.mean(pos_official["Number of Retweets"])))
    
    ##I'm honestly just curious what the 3 negative official Amazon tweets look like:
    print("Amazon's three negative official tweets:")
    print(neg_official["Content"].head(3)) #at least two of these three don't really seem all too negative
    
    ##Define tweets with subjectivity scores <= 0.5 to be objective, tweets w/ subjectivity scores > 0.5 to be subjective
    leans_objOT = official_clean[official_clean["Subjective"] <= 0.5]
    leans_subjOT = official_clean[official_clean["Subjective"] > 0.5]
    
    #Examine subjectivity:
    print("%s OT objective tweets. Average likes = %s, average RT = %s" % (len(leans_objOT["Content"]), np.mean(leans_objOT["Number of Likes"]), np.mean(leans_objOT["Number of Retweets"])))
    print("%s OT subjective tweets. Average likes = %s, average RT = %s" % (len(leans_subjOT["Content"]), np.mean(leans_subjOT["Number of Likes"]), np.mean(leans_subjOT["Number of Retweets"])))
    
    
    ##Do the same for IRT tweets:
    #Remove/replace 'smart' apostrophes and quotation marks with standard keyboard equivalents:
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"’", "'") #replace closing smart apostrophes with regular apostrophe
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"‘", "'") #replace opening smart apostrophes with regular apostrophe
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"“", "\"") #replace opening smart quotes with regular quotes
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"”", "\"") #replace closing smart quotes with regular quotes
    
    #Examine tweets after removing/replacing 'smart' apostrophes and quotes:
    #print(official_tweets["Content"].head(5))
    
    ##Not removing apostrophes for sentiment analysis
    #Remove apostrophes followed by 's' and replace with nothing (Disney's becomes Disney):
    #official_tweets["Content"] = official_tweets["Content"].str.replace(r"'s", "")
    
    #Remove remaining apostrophes and replace with nothing (convert I'm to Im and such):
    #official_tweets["Content"] = official_tweets["Content"].str.replace(r"'", "")    
    
    #IRT tweets: Remove links, @username mentions, non alphanumeric characters, replace @ with at, convert all text to lowercase
    IRT_tweets2 = standardize_text_forSentiment(IRT_tweets, "Content")
    
    #Removing tweets that weren't originally in English
    English_IRT = IRT_tweets2[IRT_tweets2["Language"] == "en"]
    
    #Removing rows with no text left inside them
    filter4 = English_IRT["Content"] != ""
    IRT_clean = English_IRT[filter4]
    
    #Create polarity and subjectivity scores for official tweets only:
    IRT_clean["Polarity"] = IRT_clean["Content"].apply(lambda x:TextBlob(x).polarity)
    IRT_clean["Subjective"] = IRT_clean["Content"].apply(lambda x:TextBlob(x).subjectivity)
    
    
    #Examine results:
    print("\n")
    print("IRT Tweets and polarity, then subjectivity scores:")
    print(IRT_clean["Content"].head(5))
    print(IRT_clean["Polarity"].head(5))
    print(IRT_clean["Subjective"].head(5))
    
    print("Average polarity across all %s IRT tweets: %s" % (company_name, np.mean(IRT_clean["Polarity"])))
    print("Average subjectivity across all %s IRT tweets: %s" % (company_name, np.mean(IRT_clean["Subjective"])))
    
    print("Before splitting, there are %s IRT tweets" % len(IRT_clean["Content"]))
    
    neg_IRT = IRT_clean[IRT_clean["Polarity"] < -1/3]
    neutral_IRT = IRT_clean[IRT_clean["Polarity"].between(-1/3, 1/3)]
    pos_IRT = IRT_clean[IRT_clean["Polarity"] > 1/3]
    
    print("There are %s negative IRT tweets, with an average polarity of %s" % (len(neg_IRT["Content"]), np.mean(neg_IRT["Polarity"])))
    print("There are %s neutral IRT tweets, with an average polarity of %s" % (len(neutral_IRT["Content"]), np.mean(neutral_IRT["Polarity"])))
    print("There are %s positive IRT tweets, with an average polarity of %s" % (len(pos_IRT["Content"]), np.mean(pos_IRT["Polarity"])))
    
    print("Negative IRT tweets: Average likes = %s, Average RT = %s" % (np.mean(neg_IRT["Number of Likes"]), np.mean(neg_IRT["Number of Retweets"])))
    print("Neutral IRT tweets: Average likes = %s, Average RT = %s" % (np.mean(neutral_IRT["Number of Likes"]), np.mean(neutral_IRT["Number of Retweets"])))
    print("Positive IRT tweets: Average likes = %s, Average RT = %s" % (np.mean(pos_IRT["Number of Likes"]), np.mean(pos_IRT["Number of Retweets"])))
    
    
    
    ##Define tweets with subjectivity scores <= 0.5 to be objective, tweets w/ subjectivity scores > 0.5 to be subjective
    leans_objIRT = IRT_clean[IRT_clean["Subjective"] <= 0.5]
    leans_subjIRT = IRT_clean[IRT_clean["Subjective"] > 0.5]
    
    #Examine subjectivity:
    print("%s IRT objective tweets. Average likes = %s, average RT = %s" % (len(leans_objIRT["Content"]), np.mean(leans_objIRT["Number of Likes"]), np.mean(leans_objIRT["Number of Retweets"])))
    print("%s IRT subjective tweets. Average likes = %s, average RT = %s" % (len(leans_subjIRT["Content"]), np.mean(leans_subjIRT["Number of Likes"]), np.mean(leans_subjIRT["Number of Retweets"])))



##In the case that there aren't enough official tweets for split analysis:
elif propOT < minThresh:
    print("It seems that %s doesn't have enough official tweets for split analysis" % company_name)
    

##In the case that there aren't enough IRT tweets for split analysis:
elif propIRT < minThresh:
    print("It seems that %s doesn't have enough IRT tweets for split analysis" % company_name)
    
