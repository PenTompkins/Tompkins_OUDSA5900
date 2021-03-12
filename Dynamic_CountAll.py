# -*- coding: utf-8 -*-
#This file is being created to perform the 'Dynamic_Count.py' process, but keeping tweets with 0 text left after standardization
#E.g. tweets contaning only some combination of links, emojis, and @username mentions will be included in analysis as well
#Furthermore, as 'Dynamic_Count.py' has now been retired, pre-processing improvements have been added to this file instead
#
#Overall, this is program: 28

import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras
import math



#Read in the data: Below line of code will need to be reconfigured for your filepath
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
#print(company_tweets["Content"].head(5))

#Remove/replace 'smart' apostrophes and quotation marks with standard keyboard equivalents:
company_tweets["Content"] = company_tweets["Content"].str.replace(r"’", "'") #replace closing smart apostrophes with regular apostrophe
company_tweets["Content"] = company_tweets["Content"].str.replace(r"‘", "'") #replace opening smart apostrophes with regular apostrophe
company_tweets["Content"] = company_tweets["Content"].str.replace(r"“", "\"") #replace opening smart quotes with regular quotes
company_tweets["Content"] = company_tweets["Content"].str.replace(r"”", "\"") #replace closing smart quotes with regular quotes

#Examine tweets after removing/replacing 'smart' apostrophes and quotes:
#print(company_tweets["Content"].head(5))

#Remove apostrophes followed by 's' and replace with nothing (Disney's becomes Disney):
company_tweets["Content"] = company_tweets["Content"].str.replace(r"'s", "")

#Remove remaining apostrophes and replace with nothing (convert I'm to Im and such):
company_tweets["Content"] = company_tweets["Content"].str.replace(r"'", "")

#Perform standardization on the textual contents of the company's tweets:
#Added in line to replace extra whitespace with a single space
#Added in line to remove commas followed by 3 digits and replace with nothing
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r".", "") #remove/replace periods w/ nothing. Should now count acronyms as one word
    df[text_field] = df[text_field].str.replace(r"&", "and") #replace ampersands with 'and'
    df[text_field] = df[text_field].str.replace(r"http\S+", "") #remove links and replace w/ nothing
    df[text_field] = df[text_field].str.replace(r"http", "") #ensure all links have been removed
    df[text_field] = df[text_field].str.replace(r"@\S+", "") #remove @username mentions and replace with nothing
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")#Remove/replace anything that's not capital/lowercase letter, number, parentheses, comma, or any of the following symbols with a space
    df[text_field] = df[text_field].str.replace(r"@", "at") #replace any remaining '@' symbols with 'at'
    df[text_field] = df[text_field].str.lower() #convert all remaining text to lowercase
    df[text_field] = df[text_field].str.replace(r"\s+", " ") #replace 2+ spaces with single space
    df[text_field] = df[text_field].str.replace(r",(?=\d{3})", "")#remove commas followed by 3 numbers and replace w/ nothing
    return df

textual_tweets = standardize_text(company_tweets, "Content")

#Examine tweets after standardization has been performed:
#print(textual_tweets["Content"].head(5))
#print(textual_tweets.shape)

#Removing tweets that weren't originally in English, or 'UND' (undecided, tweets containing only emojis and such)
English_tweets = textual_tweets[textual_tweets["Language"] == "en"]
#print(English_tweets.shape)
und_tweets = textual_tweets[textual_tweets["Language"] == "und"]
#print(und_tweets["Content"].head(5))
#print(English_tweets["Content"].head(5))
#print("Break")
all_tweets = pd.concat([English_tweets, und_tweets], axis=0)
#print(all_tweets.shape)

#Removing rows with no text left inside them...NOT DOING THAT FOR THIS VERSION
#filter1 = English_tweets["Content"] != ""
#cleanGlish_tweets = English_tweets[filter1]
cleanGlish_tweets = all_tweets.copy()

#Creating tokens:
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

cleanGlish_tweets["tokens"] = cleanGlish_tweets["Content"].apply(tokenizer.tokenize)

#Inspect the data more thoroughly
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

all_words = [word for tokens in cleanGlish_tweets["tokens"] for word in tokens]
tweet_lengths = [len(tokens) for tokens in cleanGlish_tweets["tokens"]]
VOCAB = sorted(list(set(all_words)))
print(company_name)
print("Total number of tweets: %s" % len(cleanGlish_tweets["Content"]))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max tweet length is %s" % max(tweet_lengths))
print("Average tweet length is %s" % np.mean(tweet_lengths))

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
plt.title("All %s Tweets" % company_name)
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of Tweets')
plt.hist(tweet_lengths, bins=range(max(tweet_lengths)))
plt.xticks(range(max(tweet_lengths)))
plt.xticks(rotation=90, ha='right')
plt.show()

#Remove stop words from the data:
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

    
##Expand on the initial set of stopwords:
stop_words2 = pd.DataFrame(stop_words)
stop_words2["Words"] = stop_words
add_stopwords = stop_words2["Words"].str.replace(r"'", "") #replace apostrophes in initial set of stopwords with nothing

#Add the newly created stopwords to the original set:
for word in add_stopwords:
    if word not in stop_words:
        stop_words.add(word)
        
#These words need to be added manually to the set of stopwords:
stop_words.add("wed")
stop_words.add("us")


filtered_toks = []

#Filter out the stop words:
for w in cleanGlish_tweets["tokens"]: #tweet tokens
    for j in w: #word tokens within each tweet
        if j not in stop_words:
            filtered_toks.append(j)

#Examine the word frequencies
from nltk.probability import FreqDist
fdist = FreqDist(filtered_toks)
topWords = [v for v, k in fdist.most_common(20)]
print(fdist.most_common(20))

##Examine n-grams:
##!!ngram code below comes from the following link: https://towardsdatascience.com/from-dataframe-to-n-grams-e34e29df3460
from nltk import ngrams
bigrams_series = (pd.Series(nltk.ngrams(filtered_toks, 2)).value_counts())[:20]
#print(bigrams2)
#print(bigrams_series)

#Tri-grams:
trigrams_series = (pd.Series(nltk.ngrams(filtered_toks, 3)).value_counts())[:20]
#print(trigrams_series)

#Adjust fontsize of labels to see bigrams and trigrams more clearly:
SMALL_SIZE = 5
plt.rc('ytick', labelsize=SMALL_SIZE)

#Visualize bi-grams:
bigrams_series.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
plt.title("20 Most Common %s Bi-grams" % company_name)
plt.ylabel("Bi-gram")
plt.xlabel("Number of Occurences")
plt.show()

#Visualize tri-grams:
trigrams_series.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
plt.title("20 Most Common %s Tri-grams" % company_name)
plt.ylabel("Tri-gram")
plt.xlabel("Number of Occurences")
plt.show()


#####Same analysis as above, after OT/IRT splits have been made:
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
#print(official_tweets.shape)
#print(IRT_tweets.shape)
                               
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

##If there are enough tweets in both categories, perform full analysis:
if propOT >= minThresh and propIRT >= minThresh:
    print("%s has enough tweets for full analysis" % company_name)
    
    #Examine first 5 tweets before any alterations are made
    #print(official_tweets["Content"].head(5))
    
    #Remove/replace 'smart' apostrophes and quotation marks with standard keyboard equivalents:
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"’", "'") #replace closing smart apostrophes with regular apostrophe
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"‘", "'") #replace opening smart apostrophes with regular apostrophe
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"“", "\"") #replace opening smart quotes with regular quotes
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"”", "\"") #replace closing smart quotes with regular quotes
    
    #Examine tweets after removing/replacing 'smart' apostrophes and quotes:
    #print(official_tweets["Content"].head(5))
    
    #Remove apostrophes followed by 's' and replace with nothing (Disney's becomes Disney):
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"'s", "")
    
    #Remove remaining apostrophes and replace with nothing (convert I'm to Im and such):
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"'", "")    
    
    #Official tweets: Remove links, @username mentions, non alphanumeric characters, replace @ with at, convert all text to lowercase
    official_tweets2 = standardize_text(official_tweets, "Content")
    
    
    #Removing tweets that weren't originally in English, or 'UND' (undecided, tweets containing only emojis and such)
    English_official = official_tweets2[official_tweets2["Language"] == "en"]
    #print(English_official.shape)
    und_official = official_tweets2[official_tweets2["Language"] == "und"]
    all_official = pd.concat([English_official, und_official], axis=0)
    #print(all_official.shape)
    
    #Removing rows with no text left inside them...NOT ON THIS VERSION
    #filter1 = English_tweets["Content"] != ""
    #cleanGlish_tweets = English_tweets[filter1]
    #cleanGlish_tweets = all_official.copy() 
    official_clean = all_official.copy()
    
    #Create tweet and word tokens:
    official_clean["tokens"] = official_clean["Content"].apply(tokenizer.tokenize)
    
    
    #Inspect the data more thoroughly
    all_words_official = [word for tokens in official_clean["tokens"] for word in tokens]
    OT_tweet_lengths = [len(tokens) for tokens in official_clean["tokens"]]
    VOCAB_official = sorted(list(set(all_words_official)))
    print(company_name)
    print("Total number of official tweets: %s" % len(official_clean["Content"]))
    print("Official tweets have %s words total, with a vocabulary size of %s" % (len(all_words_official), len(VOCAB_official)))
    print("Max official tweet length is %s" % max(OT_tweet_lengths))
    print("Average official tweet length is %s" % np.mean(OT_tweet_lengths))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("All %s Official Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(OT_tweet_lengths, bins=range(max(OT_tweet_lengths)))
    plt.xticks(range(max(OT_tweet_lengths)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks2 = []
    
    #Filter out the stop words:
    for w in official_clean["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks2.append(j)
    
    #Examine the word frequencies
    #from nltk.probability import FreqDist
    fdist2 = FreqDist(filtered_toks2)
    topWords_official = [v for v, k in fdist2.most_common(20)]
    print(fdist2.most_common(20))
    
    #Examine official n-grams:
    bigrams_official = (pd.Series(nltk.ngrams(filtered_toks2, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_official = (pd.Series(nltk.ngrams(filtered_toks2, 3)).value_counts())[:20]
    
    #Visualize official bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_official.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Official Tweet Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize tri-grams:
    trigrams_official.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Official Tweet Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    ##Perform same analysis over top 25% of company official tweets only:
    #Specify the number of rows to take as the top 25% of official tweets:
    #nr = int(round(0.25 * len(official_clean)))
    nr = math.ceil(0.25 * len(official_clean))
    
    #Sort the data based on number of likes:
    official_clean = official_clean.sort_values(by = "Number of Likes", ascending=False)    
    
    #Extract the top 25% of official tweets from the sorted data:
    top_official = official_clean.head(nr).copy()
    
    #Inspect the data more thoroughly
    all_words_topOT = [word for tokens in top_official["tokens"] for word in tokens]
    topOT_tweet_lengths = [len(tokens) for tokens in top_official["tokens"]]
    VOCAB_topOT = sorted(list(set(all_words_topOT)))
    #print(len(all_words_topOT))
    #print(len(VOCAB_topOT))
    print(company_name)
    print("Total number of top official tweets: %s" % len(top_official["Content"]))
    print("Top 25 percent of official tweets have %s words total, with a vocabulary size of %s" % (len(all_words_topOT), len(VOCAB_topOT)))
    print("Max top official tweet length is %s" % max(topOT_tweet_lengths))
    print("Average top official tweet length is %s" % np.mean(topOT_tweet_lengths))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("Top 25 Pct of %s Official Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(topOT_tweet_lengths, bins=range(max(topOT_tweet_lengths)))
    plt.xticks(range(max(topOT_tweet_lengths)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks3 = []
    
    #Filter out the stop words:
    for w in top_official["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks3.append(j)
    
    #Examine the word frequencies
    #from nltk.probability import FreqDist
    fdist3 = FreqDist(filtered_toks3)
    topWords_OT25 = [v for v, k in fdist3.most_common(20)]
    print(fdist3.most_common(20))    
    
    #Examine top 25% official n-grams:
    bigrams_OT25 = (pd.Series(nltk.ngrams(filtered_toks3, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_OT25 = (pd.Series(nltk.ngrams(filtered_toks3, 3)).value_counts())[:20]
    
    #Visualize top 25% official bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_OT25.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Top 25 Pct Official Tweet Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize top 25% official tri-grams:
    trigrams_OT25.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Top 25 Pct Official Tweet Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    
    ##Perform same analysis over median 50% of official company tweets:
    #Grab the median 50% of official company tweets:
    print("Across all official tweets, there are %i tweets" % len(official_clean))
    print("In the top 25 pct there are %i tweets" % len(top_official))
    desired_num = int(round((0.5 * len(official_clean))))
    desired_num2 = desired_num - 1
    print("We should get %i tweets for middle 50 pct" % desired_num2)
    nr2 = nr + 1
    endPoint = int(nr + desired_num)
    #Ensure that we're grabbing correct amount of median tweets:
    if len(official_clean) % 4 == 0: #if length of tweets modulo 4 equals 0
        endPoint = endPoint + 1 #then we need to grab one more tweet for median 50%
    midTweets_OT = official_clean.iloc[nr2:endPoint]
    print("And we got %i tweets" % len(midTweets_OT))
    
    #Examine the middle 50% of official tweets more thoroughly
    all_words_mid_OT = [word for tokens in midTweets_OT["tokens"] for word in tokens]
    tweet_lengths_mid_OT = [len(tokens) for tokens in midTweets_OT["tokens"]]
    VOCAB_mid_OT = sorted(list(set(all_words_mid_OT)))
    print(company_name)
    print("Total number of median official tweets: %s" % len(midTweets_OT["Content"]))
    print("Median official tweets have %s words total, with a vocabulary size of %s" % (len(all_words_mid_OT), len(VOCAB_mid_OT)))
    print("Max tweet length is %s" % max(tweet_lengths_mid_OT))
    print("Average tweet length is %s" % np.mean(tweet_lengths_mid_OT))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("Median 50 Pct %s Official Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(tweet_lengths_mid_OT, bins=range(max(tweet_lengths_mid_OT)))
    plt.xticks(range(max(tweet_lengths_mid_OT)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks4 = []
    
    #Filter out the stop words:
    for w in midTweets_OT["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks4.append(j)
    
    #Examine the word frequencies
    #from nltk.probability import FreqDist
    fdist4 = FreqDist(filtered_toks4)
    topWords_mid_OT = [v for v, k in fdist4.most_common(20)]
    print(fdist4.most_common(20))   
    
    #Examine median 50% official n-grams:
    bigrams_OT50 = (pd.Series(nltk.ngrams(filtered_toks4, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_OT50 = (pd.Series(nltk.ngrams(filtered_toks4, 3)).value_counts())[:20]
    #print(trigrams_series)
    
    #Visualize median 50% official bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_OT50.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Median 50 Pct Official Tweet Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize tri-grams:
    trigrams_OT50.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Median 50 Pct Official Tweet Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    
    ##Perform same analysis over bottom 25% of official tweets:
    #Grab the bottom 25% performing official tweets
    bottom_official = official_clean.tail(nr).copy()
    #print(bottom_official.shape)
    
    #Inspect the data more thoroughly
    all_words_bOT = [word for tokens in bottom_official["tokens"] for word in tokens]
    bOT_tweet_lengths = [len(tokens) for tokens in bottom_official["tokens"]]
    VOCAB_bOT = sorted(list(set(all_words_bOT)))
    #print(len(all_words_topOT))
    #print(len(VOCAB_topOT))
    print(company_name)
    print("Total number of bottom official tweets: %s" % len(bottom_official["Content"]))
    print("Bottom 25 pct of official tweets have %s words total, with a vocabulary size of %s" % (len(all_words_bOT), len(VOCAB_bOT)))
    print("Max bottom official tweet length is %s" % max(bOT_tweet_lengths))
    print("Average bottom official tweet length is %s" % np.mean(bOT_tweet_lengths))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("Bottom 25 Pct of %s Official Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(bOT_tweet_lengths, bins=range(max(bOT_tweet_lengths)))
    plt.xticks(range(max(bOT_tweet_lengths)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks5 = []
    
    #Filter out the stop words:
    for w in bottom_official["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks5.append(j)
    
    #Examine the word frequencies
    #from nltk.probability import FreqDist
    fdist5 = FreqDist(filtered_toks5)
    topWords_OTb25 = [v for v, k in fdist5.most_common(20)]
    print(fdist5.most_common(20))    

    #Examine bottom 25% official n-grams:
    bigrams_bOT = (pd.Series(nltk.ngrams(filtered_toks5, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_bOT = (pd.Series(nltk.ngrams(filtered_toks5, 3)).value_counts())[:20]
    #print(trigrams_series)
    
    #Visualize bottom 25% official bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_bOT.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Bottom 25 Pct Official Tweet Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize tri-grams:
    trigrams_bOT.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Bottom 25 Pct Official Tweet Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    
    ##Perform same analysis across all IRT tweets for the company:
    #Remove/replace 'smart' apostrophes and quotation marks with standard keyboard equivalents:
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"’", "'") #replace closing smart apostrophes with regular apostrophe
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"‘", "'") #replace opening smart apostrophes with regular apostrophe
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"“", "\"") #replace opening smart quotes with regular quotes
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"”", "\"") #replace closing smart quotes with regular quotes
    
    #Examine tweets after removing/replacing 'smart' apostrophes and quotes:
    #print(IRT_tweets["Content"].head(5))
    
    #Remove apostrophes followed by 's' and replace with nothing (Disney's becomes Disney):
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"'s", "")
    
    #Remove remaining apostrophes and replace with nothing (convert I'm to Im and such):
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"'", "")    
    
    #IRT tweets: Remove links, @username mentions, non alphanumeric characters, replace @ with at, convert all text to lowercase
    IRT_tweets2 = standardize_text(IRT_tweets, "Content")
    
    #Removing tweets that weren't originally in English or 'und'
    English_IRT = IRT_tweets2[IRT_tweets2["Language"] == "en"]
    und_IRT = IRT_tweets2[IRT_tweets2["Language"] == "und"]
    IRT_clean = pd.concat([English_IRT, und_IRT], axis=0)
    
    
    #Create tweet and word tokens:
    IRT_clean["tokens"] = IRT_clean["Content"].apply(tokenizer.tokenize)
    
    
    #Inspect the data more thoroughly
    all_words_IRT = [word for tokens in IRT_clean["tokens"] for word in tokens]
    IRT_tweet_lengths = [len(tokens) for tokens in IRT_clean["tokens"]]
    VOCAB_IRT = sorted(list(set(all_words_IRT)))
    print(company_name)
    print("Total number of IRT tweets: %s" % len(IRT_clean["Content"]))
    print("IRT tweets have %s words total, with a vocabulary size of %s" % (len(all_words_IRT), len(VOCAB_IRT)))
    print("Max IRT tweet length is %s" % max(IRT_tweet_lengths))
    print("Average IRT tweet length is %s" % np.mean(IRT_tweet_lengths))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("All %s IRT Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(IRT_tweet_lengths, bins=range(max(IRT_tweet_lengths)))
    plt.xticks(range(max(IRT_tweet_lengths)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks6 = []
    
    #Filter out the stop words:
    for w in IRT_clean["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks6.append(j)
    
    #Examine the word frequencies
    #from nltk.probability import FreqDist
    fdist6 = FreqDist(filtered_toks6)
    topWords_IRT = [v for v, k in fdist6.most_common(20)]
    print(fdist6.most_common(20))    
    
    #Examine n-grams for all company IRT tweets:
    bigrams_IRT = (pd.Series(nltk.ngrams(filtered_toks6, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_IRT = (pd.Series(nltk.ngrams(filtered_toks6, 3)).value_counts())[:20]
    #print(trigrams_series)
    
    #Visualize IRT bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_IRT.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s IRT Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize tri-grams:
    trigrams_IRT.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s IRT Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show()    
    
    
    ##Perform same analysis over top 25% of IRT tweets:
    #Specify the number of rows to take as the top 25% of IRT tweets:
    #nr = int(round(0.25 * len(IRT_clean)))
    nr = math.ceil(0.25 * len(IRT_clean))
    
    #Sort the data based on number of likes:
    IRT_clean = IRT_clean.sort_values(by = "Number of Likes", ascending=False)
    
    #Extract the top 25% of IRT tweets from the sorted data:
    top_IRT = IRT_clean.head(nr).copy()
    #print(top_official.shape)
    
    #Inspect the data more thoroughly
    all_words_topIRT = [word for tokens in top_IRT["tokens"] for word in tokens]
    topIRT_tweet_lengths = [len(tokens) for tokens in top_IRT["tokens"]]
    VOCAB_topIRT = sorted(list(set(all_words_topIRT)))
    #print(len(all_words_topOT))
    #print(len(VOCAB_topOT))
    print(company_name)
    print("Total number of top IRT tweets: %s" % len(top_IRT["Content"]))
    print("Top 25 percent of IRT tweets have %s words total, with a vocabulary size of %s" % (len(all_words_topIRT), len(VOCAB_topIRT)))
    print("Max top IRT tweet length is %s" % max(topIRT_tweet_lengths))
    print("Average top IRT tweet length is %s" % np.mean(topIRT_tweet_lengths))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("Top 25 Pct of %s IRT Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(topIRT_tweet_lengths, bins=range(max(topIRT_tweet_lengths)))
    plt.xticks(range(max(topIRT_tweet_lengths)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks7 = []
    
    #Filter out the stop words:
    for w in top_IRT["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks7.append(j)
    
    #Examine the word frequencies
    #from nltk.probability import FreqDist
    fdist7 = FreqDist(filtered_toks7)
    topWords_IRT25 = [v for v, k in fdist7.most_common(20)]
    print(fdist7.most_common(20))  
    
    #Examine n-grams for top 25% IRT tweets:
    bigrams_IRT25 = (pd.Series(nltk.ngrams(filtered_toks7, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_IRT25 = (pd.Series(nltk.ngrams(filtered_toks7, 3)).value_counts())[:20]
    #print(trigrams_series)
    
    #Visualize top 25% IRT bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_IRT25.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Top 25 Pct IRT Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize tri-grams:
    trigrams_IRT25.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Top 25 Pct IRT Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show() 
    
    
    ##Perform same analysis over median 50% of IRT tweets:
    #Grab the median 50% of IRT company tweets:
    print("Across all IRT tweets, there are %i tweets" % len(IRT_clean))
    print("In the top 25 pct there are %i tweets" % len(top_IRT))
    desired_num = int(round((0.5 * len(IRT_clean))))
    desired_num2 = desired_num - 1
    print("We should get %i tweets for middle 50 pct" % desired_num2)
    nr2 = nr + 1
    endPoint = int(nr + desired_num)
    #Ensure that we're grabbing correct number of tweets for median 50%:
    if len(IRT_clean) % 4 == 0:
        endPoint = endPoint + 1
    midTweets_IRT = IRT_clean.iloc[nr2:endPoint]
    print("And we got %i tweets" % len(midTweets_IRT))
    
    #Examine the middle 50% of IRT tweets more thoroughly
    all_words_mid_IRT = [word for tokens in midTweets_IRT["tokens"] for word in tokens]
    tweet_lengths_mid_IRT = [len(tokens) for tokens in midTweets_IRT["tokens"]]
    VOCAB_mid_IRT = sorted(list(set(all_words_mid_IRT)))
    print(company_name)
    print("Total number of median IRT tweets: %s" % len(midTweets_IRT["Content"]))
    print("Median IRT tweets have %s words total, with a vocabulary size of %s" % (len(all_words_mid_IRT), len(VOCAB_mid_IRT)))
    print("Max tweet length is %s" % max(tweet_lengths_mid_IRT))
    print("Average tweet length is %s" % np.mean(tweet_lengths_mid_IRT))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("Median 50 Pct %s IRT Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(tweet_lengths_mid_IRT, bins=range(max(tweet_lengths_mid_IRT)))
    plt.xticks(range(max(tweet_lengths_mid_IRT)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks8 = []
    
    #Filter out the stop words:
    for w in midTweets_IRT["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks8.append(j)
    
    #Examine the word frequencies
    #from nltk.probability import FreqDist
    fdist8 = FreqDist(filtered_toks8)
    topWords_mid_IRT = [v for v, k in fdist8.most_common(20)]
    print(fdist8.most_common(20))   
    
    #Examine median 50% IRT n-grams:
    bigrams_IRT50 = (pd.Series(nltk.ngrams(filtered_toks8, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_IRT50 = (pd.Series(nltk.ngrams(filtered_toks8, 3)).value_counts())[:20]
    
    #Visualize median 50% IRT bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_IRT50.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Median 50 Pct IRT Tweet Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize tri-grams:
    trigrams_IRT50.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Median 50 Pct IRT Tweet Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    
    ##Perform same analysis over bottom 25% of IRT tweets:
    #Grab the bottom 25% of company IRT tweets:
    bottom_IRT = IRT_clean.tail(nr).copy()
    #print(bottom_official.shape)
    
    #Inspect the data more thoroughly
    all_words_bIRT = [word for tokens in bottom_IRT["tokens"] for word in tokens]
    bIRT_tweet_lengths = [len(tokens) for tokens in bottom_IRT["tokens"]]
    VOCAB_bIRT = sorted(list(set(all_words_bIRT)))
    #print(len(all_words_topOT))
    #print(len(VOCAB_topOT))
    print(company_name)
    print("Total number of bottom IRT tweets: %s" % len(bottom_IRT["Content"]))
    print("Bottom 25 percent of IRT tweets have %s words total, with a vocabulary size of %s" % (len(all_words_bIRT), len(VOCAB_bIRT)))
    print("Max bottom IRT tweet length is %s" % max(bIRT_tweet_lengths))
    print("Average bottom IRT tweet length is %s" % np.mean(bIRT_tweet_lengths))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("Bottom 25 Pct of %s IRT Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(bIRT_tweet_lengths, bins=range(max(bIRT_tweet_lengths)))
    plt.xticks(range(max(bIRT_tweet_lengths)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    
    filtered_toks9 = []
    
    #Filter out the stop words:
    for w in bottom_IRT["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks9.append(j)
    
    #Examine the word frequencies
    fdist9 = FreqDist(filtered_toks9)
    topWords_IRTb25 = [v for v, k in fdist9.most_common(20)]
    print(fdist9.most_common(20))    

    #Examine bottom 25% IRT n-grams:
    bigrams_bIRT = (pd.Series(nltk.ngrams(filtered_toks9, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_bIRT = (pd.Series(nltk.ngrams(filtered_toks9, 3)).value_counts())[:20]
    
    #Visualize bottom 25% IRT bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_bIRT.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Bottom 25 Pct IRT Tweet Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize tri-grams:
    trigrams_bIRT.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Bottom 25 Pct IRT Tweet Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #####################################################################################################################################
    #Search for differences within the 9 lists:
    
    #Words unique to the list of common terms across all tweets:
    allTweets_unique = []
    
    for word in topWords: #for each word in the list of top 20 common terms, find words unique to this list
        if word not in topWords_OT25:
            if word not in topWords_official:
                if word not in topWords_OTb25:
                    if word not in topWords_IRT:
                        if word not in topWords_IRT25:
                            if word not in topWords_IRTb25:
                                if word not in topWords_mid_OT:
                                    if word not in topWords_mid_IRT:
                                        allTweets_unique.append(word)
                        
    print("The words unique to the list for all tweets:")                
    print(allTweets_unique)
    
    
    
    #Words unique to the list of common terms across all official tweets:
    allOT_unique = []
    
    for word in topWords_official: #for each word in the list of top 20 common terms for official tweets, find words unique to this list
        if word not in topWords_OT25:
            if word not in topWords:
                if word not in topWords_OTb25:
                    if word not in topWords_IRT:
                        if word not in topWords_IRT25:
                            if word not in topWords_IRTb25:
                                if word not in topWords_mid_OT:
                                    if word not in topWords_mid_IRT:
                                        allOT_unique.append(word)
                        
    print("The words unique to the list for all OT tweets:")                
    print(allOT_unique)
    
    
    
    
    #Words unique to the list of common terms across all IRT tweets:
    allIRT_unique = []
    
    for word in topWords_IRT: #for each word in the list of top 20 common terms across IRT tweets, find words unique to this list
        if word not in topWords_OT25:
            if word not in topWords_official:
                if word not in topWords_OTb25:
                    if word not in topWords:
                        if word not in topWords_IRT25:
                            if word not in topWords_IRTb25:
                                if word not in topWords_mid_OT:
                                    if word not in topWords_mid_IRT:
                                        allIRT_unique.append(word)
                        
    print("The words unique to the list for all IRT tweets:")                
    print(allIRT_unique)
    
    
    
    
    
    #Words unique to the list of common terms for the top 25% of official tweets:
    top20_OTunique = []
    
    for word in topWords_OT25: #for each word in the list of top 20 terms across the company's top 25% of official tweets 
        if word not in topWords: #if the word isn't just one of the top words across all the company's tweets
            if word not in topWords_official: #and it isn't just a common word in their official tweets
                if word not in topWords_OTb25: #and the word isn't in the list of top terms for the bottom 25% of official tweets
                    if word not in topWords_IRT: #and the word isn't common in the company's IRT tweets
                        if word not in topWords_IRT25: #and the word isn't common to their top 25% of IRT tweets...although it may not hurt to have something in common. Oh well, I'm finding words unique entirely to top25OT rn
                            if word not in topWords_IRTb25: #if the word isn't common their their bottom 25% of IRT tweets
                                if word not in topWords_mid_OT: #and word isn't common to your average official tweet
                                    if word not in topWords_mid_IRT:#and word isn't common to your average IRT tweet
                                        top20_OTunique.append(word)#then it might be worth taking a look at
                        
    
    print("The words unique to the list for top 25 pct of official tweets:")                
    print(top20_OTunique)
    
    
    
    
    #Words unique to the list of common terms across bottom 25% of official tweets:
    bottom20_OTunique = []
    
    for word in topWords_OTb25: #for each word in the list of top 20 common terms across the bottom 25% of official tweets, find words unique to this list
        if word not in topWords_OT25:
            if word not in topWords_official:
                if word not in topWords:
                    if word not in topWords_IRT:
                        if word not in topWords_IRT25:
                            if word not in topWords_IRTb25:
                                if word not in topWords_mid_OT:
                                    if word not in topWords_mid_IRT:
                                        bottom20_OTunique.append(word)
                        
    print("The words unique to the list for bottom 25 pct official tweets:")                
    print(bottom20_OTunique)
    
    
    
    
    #Words unique to the list of common terms across top 25% of IRT tweets:
    IRT25_unique = []
    
    for word in topWords_IRT25: #for each word in the list of top 20 common terms for top 25% of IRT tweets, find words unique to this list
        if word not in topWords_OT25:
            if word not in topWords_official:
                if word not in topWords_OTb25:
                    if word not in topWords_IRT:
                        if word not in topWords:
                            if word not in topWords_IRTb25:
                                if word not in topWords_mid_OT:
                                    if word not in topWords_mid_IRT:
                                        IRT25_unique.append(word)
                        
    print("The words unique to the list for top 25 pct IRT tweets:")                
    print(IRT25_unique)
    
    
    
    
    #Words unique to the list of common terms across bottom 25% of IRT tweets:
    IRTb25_unique = []
    
    for word in topWords_IRTb25: #for each word in the list of top 20 common terms for bottom 25% of IRT tweets, find words unique to this list
        if word not in topWords_OT25:
            if word not in topWords_official:
                if word not in topWords_OTb25:
                    if word not in topWords_IRT:
                        if word not in topWords_IRT25:
                            if word not in topWords:
                                if word not in topWords_mid_OT:
                                    if word not in topWords_mid_IRT:
                                        IRTb25_unique.append(word)
                        
    print("The words unique to the list for bottom 25 pct IRT tweets:")                
    print(IRTb25_unique)
    
    
    
    #Words unique to the list of common terms across median 50% of OT tweets:
    OTm50_unique = []
    
    for word in topWords_mid_OT: #for each word in the list of top 20 common terms for median 50% of OT tweets, find words unique to this list
        if word not in topWords_OT25:
            if word not in topWords_official:
                if word not in topWords_OTb25:
                    if word not in topWords_IRT:
                        if word not in topWords_IRT25:
                            if word not in topWords:
                                if word not in topWords_mid_IRT:
                                    if word not in topWords_IRTb25:
                                        OTm50_unique.append(word)
                        
    print("The words unique to the list for median 50 pct of official tweets:")                
    print(OTm50_unique)
    
    
    #Words unique to the list of common terms across median 50% of IRT tweets:
    IRTm50_unique = []
    
    for word in topWords_mid_IRT: #for each word in the list of top 20 common terms for median 50% of OT tweets, find words unique to this list
        if word not in topWords_OT25:
            if word not in topWords_official:
                if word not in topWords_OTb25:
                    if word not in topWords_IRT:
                        if word not in topWords_IRT25:
                            if word not in topWords:
                                if word not in topWords_mid_OT:
                                    if word not in topWords_IRTb25:
                                        IRTm50_unique.append(word)
                        
    print("The words unique to the list for median 50 pct of IRT tweets:")                
    print(IRTm50_unique)
    
    
    #####################################################################################################
    #Find words that are present in both top 25 categories, but nowhere else:
    inBothBest = []
    justInBest = []
    
    #Find words that are in both lists of top 25
    for word in topWords_OT25:
        if word in topWords_IRT25:
            inBothBest.append(word)
            
    
    #Only take the words from that which don't appear in any other lists
    for word in inBothBest:
        if word not in topWords: #if the word isn't common across all tweets
            if word not in topWords_official: #if the word isn't common accross official tweets
                if word not in topWords_IRT: #if the word isn't common across all IRT tweets
                    if word not in topWords_OTb25: #if the word isn't common in the bottom 25% of official tweets
                        if word not in topWords_IRTb25: #if the word isn't common in the bottom 25% of IRT tweets
                            if word not in topWords_mid_OT:#if the word isn't common to your average official tweet
                                if word not in topWords_mid_IRT:#and the word isn't common to your average IRT tweet
                                    justInBest.append(word)#then let's take a look at this word
    
    print("Words unique to the lists of top 25's:")
    print(justInBest)
    #For Amazon, there were no words unique to the list of top 25's
    #Same for BMW
    
    #################################################################################################################
    #Find words that are present in both bottom 25 categories, but nowhere else:
    inBothWorst = []
    justInWorst = []
    
    #Find words that appear in both bottom 25% lists:
    for word in topWords_OTb25:
        if word in topWords_IRTb25:
            inBothWorst.append(word)
    
    #Only take the words from that which don't appear in any other lists:
    for word in inBothWorst:
        if word not in topWords:#if the word isn't common across all tweets
            if word not in topWords_official: #if the word isn't common to official tweets
                if word not in topWords_IRT: #if the word isn't common to IRT tweets
                    if word not in topWords_OT25: #if the word isn't common to top 25% of official tweets
                        if word not in topWords_IRT25: #if the word isn't common to top 25% of IRT tweets
                            if word not in topWords_mid_OT:#if the word isn't common to your average official tweet
                                if word not in topWords_mid_IRT:#and the word isn't common to your average IRT tweet
                                    justInWorst.append(word)#then this word is uniquely common to poorly performing tweets
                            justInWorst.append(word) #then this word is uniquely common to poorly performing tweets
                            
    print("Words unique to the list of bottom 25's:")
    print(justInWorst)    
    
    
    
###########################################################################################################################################################
###########################################################################################################################################################
#In the case that there aren't enough official tweets for analysis:
elif propOT < minThresh:
    print("%s doesn't seem to have enough official tweets for analysis. All tweets will be used, which mostly consist of IRT" % company_name)
    
    #To make things easier on myself, instead of replacing all 'IRT' occurrences with either 'all' or 'company', I'll instead do this:
    IRT_tweets = company_tweets.copy()
    
    ##Perform same analysis across all IRT tweets (e.g. all tweets) for the company:
    #Remove/replace 'smart' apostrophes and quotation marks with standard keyboard equivalents:
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"’", "'") #replace closing smart apostrophes with regular apostrophe
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"‘", "'") #replace opening smart apostrophes with regular apostrophe
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"“", "\"") #replace opening smart quotes with regular quotes
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"”", "\"") #replace closing smart quotes with regular quotes
    
    #Examine tweets after removing/replacing 'smart' apostrophes and quotes:
    #print(IRT_tweets["Content"].head(5))
    
    #Remove apostrophes followed by 's' and replace with nothing (Disney's becomes Disney):
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"'s", "")
    
    #Remove remaining apostrophes and replace with nothing (convert I'm to Im and such):
    IRT_tweets["Content"] = IRT_tweets["Content"].str.replace(r"'", "")    
    
    #All tweets: Remove links, @username mentions, non alphanumeric characters, replace @ with at, convert all text to lowercase
    IRT_tweets2 = standardize_text(IRT_tweets, "Content")
    
    #Removing tweets that weren't originally in English or 'und'
    English_IRT = IRT_tweets2[IRT_tweets2["Language"] == "en"]
    und_IRT = IRT_tweets2[IRT_tweets2["Language"] == "und"]
    IRT_clean = pd.concat([English_IRT, und_IRT], axis=0)
    
    #Removing rows with no text left inside them
    #filter4 = English_IRT["Content"] != ""
    #IRT_clean = English_IRT[filter4]
    #print(IRT_clean.shape) #2753, thus above processes removed 175 rows of data from the IRT category
    
    #Create tweet and word tokens:
    IRT_clean["tokens"] = IRT_clean["Content"].apply(tokenizer.tokenize)
       
    
    
    ##Perform same analysis over top 25% of all tweets: (in this case, mostly IRT)
    #Specify the number of rows to take as the top 25% of all tweets:
    #nr = int(round(0.25 * len(IRT_clean)))
    nr = math.ceil(0.25 * len(IRT_clean))
    
    #Sort the data based on number of likes:
    IRT_clean = IRT_clean.sort_values(by = "Number of Likes", ascending=False)
    
    #Extract the top 25% of all tweets from the sorted data:
    top_IRT = IRT_clean.head(nr).copy()
    #print(top_official.shape)
    
    #Inspect the data more thoroughly
    all_words_topIRT = [word for tokens in top_IRT["tokens"] for word in tokens]
    topIRT_tweet_lengths = [len(tokens) for tokens in top_IRT["tokens"]]
    VOCAB_topIRT = sorted(list(set(all_words_topIRT)))
    #print(len(all_words_topOT))
    #print(len(VOCAB_topOT))
    print(company_name)
    print("Total number in top 25 pct of tweets: %s" % len(top_IRT["Content"]))
    print("Top 25 percent of tweets have %s words total, with a vocabulary size of %s" % (len(all_words_topIRT), len(VOCAB_topIRT)))
    print("Max top tweet length is %s" % max(topIRT_tweet_lengths))
    print("Average top tweet length is %s" % np.mean(topIRT_tweet_lengths))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("Top 25 Pct of %s Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(topIRT_tweet_lengths, bins=range(max(topIRT_tweet_lengths)))
    plt.xticks(range(max(topIRT_tweet_lengths)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks7 = []
    
    #Filter out the stop words:
    for w in top_IRT["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks7.append(j)
    
    #Examine the word frequencies
    #from nltk.probability import FreqDist
    fdist7 = FreqDist(filtered_toks7)
    topWords_IRT25 = [v for v, k in fdist7.most_common(20)]
    print(fdist7.most_common(20))  
    
    #Examine n-grams for top 25% tweets:
    bigrams_IRT25 = (pd.Series(nltk.ngrams(filtered_toks7, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_IRT25 = (pd.Series(nltk.ngrams(filtered_toks7, 3)).value_counts())[:20]
    
    #Visualize top 25% bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_IRT25.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Top 25 Pct Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize tri-grams:
    trigrams_IRT25.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Top 25 Pct Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show() 
    
    
    ##Perform same analysis over median 50% of IRT tweets: (in this case, will mostly be IRT)
    #Grab the median 50% of all company tweets:
    print("Across all tweets, there are %i tweets" % len(IRT_clean))
    print("In the top 25 pct there are %i tweets" % len(top_IRT))
    desired_num = int(round((0.5 * len(IRT_clean))))
    desired_num2 = desired_num - 1
    print("We should get %i tweets for middle 50 pct" % desired_num2)
    nr2 = nr + 1
    endPoint = int(nr + desired_num)
    #Ensure that we're grabbing correct number of tweets for median 50%:
    if len(IRT_clean) % 4 == 0:
        endPoint = endPoint + 1
    midTweets_IRT = IRT_clean.iloc[nr2:endPoint]
    print("And we got %i tweets" % len(midTweets_IRT))
    
    #Examine the middle 50% of all tweets more thoroughly
    all_words_mid_IRT = [word for tokens in midTweets_IRT["tokens"] for word in tokens]
    tweet_lengths_mid_IRT = [len(tokens) for tokens in midTweets_IRT["tokens"]]
    VOCAB_mid_IRT = sorted(list(set(all_words_mid_IRT)))
    print(company_name)
    print("Total number of median tweets: %s" % len(midTweets_IRT["Content"]))
    print("Median tweets have %s words total, with a vocabulary size of %s" % (len(all_words_mid_IRT), len(VOCAB_mid_IRT)))
    print("Max tweet length is %s" % max(tweet_lengths_mid_IRT))
    print("Average tweet length is %s" % np.mean(tweet_lengths_mid_IRT))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("Median 50 Pct %s Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(tweet_lengths_mid_IRT, bins=range(max(tweet_lengths_mid_IRT)))
    plt.xticks(range(max(tweet_lengths_mid_IRT)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks8 = []
    
    #Filter out the stop words:
    for w in midTweets_IRT["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks8.append(j)
    
    #Examine the word frequencies
    fdist8 = FreqDist(filtered_toks8)
    topWords_mid_IRT = [v for v, k in fdist8.most_common(20)]
    print(fdist8.most_common(20))   
    
    #Examine median 50% n-grams:
    bigrams_IRT50 = (pd.Series(nltk.ngrams(filtered_toks8, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_IRT50 = (pd.Series(nltk.ngrams(filtered_toks8, 3)).value_counts())[:20]
    
    #Visualize median 50% bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_IRT50.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Median 50 Pct Tweet Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize tri-grams:
    trigrams_IRT50.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Median 50 Pct Tweet Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    
    ##Perform same analysis over bottom 25% of all tweets: (in this case, mostly IRT)
    #Grab the bottom 25% of company all tweets:
    bottom_IRT = IRT_clean.tail(nr).copy()
    #print(bottom_official.shape)
    
    #Inspect the data more thoroughly
    all_words_bIRT = [word for tokens in bottom_IRT["tokens"] for word in tokens]
    bIRT_tweet_lengths = [len(tokens) for tokens in bottom_IRT["tokens"]]
    VOCAB_bIRT = sorted(list(set(all_words_bIRT)))
    #print(len(all_words_topOT))
    #print(len(VOCAB_topOT))
    print(company_name)
    print("Total number of bottom 25 pct tweets: %s" % len(bottom_IRT["Content"]))
    print("Bottom 25 percent of tweets have %s words total, with a vocabulary size of %s" % (len(all_words_bIRT), len(VOCAB_bIRT)))
    print("Max bottom tweet length is %s" % max(bIRT_tweet_lengths))
    print("Average bottom tweet length is %s" % np.mean(bIRT_tweet_lengths))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("Bottom 25 Pct of %s Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(bIRT_tweet_lengths, bins=range(max(bIRT_tweet_lengths)))
    plt.xticks(range(max(bIRT_tweet_lengths)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks9 = []
    
    #Filter out the stop words:
    for w in bottom_IRT["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks9.append(j)
    
    #Examine the word frequencies
    #from nltk.probability import FreqDist
    fdist9 = FreqDist(filtered_toks9)
    topWords_IRTb25 = [v for v, k in fdist9.most_common(20)]
    print(fdist9.most_common(20))    

    #Examine bottom 25% n-grams:
    bigrams_bIRT = (pd.Series(nltk.ngrams(filtered_toks9, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_bIRT = (pd.Series(nltk.ngrams(filtered_toks9, 3)).value_counts())[:20]
    
    #Visualize bottom 25% bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_bIRT.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Bottom 25 Pct Tweet Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize tri-grams:
    trigrams_bIRT.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Bottom 25 Pct Tweet Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    ####################################################################################################################################################
    #Search for differences within the 4 lists:
    
    #Words unique to the list of common terms across all tweets:
    allTweets_unique = []
    
    for word in topWords: #for each word in the list of top 20 common terms, find words unique to this list
        if word not in topWords_IRT25:
            if word not in topWords_IRTb25:
                if word not in topWords_mid_IRT:
                    allTweets_unique.append(word)
                        
    print("The words unique to the list for all tweets:")                
    print(allTweets_unique)
    
        
    
    
    #Words unique to the list of common terms across top 25% of IRT tweets:
    IRT25_unique = []
    
    for word in topWords_IRT25: #for each word in the list of top 20 common terms for top 25% of IRT tweets, find words unique to this list
        if word not in topWords:
            if word not in topWords_IRTb25:
                if word not in topWords_mid_IRT:
                    IRT25_unique.append(word)
                        
    print("The words unique to the list for top 25 pct of tweets:")                
    print(IRT25_unique)
    

    #Words unique to the list of common terms across median 50% of IRT tweets:
    IRTm50_unique = []
    
    for word in topWords_mid_IRT: #for each word in the list of top 20 common terms for median 50% of OT tweets, find words unique to this list
        if word not in topWords_IRT25:
            if word not in topWords:
                if word not in topWords_IRTb25:
                    IRTm50_unique.append(word)
                        
    print("The words unique to the list for median 50 pct of tweets:")                
    print(IRTm50_unique)
    
    
    #Words unique to the list of common terms across bottom 25% of IRT tweets:
    IRTb25_unique = []
    
    for word in topWords_IRTb25: #for each word in the list of top 20 common terms for bottom 25% of IRT tweets, find words unique to this list
        if word not in topWords_IRT25:
            if word not in topWords:
                if word not in topWords_mid_IRT:
                    IRTb25_unique.append(word)
                        
    print("The words unique to the list for bottom 25 pct of tweets:")                
    print(IRTb25_unique)
    


##############################################################################################################################################################
##############################################################################################################################################################
#In case there aren't enough IRT tweets for analysis:
elif propIRT < minThresh:
    print("%s doesn't seem to have enough IRT tweets for analysis. All tweets will be used, these are mostly official" % company_name)
    
    #To make things easier for myself,
    official_tweets = company_tweets.copy()
    
    #Examine first 5 tweets before any alterations are made
    #print(official_tweets["Content"].head(5))
    
    #Remove/replace 'smart' apostrophes and quotation marks with standard keyboard equivalents:
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"’", "'") #replace closing smart apostrophes with regular apostrophe
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"‘", "'") #replace opening smart apostrophes with regular apostrophe
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"“", "\"") #replace opening smart quotes with regular quotes
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"”", "\"") #replace closing smart quotes with regular quotes
    
    #Examine tweets after removing/replacing 'smart' apostrophes and quotes:
    #print(official_tweets["Content"].head(5))
    
    #Remove apostrophes followed by 's' and replace with nothing (Disney's becomes Disney):
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"'s", "")
    
    #Remove remaining apostrophes and replace with nothing (convert I'm to Im and such):
    official_tweets["Content"] = official_tweets["Content"].str.replace(r"'", "")    
    
    #All tweets: Remove links, @username mentions, non alphanumeric characters, replace @ with at, convert all text to lowercase
    official_tweets2 = standardize_text(official_tweets, "Content")
    
    #Removing tweets that weren't originally in English or 'und'
    English_official = official_tweets2[official_tweets2["Language"] == "en"]
    und_official = official_tweets2[official_tweets2["Language"] == "und"]
    official_clean = pd.concat([English_official, und_official], axis=0)
    
    #Removing rows with no text left inside them
    #filter3 = English_official["Content"] != ""
    #official_clean = English_official[filter3]
    
    #Create tweet and word tokens:
    official_clean["tokens"] = official_clean["Content"].apply(tokenizer.tokenize)
    
    ##Perform same analysis over top 25% of all tweets only: (in this case, mostly official)
    #Specify the number of rows to take as the top 25% of all tweets:
    #nr = int(round(0.25 * len(official_clean)))
    nr = math.ceil(0.25 * len(official_clean))
    
    #Sort the data based on number of likes:
    official_clean = official_clean.sort_values(by = "Number of Likes", ascending=False)    
    
    #Extract the top 25% of all tweets from the sorted data:
    top_official = official_clean.head(nr).copy()
    
    #Inspect the data more thoroughly
    all_words_topOT = [word for tokens in top_official["tokens"] for word in tokens]
    topOT_tweet_lengths = [len(tokens) for tokens in top_official["tokens"]]
    VOCAB_topOT = sorted(list(set(all_words_topOT)))
    #print(len(all_words_topOT))
    #print(len(VOCAB_topOT))
    print(company_name)
    print("Total number of top 25 pct tweets: %s" % len(top_official["Content"]))
    print("Top 25 percent of tweets have %s words total, with a vocabulary size of %s" % (len(all_words_topOT), len(VOCAB_topOT)))
    print("Max top tweet length is %s" % max(topOT_tweet_lengths))
    print("Average top tweet length is %s" % np.mean(topOT_tweet_lengths))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("Top 25 Pct of %s Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(topOT_tweet_lengths, bins=range(max(topOT_tweet_lengths)))
    plt.xticks(range(max(topOT_tweet_lengths)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks3 = []
    
    #Filter out the stop words:
    for w in top_official["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks3.append(j)
    
    #Examine the word frequencies
    fdist3 = FreqDist(filtered_toks3)
    topWords_OT25 = [v for v, k in fdist3.most_common(20)]
    print(fdist3.most_common(20))    
    
    #Examine top 25% n-grams:
    bigrams_OT25 = (pd.Series(nltk.ngrams(filtered_toks3, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_OT25 = (pd.Series(nltk.ngrams(filtered_toks3, 3)).value_counts())[:20]
    
    #Visualize top 25% bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_OT25.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Top 25 Pct Official Tweet Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize top 25% tri-grams:
    trigrams_OT25.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Top 25 Pct Official Tweet Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    
    ##Perform same analysis over median 50% of all company tweets: (in this case, mostly official)
    #Grab the median 50% of all company tweets:
    print("Across all tweets, there are %i tweets" % len(official_clean))
    print("In the top 25 pct there are %i tweets" % len(top_official))
    desired_num = int(round((0.5 * len(official_clean))))
    desired_num2 = desired_num - 1
    print("We should get %i tweets for middle 50 pct" % desired_num2)
    nr2 = nr + 1
    endPoint = int(nr + desired_num)
    #Ensure that we're grabbing correct number of tweets for median 50%:
    if len(official_clean) % 4 == 0:
        endPoint = endPoint + 1
    midTweets_OT = official_clean.iloc[nr2:endPoint]
    print("And we got %i tweets" % len(midTweets_OT))
    
    #Examine the middle 50% of data more thoroughly
    all_words_mid_OT = [word for tokens in midTweets_OT["tokens"] for word in tokens]
    tweet_lengths_mid_OT = [len(tokens) for tokens in midTweets_OT["tokens"]]
    VOCAB_mid_OT = sorted(list(set(all_words_mid_OT)))
    print(company_name)
    print("Total number of median tweets: %s" % len(midTweets_OT["Content"]))
    print("Median tweets have %s words total, with a vocabulary size of %s" % (len(all_words_mid_OT), len(VOCAB_mid_OT)))
    print("Max tweet length is %s" % max(tweet_lengths_mid_OT))
    print("Average tweet length is %s" % np.mean(tweet_lengths_mid_OT))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("Median 50 Pct %s Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(tweet_lengths_mid_OT, bins=range(max(tweet_lengths_mid_OT)))
    plt.xticks(range(max(tweet_lengths_mid_OT)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks4 = []
    
    #Filter out the stop words:
    for w in midTweets_OT["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks4.append(j)
    
    #Examine the word frequencies
    fdist4 = FreqDist(filtered_toks4)
    topWords_mid_OT = [v for v, k in fdist4.most_common(20)]
    print(fdist4.most_common(20))   
    
    #Examine median 50% n-grams:
    bigrams_OT50 = (pd.Series(nltk.ngrams(filtered_toks4, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_OT50 = (pd.Series(nltk.ngrams(filtered_toks4, 3)).value_counts())[:20]
    
    #Visualize median 50% bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_OT50.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Median 50 Pct Tweet Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize tri-grams:
    trigrams_OT50.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Median 50 Pct Tweet Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    
    ##Perform same analysis over bottom 25% of all tweets: (in this case, mostly official)
    #Grab the bottom 25% performing official tweets
    bottom_official = official_clean.tail(nr).copy()
    #print(bottom_official.shape)
    
    #Inspect the data more thoroughly
    all_words_bOT = [word for tokens in bottom_official["tokens"] for word in tokens]
    bOT_tweet_lengths = [len(tokens) for tokens in bottom_official["tokens"]]
    VOCAB_bOT = sorted(list(set(all_words_bOT)))
    #print(len(all_words_topOT))
    #print(len(VOCAB_topOT))
    print(company_name)
    print("Total number of bottom 25 pct tweets: %s" % len(bottom_official["Content"]))
    print("Bottom 25 pct of tweets have %s words total, with a vocabulary size of %s" % (len(all_words_bOT), len(VOCAB_bOT)))
    print("Max bottom tweet length is %s" % max(bOT_tweet_lengths))
    print("Average bottom tweet length is %s" % np.mean(bOT_tweet_lengths))
    
    #import matplotlib.pyplot as plt
    plt.rcdefaults()
    fig = plt.figure(figsize=(10, 10))
    plt.title("Bottom 25 Pct of %s Tweets" % company_name)
    plt.xlabel('Tweet Word Count')
    plt.ylabel('Number of Tweets')
    plt.hist(bOT_tweet_lengths, bins=range(max(bOT_tweet_lengths)))
    plt.xticks(range(max(bOT_tweet_lengths)))
    plt.xticks(rotation=90, ha='right')    
    plt.show()
    
    #Remove stop words from the data:
    filtered_toks5 = []
    
    #Filter out the stop words:
    for w in bottom_official["tokens"]: #tweet tokens
        for j in w: #word tokens within each tweet
            if j not in stop_words:
                filtered_toks5.append(j)
    
    #Examine the word frequencies
    #from nltk.probability import FreqDist
    fdist5 = FreqDist(filtered_toks5)
    topWords_OTb25 = [v for v, k in fdist5.most_common(20)]
    print(fdist5.most_common(20))    

    #Examine bottom 25% n-grams:
    bigrams_bOT = (pd.Series(nltk.ngrams(filtered_toks5, 2)).value_counts())[:20]
    
    #Tri-grams:
    trigrams_bOT = (pd.Series(nltk.ngrams(filtered_toks5, 3)).value_counts())[:20]
    
    #Visualize bottom 25% bi-grams:
    plt.rc('ytick', labelsize=SMALL_SIZE)
    bigrams_bOT.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Bottom 25 Pct Tweet Bi-grams" % company_name)
    plt.ylabel("Bi-gram")
    plt.xlabel("Number of Occurences")
    plt.show()
    
    #Visualize tri-grams:
    trigrams_bOT.sort_values().plot.barh(color='blue', width=0.9, figsize=(12,8))
    plt.title("20 Most Common %s Bottom 25 Pct Tweet Tri-grams" % company_name)
    plt.ylabel("Tri-gram")
    plt.xlabel("Number of Occurences")
    plt.show()

#######################################################################################################################################
    #Search for differences within the 4 lists:
    
    #Words unique to the list of common terms across all tweets:
    allTweets_unique = []
    
    for word in topWords: #for each word in the list of top 20 common terms, find words unique to this list
        if word not in topWords_OT25:
            if word not in topWords_OTb25:
                if word not in topWords_mid_OT:
                    allTweets_unique.append(word)
                        
    print("The words unique to the list for all tweets:")                
    print(allTweets_unique)
    
        
    
    
    #Words unique to the list of common terms across top 25% of tweets:
    OT25_unique = []
    
    for word in topWords_OT25: #for each word in the list of top 20 common terms for top 25% of tweets, find words unique to this list
        if word not in topWords:
            if word not in topWords_OTb25:
                if word not in topWords_mid_OT:
                    OT25_unique.append(word)
                        
    print("The words unique to the list for top 25 pct of tweets:")                
    print(OT25_unique)
    

    #Words unique to the list of common terms across median 50% of tweets:
    OTm50_unique = []
    
    for word in topWords_mid_OT: #for each word in the list of top 20 common terms for median 50% of tweets, find words unique to this list
        if word not in topWords_OT25:
            if word not in topWords:
                if word not in topWords_OTb25:
                    OTm50_unique.append(word)
                        
    print("The words unique to the list for median 50 pct of tweets:")                
    print(OTm50_unique)
    
    
    #Words unique to the list of common terms across bottom 25% of tweets:
    OTb25_unique = []
    
    for word in topWords_OTb25: #for each word in the list of top 20 common terms for bottom 25% of tweets, find words unique to this list
        if word not in topWords_OT25:
            if word not in topWords:
                if word not in topWords_mid_OT:
                    OTb25_unique.append(word)
                        
    print("The words unique to the list for bottom 25 pct of tweets:")                
    print(OTb25_unique)



