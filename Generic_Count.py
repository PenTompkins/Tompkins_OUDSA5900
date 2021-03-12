#In this file, I plan to create a standard template which performs the full processes found in 'AmazonExploration2.py' as well as 'AmazonExploration3.py'
#Such that I can just change the target dataset at the beginning and perform same analysis without re-writing for each new dataset
#
#Overall, this is program: 10

import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras



#Read in the data: Below line of code will need to be reconfigured for your filepath
##!!This program loads data in twice, filepath around line 96 in code will need to be changed as well!
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Amazon_Dec_1_2020.xlsx')
company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]
#print(company_data.shape)

#Remove retweets from the company account, as they aren't technically company account tweets
patternDel = "^RT @"
filter1 = company_data["Content"].str.contains(patternDel)
company_tweets = company_data[~filter1].copy()
#print(company_tweets.shape)

#Perform standardization on the textual contents of the company's tweets:
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "") #remove links and replace w/ nothing
    df[text_field] = df[text_field].str.replace(r"http", "") #ensure all links have been removed
    df[text_field] = df[text_field].str.replace(r"@\S+", "") #remove @username mentions and replace with nothing
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")#Remove/replace anything that's not capital/lowercase letter, number, parentheses, comma, or any of the following symbols with a space
    df[text_field] = df[text_field].str.replace(r"@", "at") #replace any remaining '@' symbols with 'at'
    df[text_field] = df[text_field].str.lower() #convert all remaining text to lowercase
    return df

textual_tweets = standardize_text(company_tweets, "Content")

#Removing tweets that weren't originally in English
English_tweets = textual_tweets[textual_tweets["Language"] == "en"]

#Removing rows with no text left inside them
filter1 = English_tweets["Content"] != ""
cleanGlish_tweets = English_tweets[filter1]

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
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max tweet length is %s" % max(tweet_lengths))
print("Average tweet length is %s" % np.mean(tweet_lengths))

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
plt.title("All %s Tweets" % company_name)
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of tweets')
plt.hist(tweet_lengths)
plt.show()

#Remove stop words from the data:
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))


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

########################################################################################################################################################################
#Perform same process as above, but first split based on OT vs. IRT tweets:
#Official Tweets:
##!Change filepath below here as well!
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Amazon_Dec_1_2020.xlsx')
company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]
#print(company_data.shape)

#Remove retweets from the company account, as they aren't technically company account tweets
patternDel = "^RT @"
filter1 = company_data["Content"].str.contains(patternDel)
company_tweets = company_data[~filter1].copy()

#Split based on OT vs. IRT:
pattern2 = "^@"
filter2 = company_tweets["Content"].str.contains(pattern2)
official_tweets = company_tweets[~filter2].copy()
#print(company_tweets.shape)
#print(official_tweets.shape)
IRT_tweets = company_tweets[filter2].copy()
#print(IRT_tweets.shape)

#Ensure that no official tweets in reply to themselves, but starting with another @username mention, were counted as IRT tweets:
#Find all these tweets:
wrong_place = IRT_tweets[IRT_tweets["Author"] == IRT_tweets["In Reply To"]]
#print(wrong_place.shape)

#Add them to the official tweets dataset:
if len(wrong_place) != 0:
    official_tweets = official_tweets.append(wrong_place)
print(official_tweets.shape)

#Remove them from the IRT tweet dataset:
IRT_tweets = IRT_tweets[IRT_tweets["Author"] != IRT_tweets["In Reply To"]]
#print(IRT_tweets.shape)
        
    

#Re-do standardization:
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

#Official tweets: Remove links, @username mentions, non alphanumeric characters, replace @ with at, convert all text to lowercase
official_tweets2 = standardize_text(official_tweets, "Content")

#Removing tweets that weren't originally in English
English_official = official_tweets2[official_tweets2["Language"] == "en"]

#Removing rows with no text left inside them
filter3 = English_official["Content"] != ""
official_clean = English_official[filter3]

#Create tweet and word tokens:
official_clean["tokens"] = official_clean["Content"].apply(tokenizer.tokenize)


#Inspect the data more thoroughly
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical

all_words_official = [word for tokens in official_clean["tokens"] for word in tokens]
OT_tweet_lengths = [len(tokens) for tokens in official_clean["tokens"]]
VOCAB_official = sorted(list(set(all_words_official)))
print(company_name)
print("Official tweets have %s words total, with a vocabulary size of %s" % (len(all_words_official), len(VOCAB_official)))
print("Max official tweet length is %s" % max(OT_tweet_lengths))
print("Average official tweet length is %s" % np.mean(OT_tweet_lengths))

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
plt.title("All %s Official Tweets" % company_name)
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of tweets')
plt.hist(OT_tweet_lengths)
plt.show()

#Remove stop words from the data:
filtered_toks = []

#Filter out the stop words:
for w in official_clean["tokens"]: #tweet tokens
    for j in w: #word tokens within each tweet
        if j not in stop_words:
            filtered_toks.append(j)

#Examine the word frequencies
from nltk.probability import FreqDist
fdist = FreqDist(filtered_toks)
topWords_official = [v for v, k in fdist.most_common(20)]
print(fdist.most_common(20))


#################################################################################################################################################
#IRT Tweets: Remove links, @username mentions, non alphanumeric characters, replace @ with at, convert all text to lowercase
print(IRT_tweets.shape)
IRT_tweets2 = standardize_text(IRT_tweets, "Content")

#Removing tweets that weren't originally in English
English_IRT = IRT_tweets2[IRT_tweets2["Language"] == "en"]

#Removing rows with no text left inside them
filter4 = English_IRT["Content"] != ""
IRT_clean = English_IRT[filter4]
#print(IRT_clean.shape) #2753, thus above processes removed 175 rows of data from the IRT category

#Create tweet and word tokens:
#from nltk.tokenize import RegexpTokenizer

#tokenizer = RegexpTokenizer(r'\w+')

IRT_clean["tokens"] = IRT_clean["Content"].apply(tokenizer.tokenize)


#Inspect the data more thoroughly
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical

all_words_IRT = [word for tokens in IRT_clean["tokens"] for word in tokens]
IRT_tweet_lengths = [len(tokens) for tokens in IRT_clean["tokens"]]
VOCAB_IRT = sorted(list(set(all_words_IRT)))
print(company_name)
print("IRT tweets have %s words total, with a vocabulary size of %s" % (len(all_words_IRT), len(VOCAB_IRT)))
print("Max IRT tweet length is %s" % max(IRT_tweet_lengths))
print("Average IRT tweet length is %s" % np.mean(IRT_tweet_lengths))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
plt.title("All %s IRT Tweets" % company_name)
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of tweets')
plt.hist(IRT_tweet_lengths)
plt.show()

#Remove stop words from the data:
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words("english"))

filtered_toks2 = []

#Filter out the stop words:
for w in IRT_clean["tokens"]: #tweet tokens
    for j in w: #word tokens within each tweet
        if j not in stop_words:
            filtered_toks2.append(j)

#Examine the word frequencies
#from nltk.probability import FreqDist
fdist2 = FreqDist(filtered_toks2)
topWords_IRT = [v for v, k in fdist2.most_common(20)]
print(fdist2.most_common(20))

#############################################################################################################################################################
#Examine top 25% of company's official tweets (based on likes):
#official_tweets2 has already done most of the textual preprocessing
#Thus, I'm just going to remove non-english tweets from it and find top 25% official tweets from there

#Removing tweets that weren't originally in English
official_tweets3 = official_tweets2[official_tweets2["Language"] == "en"]

#Removing rows with no text left inside them
filter5 = official_tweets3["Content"] != ""
OT3_clean = official_tweets3[filter5]
#print(OT3_clean.shape)

#Specify the number of rows to take as the top 25% of official tweets:
nr = int(round(0.25 * len(OT3_clean)))
#print(nr)

#Sort the data based on number of likes:
OT3_clean = OT3_clean.sort_values(by = "Number of Likes", ascending=False)
#print(OT3_clean["Number of Likes"])
#print(OT3_clean["Number of Likes"].head(5))
#print(OT3_clean["Number of Likes"].tail(5))

#Extract the top 25% of official tweets from the sorted data:
top_official = OT3_clean.head(nr).copy()
#print(top_official.shape)

#Create tokens:
top_official["tokens"] = top_official["Content"].apply(tokenizer.tokenize)

#Inspect the data more thoroughly
all_words_topOT = [word for tokens in top_official["tokens"] for word in tokens]
topOT_tweet_lengths = [len(tokens) for tokens in top_official["tokens"]]
VOCAB_topOT = sorted(list(set(all_words_topOT)))
#print(len(all_words_topOT))
#print(len(VOCAB_topOT))
print(company_name)
print("Top 25 percent of official tweets have %s words total, with a vocabulary size of %s" % (len(all_words_topOT), len(VOCAB_topOT)))
print("Max top official tweet length is %s" % max(topOT_tweet_lengths))
print("Average top official tweet length is %s" % np.mean(topOT_tweet_lengths))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
plt.title("Top 25 Pct of %s Official Tweets" % company_name)
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of tweets')
plt.hist(topOT_tweet_lengths)
plt.show()

#Remove stop words from the data:
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words("english"))

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

###################################################################################################################################################################################################
#Examine bottom 25% of company's official tweets (based on likes):
#Grab the bottom 25% performing official tweets
bottom_official = OT3_clean.tail(nr).copy()
#print(bottom_official.shape)

#Create tokens:
bottom_official["tokens"] = bottom_official["Content"].apply(tokenizer.tokenize)

#Inspect the data more thoroughly
all_words_bOT = [word for tokens in bottom_official["tokens"] for word in tokens]
bOT_tweet_lengths = [len(tokens) for tokens in bottom_official["tokens"]]
VOCAB_bOT = sorted(list(set(all_words_bOT)))
#print(len(all_words_topOT))
#print(len(VOCAB_topOT))
print(company_name)
print("Bottom 25 percent of official tweets have %s words total, with a vocabulary size of %s" % (len(all_words_bOT), len(VOCAB_bOT)))
print("Max bottom official tweet length is %s" % max(bOT_tweet_lengths))
print("Average bottom official tweet length is %s" % np.mean(bOT_tweet_lengths))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
plt.title("Bottom 25 Pct of %s Official Tweets" % company_name)
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of tweets')
plt.hist(bOT_tweet_lengths)
plt.show()

#Remove stop words from the data:
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words("english"))

filtered_toks4 = []

#Filter out the stop words:
for w in bottom_official["tokens"]: #tweet tokens
    for j in w: #word tokens within each tweet
        if j not in stop_words:
            filtered_toks4.append(j)

#Examine the word frequencies
#from nltk.probability import FreqDist
fdist4 = FreqDist(filtered_toks4)
topWords_OTb25 = [v for v, k in fdist4.most_common(20)]
print(fdist4.most_common(20))

###########################################################################################################################################################################
#Perform analysis on top 25% of company's IRT tweets (based on likes):
#Removing tweets that weren't originally in English
IRT_tweets3 = IRT_tweets2[IRT_tweets2["Language"] == "en"]

#Removing rows with no text left inside them
filter6 = IRT_tweets3["Content"] != ""
IRT3_clean = IRT_tweets3[filter6]
#print(OT3_clean.shape) #238

#Specify the number of rows to take as the top 25% of IRT tweets:
nr2 = int(round(0.25 * len(IRT3_clean)))
#print(nr2)

#Sort the data based on number of likes:
IRT3_clean = IRT3_clean.sort_values(by = "Number of Likes", ascending=False)
#print(IRT3_clean["Number of Likes"].head(5))
#print(IRT3_clean["Number of Likes"].tail(5))
#print(OT3_clean.shape)

#Extract the top 25% of IRT tweets from the sorted data:
top_IRT = IRT3_clean.head(nr2).copy()
#print(top_official.shape)

#Create tokens:
top_IRT["tokens"] = top_IRT["Content"].apply(tokenizer.tokenize)

#Inspect the data more thoroughly
all_words_topIRT = [word for tokens in top_IRT["tokens"] for word in tokens]
topIRT_tweet_lengths = [len(tokens) for tokens in top_IRT["tokens"]]
VOCAB_topIRT = sorted(list(set(all_words_topIRT)))
#print(len(all_words_topOT))
#print(len(VOCAB_topOT))
print(company_name)
print("Top 25 percent of IRT tweets have %s words total, with a vocabulary size of %s" % (len(all_words_topIRT), len(VOCAB_topIRT)))
print("Max top IRT tweet length is %s" % max(topIRT_tweet_lengths))
print("Average top IRT tweet length is %s" % np.mean(topIRT_tweet_lengths))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
plt.title("Top 25 Pct of %s IRT Tweets" % company_name)
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of tweets')
plt.hist(topIRT_tweet_lengths)
plt.show()

#Remove stop words from the data:
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words("english"))

filtered_toks5 = []

#Filter out the stop words:
for w in top_IRT["tokens"]: #tweet tokens
    for j in w: #word tokens within each tweet
        if j not in stop_words:
            filtered_toks5.append(j)

#Examine the word frequencies
#from nltk.probability import FreqDist
fdist5 = FreqDist(filtered_toks5)
topWords_IRT25 = [v for v, k in fdist5.most_common(20)]
print(fdist5.most_common(20))

##################################################################################################################################################################################################
#Perform the same for the bottom 25% of company's IRT tweets (based on likes):
#Grab the bottom 25% of company IRT tweets:
bottom_IRT = IRT3_clean.tail(nr2).copy()
#print(bottom_official.shape)

#Create tokens:
bottom_IRT["tokens"] = bottom_IRT["Content"].apply(tokenizer.tokenize)

#Inspect the data more thoroughly
all_words_bIRT = [word for tokens in bottom_IRT["tokens"] for word in tokens]
bIRT_tweet_lengths = [len(tokens) for tokens in bottom_IRT["tokens"]]
VOCAB_bIRT = sorted(list(set(all_words_bIRT)))
#print(len(all_words_topOT))
#print(len(VOCAB_topOT))
print(company_name)
print("Bottom 25 percent of IRT tweets have %s words total, with a vocabulary size of %s" % (len(all_words_bIRT), len(VOCAB_bIRT)))
print("Max bottom IRT tweet length is %s" % max(bIRT_tweet_lengths))
print("Average bottom IRT tweet length is %s" % np.mean(bIRT_tweet_lengths))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
plt.title("Bottom 25 Pct of %s IRT Tweets" % company_name)
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of tweets')
plt.hist(bIRT_tweet_lengths)
plt.show()

#Remove stop words from the data:
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words("english"))

filtered_toks6 = []

#Filter out the stop words:
for w in bottom_IRT["tokens"]: #tweet tokens
    for j in w: #word tokens within each tweet
        if j not in stop_words:
            filtered_toks6.append(j)

#Examine the word frequencies
#from nltk.probability import FreqDist
fdist6 = FreqDist(filtered_toks6)
topWords_IRTb25 = [v for v, k in fdist6.most_common(20)]
print(fdist6.most_common(20))


#######################################################################################################################################################################################################
#Search for differences within the 7 lists:

#Words unique to the list of common terms across all tweets:
allTweets_unique = []

for word in topWords: #for each word in the list of top 20 common terms, find words unique to this list
    if word not in topWords_OT25:
        if word not in topWords_official:
            if word not in topWords_OTb25:
                if word not in topWords_IRT:
                    if word not in topWords_IRT25:
                        if word not in topWords_IRTb25:
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
                            top20_OTunique.append(word) #then it might be worth taking a look at
                    

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
                            IRTb25_unique.append(word)
                    
print("The words unique to the list for bottom 25 pct IRT tweets:")                
print(IRTb25_unique)


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
                        justInBest.append(word) #then let's take a look at this word

print("Words unique to the lists of top 25's:")
print(justInBest)

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
                        justInWorst.append(word) #then this word is uniquely common to poorly performing tweets
                        
print("Words unique to the list of bottom 25's:")
print(justInWorst)









































