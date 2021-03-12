#In this script, I plan to separate official vs. IRT tweets and re-do the wordcount process (from AmazonExploration2.py, and its related links)
#I may also try to group top 25% (or maybe 10) performing tweets, bottom 25% tweets, and search for differences in the two categories
#Could do that for all tweets, as well as within the official and IRT categories
#Overall, this is program: 8


import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras



#Read in the Amazon data: Below line of code will need to be reconfigured for your filepath
amazon_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Amazon_Dec_1_2020.xlsx')
#print(amazon_data.shape)

#Remove retweets from the Amazon account, as they aren't technically Amazon account tweets
patternDel = "^RT @"
filter1 = amazon_data["Content"].str.contains(patternDel)
#print(filter1)

amazon_tweets = amazon_data[~filter1].copy()
#print(amazon_tweets.shape)

#Separate Official vs. IRT tweets
pattern2 = "^@"
filter2 = amazon_tweets["Content"].str.contains(pattern2)
official_tweets = amazon_tweets[~filter2].copy()
IRT_tweets = amazon_tweets[filter2].copy()
#print(official_tweets.shape) #246 rows, just like in R
#print(IRT_tweets.shape) #2928 rows, just like in R


#Re-do processes in 'AmazonExploration2.py' for each tweet category:
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
#print(official_clean.shape) #238, thus the above processes removed 8 rows of data from official tweets

#Create tweet and word tokens:
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

official_clean["tokens"] = official_clean["Content"].apply(tokenizer.tokenize)


#Inspect the data more thoroughly
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

all_words_official = [word for tokens in official_clean["tokens"] for word in tokens]
OT_tweet_lengths = [len(tokens) for tokens in official_clean["tokens"]]
VOCAB_official = sorted(list(set(all_words_official)))
print("Before removing stopwords")
print("Official tweets have %s words total, with a vocabulary size of %s" % (len(all_words_official), len(VOCAB_official)))
print("Max official tweet length is %s" % max(OT_tweet_lengths))
print("Average official tweet length is %s" % np.mean(OT_tweet_lengths))

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10)) 
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of tweets')
plt.hist(OT_tweet_lengths)
plt.show()

#Remove stop words from the data:
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

filtered_toks = []

#Filter out the stop words:
for w in official_clean["tokens"]: #tweet tokens
    for j in w: #word tokens within each tweet
        if j not in stop_words:
            filtered_toks.append(j)

#Examine the word frequencies
from nltk.probability import FreqDist
fdist = FreqDist(filtered_toks)
print(fdist.most_common(20))
#For official tweets only, the top 20 most common words are:
#1) amazon, 86
#2) new, 30
#3A) see, 25
#3B) delivery, 25
#5) help, 23
#6) get, 22
#7) holiday, 21
#8A) us, 20
#8B) check, 20
#10A) today, 19
#10B), day, 19
#10C) people, 19
#13) customers, 18
#14) support, 16
#15A) family, 15
#15B) like, 15
#17A) thank, 14
#17B) learn, 14
#19A) items, 13
#19B) around, 13


##############################################################################################################################################################################
#Doing the same for IRT tweets:
#IRT tweets: Remove links, @username mentions, non alphanumeric characters, replace @ with at, convert all text to lowercase
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
print("Before removing stopwords")
print("IRT tweets have %s words total, with a vocabulary size of %s" % (len(all_words_IRT), len(VOCAB_IRT)))
print("Max IRT tweet length is %s" % max(IRT_tweet_lengths))
print("Average IRT tweet length is %s" % np.mean(IRT_tweet_lengths))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10)) 
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
print(fdist2.most_common(20))
#For IRT tweets only, the top 20 most common words are:
#1) us, 981
#2) send, 938
#3) love, 723
#4) details, 624
#5) deliveringsmiles, 468
#6) like, 466
#7) please, 434
#8) thanks, 405
#9) holiday, 399
#10) happy, 377
#11) season, 347
#12) hear, 322
#13) help, 273
#14) thank, 271
#15) surprise, 233
#16) hope, 231
#17) great, 227
#18) sharing, 186
#19) time, 179
#20) day, 174


###################################################################################################################################################################################
#Doing the same for the top 25% of Official tweets: (based on likes)
#official_tweets2 has already done most of the textual preprocessing
#Thus, I'm just going to remove non-english tweets from it and find top 25% official tweets from there

#Removing tweets that weren't originally in English
official_tweets3 = official_tweets2[official_tweets2["Language"] == "en"]

#Removing rows with no text left inside them
filter5 = official_tweets3["Content"] != ""
OT3_clean = official_tweets3[filter5]
#print(OT3_clean.shape) #238

#Specify the number of rows to take as the top 25% of official tweets:
nr = int(round(0.25 * len(OT3_clean)))
#print(nr)

#Sort the data based on number of likes:
OT3_clean = OT3_clean.sort_values(by = "Number of Likes", ascending=False)
#print(OT3_clean.shape)

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
print("Before removing stopwords")
print("Top 25 percent of official tweets have %s words total, with a vocabulary size of %s" % (len(all_words_topOT), len(VOCAB_topOT)))
print("Max top official tweet length is %s" % max(topOT_tweet_lengths))
print("Average top official tweet length is %s" % np.mean(topOT_tweet_lengths))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10)) 
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
print(fdist3.most_common(20))
#For the top 25% of official tweets, the 20 most common words are:
#1) amazon, 18
#2) holiday, 10
#3) shop, 8
#4) check, 8
#5) help, 7
#6) live, 6
#7) us, 6
#8) gifts, 6
#9) app, 6
#10) favorite, 6
#11) teamed, 5
#12) shopping, 5
#13) rufus, 4
#14) start, 4
#15) deliveringsmiles, 4
#16) beauty, 4
#17) today, 4
#18) new, 4
#19) deals, 4
#20) season, 4

#######################################################################################################################################################################################################################
#Doing the same for the bottom 25% of Amazon official tweets: (based on likes)
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
print("Before removing stopwords")
print("Bottom 25 percent of official tweets have %s words total, with a vocabulary size of %s" % (len(all_words_bOT), len(VOCAB_bOT)))
print("Max bottom official tweet length is %s" % max(bOT_tweet_lengths))
print("Average bottom official tweet length is %s" % np.mean(bOT_tweet_lengths))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10)) 
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
print(fdist4.most_common(20))
#Top 20 most common terms within the bottom 25% of official Amazon tweets:
#1) amazon, 28
#2) delivery, 11
#3) amp, 9
#4) thank, 8
#5) customers, 8
#6) ok, 8
#7) new, 7
#8) family, 7
#9) today, 6
#10) world, 6
#11) see, 6
#12) like, 6
#13) support, 6
#14) prime, 5
#15) get, 5
#16) year, 5
#17) learn, 5
#18) cart, 5
#19) primeday, 5
#20) employees, 5

#########################################################################################################################################################################################
#Re-doing above for the top 25% of IRT tweets: (based on likes)

#Removing tweets that weren't originally in English
IRT_tweets3 = IRT_tweets2[IRT_tweets2["Language"] == "en"]

#Removing rows with no text left inside them
filter6 = IRT_tweets3["Content"] != ""
IRT3_clean = IRT_tweets3[filter6]
#print(OT3_clean.shape) #238

#Specify the number of rows to take as the top 25% of IRT tweets:
nr2 = int(round(0.25 * len(IRT3_clean)))
#print(nr)

#Sort the data based on number of likes:
IRT3_clean = IRT3_clean.sort_values(by = "Number of Likes", ascending=False)
#print(OT3_clean.shape)

#Extract the top 25% of official tweets from the sorted data:
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
print("Before removing stopwords")
print("Top 25 percent of IRT tweets have %s words total, with a vocabulary size of %s" % (len(all_words_topIRT), len(VOCAB_topIRT)))
print("Max top IRT tweet length is %s" % max(topIRT_tweet_lengths))
print("Average top IRT tweet length is %s" % np.mean(topIRT_tweet_lengths))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10)) 
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
print(fdist5.most_common(20))
#The 20 most common terms within the top 25% of IRT tweets are:
#1) send, 384
#2) us, 323
#3) details, 243
#4) love, 223
#5) deliveringsmiles, 217
#6) holiday, 183
#7) season, 161
#8) like, 138
#9) please, 134
#10) surprise, 82
#11) help, 77
#12) happy, 74
#13) hear, 73
#14) thanks, 71
#15) thank, 62
#16) something, 58
#17) use, 57
#18) great, 56
#19) time, 54
#20) sounds, 54

################################################################################################################################################################################
#Doing the same for the bottom 25% of IRT tweets:

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
print("Before removing stopwords")
print("Bottom 25 percent of IRT tweets have %s words total, with a vocabulary size of %s" % (len(all_words_bIRT), len(VOCAB_bIRT)))
print("Max bottom IRT tweet length is %s" % max(bIRT_tweet_lengths))
print("Average bottom IRT tweet length is %s" % np.mean(bIRT_tweet_lengths))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10)) 
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of tweets')
plt.hist(bIRT_tweet_lengths)
plt.show() #Interesting that most of the bottom IRT tweets seem to be under 20 words or so

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
print(fdist6.most_common(20))
#The top 20 common words within the bottom 25% of IRT tweets are:
#1) us, 180
#2) love, 135
#3) thanks, 116
#4) happy, 107
#5) send, 106
#6) like, 93
#7) hear, 87
#8) hope, 76
#9) please, 75
#10) details, 73
#11) thank, 70
#12) sharing, 57
#13) glad, 56
#14) help, 55
#15) great, 54
#16) day, 52
#17) shout, 49...interesting that the first time I'm seeing this is when looking at bottom 25% of IRT
#18) enjoying, 42...same goes for these 3
#19) enjoy, 40
#20) could, 39











































































