#Second text analysis and exploration of Amazon tweets
#I used the following link to help out with this code: https://www.kdnuggets.com/2019/01/solve-90-nlp-problems-step-by-step-guide.html
#Initial attempt used this, code from here is likely present as well: https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
#Overall, this is program: 5

import sklearn
import keras
import nltk
import pandas as pd
import numpy as np
import re
import codecs

#Read in the Amazon data: Below line of code will need to be reconfigured for your filepath
amazon_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Amazon_Dec_1_2020.xlsx')
#print(amazon_data.shape)

#Remove retweets from the Amazon account, as they aren't technically Amazon account tweets
patternDel = "^RT @"
filter1 = amazon_data["Content"].str.contains(patternDel)
#print(filter1)

amazon_tweets = amazon_data[~filter1].copy()

#This time, I'm going to try to isolate the textual information as much as possible
#Thus, I plan to convert all text to lowercase, and lemmatize words to their common root
#Also, I plan to remove: twitter handles (@xyz), links, floating @s with at, non alphanumeric characters, and hopefully emojis

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ") #what exactly does this one do? I think this removes any non alphanumeric chars (including emojis)
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

textual_tweets = standardize_text(amazon_tweets, "Content")

#print(textual_tweets["Content"])
#Hard to see the contents from this

#I'm going to save to a csv to be able to examine performance manually
#textual_tweets.to_csv("textual_Amazon.csv")

#Seems to have removed hashtags, handles, links, emojis, and pretty much everything besides text
#However, it created a few empty rows of "Content," I'll need to remove those
#I'll also need to filter out any tweets that weren't in English

#Removing tweets that weren't originally in English
English_tweets = textual_tweets[textual_tweets["Language"] == "en"]

#Removing rows with no text left inside them
filter1 = English_tweets["Content"] != ""
cleanGlish_tweets = English_tweets[filter1]
#cleanGlish_tweets.to_csv("cleanGlish_tweets.csv")
#print(cleanGlish_tweets.iloc[8]["Content"])


#I suppose, for now, I'm not going to lemmatize
#Creating tokens:
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

cleanGlish_tweets["tokens"] = cleanGlish_tweets["Content"].apply(tokenizer.tokenize)
#print(cleanGlish_tweets["tokens"])
#cleanGlish_tweets.to_csv("cleanGlish_tweets.csv")

#Inspect the data more thoroughly
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

all_words = [word for tokens in cleanGlish_tweets["tokens"] for word in tokens]
tweet_lengths = [len(tokens) for tokens in cleanGlish_tweets["tokens"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max tweet length is %s" % max(tweet_lengths))
print("Average tweet length is %s" % np.mean(tweet_lengths))

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10)) 
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of tweets')
plt.hist(tweet_lengths)
plt.show()

#If I apply transformation, will the above graph look more like a normal distribution?
tweet_lengths2 = np.sqrt(tweet_lengths) #square root transformation, not bad
#tweet_lengths2 = [i ** (1/3) for i in tweet_lengths] #cube root transformation, probably a little worse than square root
#tweet_lengths2 = np.log(tweet_lengths) #log transformation, still worse than square root
fig = plt.figure(figsize=(10, 10)) 
plt.xlabel('Tweet Square Root Word Count')
plt.ylabel('Number of tweets')
plt.hist(tweet_lengths2)
plt.show()

#At this point, at least for now, I'm going to stop following the top link
#Now, I'll be re-doing the processes from AmazonExploration.py (second link)

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
print(fdist.most_common(20))
#Without adding any stop words, the top 20 most common occurring terms are:
#1) us, 1001
#2) send, 939
#3) love, 732
#4) details, 628
#5) like, 481
#5) deliveringsmiles, 481
#7) please, 437
#8) holiday, 420
#9) thanks, 412
#10) happy, 386
#11) season, 357
#12) hear, 323
#13) help, 296
#14) thank, 285...probably should group "thanks," "thank you," "thx," and anything like that
#15) surprise, 238
#16) hope, 232
#17) great, 231...seems like they probably say 'great day' and 'great time' a lot
#18) day, 193
#19) time, 188
#20) sharing, 187




























































