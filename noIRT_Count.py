#In this file, I plan to address the issue that some datasets don't run well through 'Generic_Count.py'
#For example, Disney, Samsung, and Toyota seem to not really have an IRT tweet category
#Overall, this is program: 12

import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras



#Read in the data: Below line of code will need to be reconfigured for your filepath
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Disney_Dec_1_2020.xlsx')
company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]
print(company_data.shape)

#Remove retweets from the company account, as they aren't technically company account tweets
patternDel = "^RT @"
filter1 = company_data["Content"].str.contains(patternDel)
company_tweets = company_data[~filter1].copy()
print(company_tweets.shape)


#Perform standardization on the textual contents of the company's tweets:
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
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

######################################################################################################################################################################################################################
#Perform same analysis on top 25% of company tweets only:

#Determine number of tweets that constitute 25% of data:
nr = int(round(0.25 * len(cleanGlish_tweets)))
print("nr is %i" % nr)

#Sort tweets, in descending order, based on likes:
cleanGlish_tweets = cleanGlish_tweets.sort_values(by="Number of Likes", ascending=False)
print("cleanGlish_tweets has %i rows" % len(cleanGlish_tweets))

#Grab the top 25% performing tweets:
topTweets = cleanGlish_tweets.head(nr)
print(topTweets.shape)

#Examine the top 25% of tweets more thoroughly:
all_words_top = [word for tokens in topTweets["tokens"] for word in tokens]
tweet_lengths_top = [len(tokens) for tokens in topTweets["tokens"]]
VOCAB_top = sorted(list(set(all_words_top)))
print(company_name)
print("Top tweets have %s words total, with a vocabulary size of %s" % (len(all_words_top), len(VOCAB_top)))
print("Max tweet length is %s" % max(tweet_lengths_top))
print("Average tweet length is %s" % np.mean(tweet_lengths_top))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
plt.title("Top 25 Pct %s Tweets" % company_name)
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of Tweets')
plt.hist(tweet_lengths_top)
plt.show()

#Remove stop words from the data:
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words("english"))

filtered_toks2 = []

#Filter out the stop words:
for w in topTweets["tokens"]: #tweet tokens
    for j in w: #word tokens within each tweet
        if j not in stop_words:
            filtered_toks2.append(j)

#Examine the word frequencies
#from nltk.probability import FreqDist
fdist2 = FreqDist(filtered_toks2)
topWords_top = [v for v, k in fdist2.most_common(20)]
print(fdist2.most_common(20))

##########################################################################################################################################################################################################
#Perform same analysis over median 50% of company tweets:

#Grab the median 50% of company tweets:
print("Across all data, there are %i tweets" % len(cleanGlish_tweets))
print("In the top 25 pct there are %i tweets" % len(topTweets))
desired_num = int(round((0.5 * len(cleanGlish_tweets))))
print("We should get %i tweets for middle 50 pct" % desired_num)
nr2 = nr + 1
endPoint = int(nr + desired_num)
midTweets = cleanGlish_tweets.iloc[nr2:endPoint]
print("And we got %i tweets" % len(midTweets))

#Examine the middle 50% of data more thoroughly
all_words_mid = [word for tokens in midTweets["tokens"] for word in tokens]
tweet_lengths_mid = [len(tokens) for tokens in midTweets["tokens"]]
VOCAB_mid = sorted(list(set(all_words_mid)))
print(company_name)
print("Median tweets have %s words total, with a vocabulary size of %s" % (len(all_words_mid), len(VOCAB_mid)))
print("Max tweet length is %s" % max(tweet_lengths_mid))
print("Average tweet length is %s" % np.mean(tweet_lengths_mid))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
plt.title("Median 50 Pct %s Tweets" % company_name)
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of Tweets')
plt.hist(tweet_lengths_mid)
plt.show()

#Remove stop words from the data:
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words("english"))

filtered_toks3 = []

#Filter out the stop words:
for w in midTweets["tokens"]: #tweet tokens
    for j in w: #word tokens within each tweet
        if j not in stop_words:
            filtered_toks3.append(j)

#Examine the word frequencies
#from nltk.probability import FreqDist
fdist3 = FreqDist(filtered_toks3)
topWords_mid = [v for v, k in fdist3.most_common(20)]
print(fdist3.most_common(20))

####################################################################################################################################################################################################################
#Perform same analysis over bottom 25% of company tweets:

#Grab the bottom 25% of company tweets:
bottomTweets = cleanGlish_tweets.tail(nr)
print(bottomTweets.shape)

#Examine the bottom 25% of data more thoroughly
all_words_bot = [word for tokens in bottomTweets["tokens"] for word in tokens]
tweet_lengths_bot = [len(tokens) for tokens in bottomTweets["tokens"]]
VOCAB_bot = sorted(list(set(all_words_bot)))
print(company_name)
print("Bottom tweets have %s words total, with a vocabulary size of %s" % (len(all_words_bot), len(VOCAB_bot)))
print("Max tweet length is %s" % max(tweet_lengths_bot))
print("Average tweet length is %s" % np.mean(tweet_lengths_bot))

#import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
plt.title("Bottom 25 Pct %s Tweets" % company_name)
plt.xlabel('Tweet Word Count')
plt.ylabel('Number of Tweets')
plt.hist(tweet_lengths_bot)
plt.show()

#Remove stop words from the data:
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words("english"))

filtered_toks4 = []

#Filter out the stop words:
for w in bottomTweets["tokens"]: #tweet tokens
    for j in w: #word tokens within each tweet
        if j not in stop_words:
            filtered_toks4.append(j)

#Examine the word frequencies
#from nltk.probability import FreqDist
fdist4 = FreqDist(filtered_toks4)
topWords_bot = [v for v, k in fdist4.most_common(20)]
print(fdist4.most_common(20))

#####################################################################################################################################################################################################
#Find words that are uniquely common to each tweet category above:

#For the list of all tweets:
allTweets_uniq = []

for word in topWords:
    if word not in topWords_top:
        if word not in topWords_mid:
            if word not in topWords_bot:
                allTweets_uniq.append(word)

print("The words uniquely common to all tweets:")
print(allTweets_uniq)

#For the list of top 25% tweets:
top25_uniq = []

for word in topWords_top:
    if word not in topWords:
        if word not in topWords_mid:
            if word not in topWords_bot:
                top25_uniq.append(word)

print("The words uniquely common to top 25 pct of tweets:")
print(top25_uniq)

#For the list of middle 50% tweets:
mid50_uniq = []

for word in topWords_mid:
    if word not in topWords:
        if word not in topWords_top:
            if word not in topWords_bot:
                mid50_uniq.append(word)

print("The words uniquely common to the median 50 pct:")
print(mid50_uniq)

#For the list of bottom 25% tweets:
bot25_uniq = []

for word in topWords_bot:
    if word not in topWords:
        if word not in topWords_top:
            if word not in topWords_mid:
                bot25_uniq.append(word)

print("The words uniquely common to the bottom 25 pct of tweets:")
print(bot25_uniq)








































































