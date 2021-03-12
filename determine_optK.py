# -*- coding: utf-8 -*-
#This file is being created solely to use KMeans clustering and elbow method as a quick way of determining 'optimal' k in BTM topic modeling
#!For now, cleaning data same as in 'Topic_Sentiment.py', just to see if recommended k improves from before 
#I'm also now using silhouette scores to help determine 'optimal' (e.g. adequate) k
#
#Overall, this is program: 37

import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer

#Read in the data: Below line of code will need to be reconfigured for your filepath
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Disney_Dec_1_2020.xlsx')
company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]

#Remove retweets from the company account, as they aren't technically company account tweets
patternDel = "^RT @"
filter1 = company_data["Content"].str.contains(patternDel)
company_tweets = company_data[~filter1].copy()
company_tweets2 = company_data[~filter1].copy()

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
#No longer keep newline chars in text, replace double spaces with spaces, now keeping hashtag symbols themselves
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r".", "") #remove/replace periods w/ nothing. Should now count acronyms as one word
    df[text_field] = df[text_field].str.replace(r"&", "and") #replace ampersands with 'and'
    df[text_field] = df[text_field].str.replace(r"http\S+", "") #remove links and replace w/ nothing
    df[text_field] = df[text_field].str.replace(r"http", "") #ensure all links have been removed
    df[text_field] = df[text_field].str.replace(r"@\S+", "") #remove @username mentions and replace with nothing
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?#@\'\`\"\_]", " ")#Remove/replace anything that's not capital/lowercase letter, number, parentheses, comma, or any of the following symbols with a space
    df[text_field] = df[text_field].str.replace(r"@", "at") #replace any remaining '@' symbols with 'at'
    df[text_field] = df[text_field].str.lower() #convert all remaining text to lowercase
    #remove double spaces and replace with single space
    df[text_field] = df[text_field].str.replace(r"\s+", " ")
    return df

textual_tweets = standardize_text(company_tweets, "Content")

#Examine tweets after standardization has been performed:
#print(textual_tweets["Content"].head(5))

#Perform lemmatization on the textual contents of the tweets:
##! Code for this function derived from the following link: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
from textblob import TextBlob, Word

def lem_with_postag(df, text_field):
    tag_dict = {"J": 'a',
                "N": 'n',
                "V": 'v',
                "R": 'r'}
    output = []
    for tweet in df[text_field]:
        sent = TextBlob(tweet)
        words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]
        lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
        lemTweet = " ".join(lemmatized_list)
        output.append(lemTweet)
    return output

textual_tweets["Content"] = lem_with_postag(textual_tweets, "Content")
print(textual_tweets["Content"].head(5))

#Removing tweets that weren't originally in English
English_tweets = textual_tweets[textual_tweets["Language"] == "en"]

#Removing rows with no text left inside them
filter1 = English_tweets["Content"] != ""
cleanGlish_tweets = English_tweets[filter1]

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
#Lemmatization, for some reason, converts "us" to "u". Therefore, "u" should be added as a stopword as well (for lemmatized versions)
stop_words.add("u")

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=stop_words)

##Filter out tweets w/ less than 3 words after stop word removal:
def clean_tokenize(df, text_field, stop_set):
    output = []
    for tweet in df[text_field]:
        clean_toks = []
        for tok in tweet:
            if tok not in stop_set:
                clean_toks.append(tok)
        output.append(clean_toks)
    return output


from nltk.tokenize import RegexpTokenizer
    
tokenizer = RegexpTokenizer(r'\w+')
    
cleanGlish_tweets["tokens"] = cleanGlish_tweets["Content"].apply(tokenizer.tokenize)
cleanGlish_tweets["clean_tokens"] = clean_tokenize(cleanGlish_tweets, "tokens", stop_words)

#Filter out tweets with less than 3 words:
cleanGlish_tweets["num_words"] = [len(token) for token in cleanGlish_tweets["clean_tokens"]]
cleanGlish_tweets2 = cleanGlish_tweets[cleanGlish_tweets["num_words"] >= 3].copy()

#Extract the remaining textual contents of tweets:
#clean_tokens = cleanGlish_tweets2["clean_tokens"]
#Doesn't hurt to examine some of them:
print(cleanGlish_tweets2["clean_tokens"].head(5))
#print("Break")

#x = vectorizer.fit_transform(clean_tokens)
#x = vectorizer.fit_transform(cleanGlish_tweets2["clean_tokens"])
#x = vectorizer.fit_transform(str(clean_tokens))
#clean_tokens = [clean_tokens]
#x = vectorizer.fit_transform(clean_tokens)
#x = vectorizer.fit_transform(str(clean_tokens))
#x = vectorizer.fit_transform(cleanGlish_tweets2["clean_tokens"].str)
cleanGlish_tweets2["clean_tokens"] = [" ".join(tok) for tok in cleanGlish_tweets2["clean_tokens"].values]
print(cleanGlish_tweets2["clean_tokens"].head(5))
#print("Break")
clean_tweets = cleanGlish_tweets2["clean_tokens"]
x = vectorizer.fit_transform(clean_tweets)


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sum_squared_dists = []
km_silh = []
K = range(2, 21)

for k in K:
    km = KMeans(n_clusters=k, max_iter=200, n_init=10)
    km = km.fit(x)
    preds = km.predict(x)
    silh = silhouette_score(x, preds)
    sum_squared_dists.append(km.inertia_)
    km_silh.append(silh)
    
plt.plot(K, sum_squared_dists, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('%s Elbow Method for Optimal k' % company_name)
plt.show()

#######################################################################
#See if silhouette scores are better for determining optimal k
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#Actually, think this can all be done above as well

plt.figure(figsize=(7,4))
plt.title("The silhouette coefficient method \nfor determining number of clusters\n",fontsize=10)
plt.scatter(x=[i for i in range(2,21)],y=km_silh,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=8)
plt.ylabel("Silhouette score",fontsize=8)
plt.xticks([i for i in range(2,21)],fontsize=10)
plt.yticks(fontsize=10)
plt.show()

