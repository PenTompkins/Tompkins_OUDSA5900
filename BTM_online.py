# -*- coding: utf-8 -*-
#This file is being created to perform online BTM training faster (hopefully) than the original version
#Data will be pre-processed in the same manner as in 'Topic_Sentiment.py', for more fair comparison
#
#Overall, this is program: 38

import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras
import random

#Set a seed before doing anything else
random.seed(1)
np.random.seed(2)

#Read in the data: Below line of code will need to be reconfigured for your filepath
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Google_Dec_1_2020.xlsx')
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

#Creating tokens:
#from nltk.tokenize import RegexpTokenizer

#tokenizer = RegexpTokenizer(r'\w+')

#cleanGlish_tweets["tokens"] = cleanGlish_tweets["Content"].apply(tokenizer.tokenize)

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

##!!This doesn't seem to be compatible with BTM!
#filtered_toks = []

#Filter out the stop words:
#for w in cleanGlish_tweets["tokens"]: #tweet tokens
#    for j in w: #word tokens within each tweet
#        if j not in stop_words:
#            filtered_toks.append(j)

##!Seems possible that I need to filter out tweets with less than 3 words remaining for below to work:
#from nltk.tokenize import RegexpTokenizer

#tokenizer = RegexpTokenizer(r'\w+')

#cleanGlish_tweets["tokens"] = cleanGlish_tweets["Content"].apply(tokenizer.tokenize)
#print("Before filtering out tweets with 3 words or less, cleanGlish has %s tweets" % len(cleanGlish_tweets["Content"]))
#Filter out tweets with less than 3 words:
#cleanGlish_tweets["num_words"] = [len(token) for token in cleanGlish_tweets["tokens"]]
#cleanGlish_tweets2 = cleanGlish_tweets[cleanGlish_tweets["num_words"] >= 3].copy()
#print("After filtering, cleanGlish2 has %s tweets" % len(cleanGlish_tweets2["Content"]))
#print("Breakpoint")

            

##Vectorize the cleaned tweets
from sklearn.feature_extraction.text import CountVectorizer

#Filter out stopwords here:
vec = CountVectorizer(stop_words=stop_words)
##Seems that a potential problem above is that I'm filtering out tweets w/ less than 3 words before stopword removal:
##Thus, making it possible that tweets with less than 3 counting words are being fed to the model
##!!I think I can supply my own set of stopwords above, rather than use CountVectorizer's pre-defined set
#print("Stop words:")
#for word in vec.stop_words:
#    print(word)

#Save CountVectorizer's set of stop words:
#stop_words = [word for word in vec.stop_words]
#print("Stop words variable:")
#for word in stop_words:
#    print(word)

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

print("Token differences:")
print(cleanGlish_tweets["tokens"].head(5))
print(cleanGlish_tweets["clean_tokens"].head(5))

print("Before filtering out tweets with 3 words or less, cleanGlish has %s tweets" % len(cleanGlish_tweets["Content"]))
#Filter out tweets with less than 3 words:
cleanGlish_tweets["num_words"] = [len(token) for token in cleanGlish_tweets["clean_tokens"]]
cleanGlish_tweets2 = cleanGlish_tweets[cleanGlish_tweets["num_words"] >= 3].copy()
print("After filtering, cleanGlish2 has %s tweets" % len(cleanGlish_tweets2["Content"]))
#Determine if filtering out tweets with less than 3 words after stop word removal makes a difference
cleanGlish_tweets["num_words2"] = [len(token) for token in cleanGlish_tweets["tokens"]]
cleanGlish_tweets3 = cleanGlish_tweets[cleanGlish_tweets["num_words2"] >=3].copy()
print("Originally, cleanGlish2 would have had %s tweets" % len(cleanGlish_tweets3["Content"]))

#print("Breakpoint")

X = vec.fit_transform(cleanGlish_tweets2["Content"]).toarray()
print("X looks like:")
print(X)

#Get the vocabulary and the biterms from the tweets:
from biterm.utility import vec_to_biterms, topic_summuary

vocab = np.array(vec.get_feature_names())
#print("Vocab is:")
#print(vocab)
biterms = vec_to_biterms(X)


#Create a BTM and pass the biterms to train it:
from biterm.btm import oBTM
import time
start_time = time.time()

btm = oBTM(num_topics=13, V=vocab)
##Online BTM training, link = https://pypi.org/project/biterm/
print("\nTrain Online BTM")
for i in range(0, len(biterms), 100): #process chunks of 200 texts
    biterms_chunk = biterms[i:i + 100]
    btm.fit(biterms_chunk, iterations=50)
topics = btm.transform(biterms)

end_time = time.time()
run_time = end_time - start_time
print("BTM online took %s seconds to train" % run_time)

print("\nTweets and Topics:")
for i in range(len(cleanGlish_tweets2["Content"])):
    print("{} (topic: {})".format(cleanGlish_tweets2.iloc[i, cleanGlish_tweets2.columns.get_loc("Content")], topics[i].argmax()))

#Examine topic coherence scores:
print("\nTopic Coherence:")
topic_summuary(btm.phi_wz.T, X, vocab, 10)