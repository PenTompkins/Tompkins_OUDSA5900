# -*- coding: utf-8 -*-
#This file is being created to create final tweet topic assignments for companies having sufficient IRT tweets
#Thus, results from 'approximate_BTM_IRT.py' will be used here
#I'm beginning by copying/pasting 'BTM_model_OT.py' in here, then making relevant changes (filepaths, k values, extract IRT instead of OT, etc.)
#
#Overall, this is program: 71

import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from textblob import TextBlob, Word
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary
from biterm.btm import oBTM
import time

random.seed(1)
np.random.seed(2)

#Below line of code will need to be reconfigured to match the beginning of filepath to your data
path = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\'
commonEnd = '_Dec_1_2020.xlsx'
files = ['Amazon', 'BMW', 'CocaCola', 'Google', 'McDonalds', 'MercedesBenz', 'Microsoft']
respath = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Pictures\\BTMresults_IRT\\'
resEnding = '_BTMresults_IRT.xlsx'
#optKs = [6, 8, 10, 10, 6, 10, 11, 6, 8, 10] #optKs for BTM_model1.py
#optKs = [5, 4, 4, 16, 3, 6, 13, 15, 6, 19] #optKs for BTM_model2.py
#optKs = [17, 3, 3, 19, 11, 3, 4, 14] #optKs for BTM_model_OT.py
optKs = [3, 8, 3, 4, 6, 6, 20]


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



def clean_tokenize(df, text_field, stop_set):
    output = []
    for tweet in df[text_field]:
        clean_toks = []
        for tok in tweet:
            if tok not in stop_set:
                clean_toks.append(tok)
        output.append(clean_toks)
    return output


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



def perform_BTM(fpath, num_top):
    company_data = pd.read_excel(fpath)
    company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]
    print("\n\n\n\n\nBeginning BTM modeling for %s" % company_name)
    print("This is using %s topics" % num_top)
    
    #Remove retweets from the company account, as they aren't technically company account tweets
    patternDel = "^RT @"
    filter1 = company_data["Content"].str.contains(patternDel)
    company_tweets2 = company_data[~filter1].copy()
    
    ##Designate tweets as 'OT' or 'IRT' prior to removing more tweets or altering tweet contents
    #Perform initial separation based on "^@" regex:
    initIRT = [bool(re.search("^@", i)) for i in company_tweets2["Content"]]
    initOT = [not elem for elem in initIRT]
    #print(initOT)
    
    #Create IRT and OT variables in the data:
    company_tweets2["IRT"] = initIRT
    company_tweets2["OT"] = initOT
    
    
    #Fill in NAs under the 'In Reply To' field with "OT":
    company_tweets2["In Reply To"] = company_tweets2["In Reply To"].replace(np.nan, "OT", regex=True)
    #print(company_tweets["In Reply To"].head(5))
    
    #Call function to improve on initial OT vs. IRT splits:
    company_tweets3 = cleanSplit(company_tweets2, "Content", "IRT", "In Reply To", "Author", "OT")
    
    #For this version, extract IRT tweets only:
    company_tweets = company_tweets3[company_tweets3["IRT"] == True].copy()
    #print(company_tweets.shape)
    #print("Break")    
    
    #Create column such that original tweet contents aren't totally lost after textual pre-processing
    company_tweets["Content2"] = company_tweets["Content"]
    
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
    
    #Standardize the textual contents of tweets:
    textual_tweets = standardize_text(company_tweets, "Content")
    
    
    #Perform lemmatization on the textual contents of tweets:
    textual_tweets["Content"] = lem_with_postag(textual_tweets, "Content")
    #print(textual_tweets["Content"].head(5))
    
    #Removing tweets that weren't originally in English
    English_tweets = textual_tweets[textual_tweets["Language"] == "en"]
    
    #Removing rows with no text left inside them
    filter1 = English_tweets["Content"] != ""
    cleanGlish_tweets = English_tweets[filter1]
    
    
    #Remove stop words from the data:
    #from nltk.corpus import stopwords
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
    
    
                
    
    ##Vectorize the cleaned tweets
    #from sklearn.feature_extraction.text import CountVectorizer
    
    #Filter out stopwords here:
    vec = CountVectorizer(stop_words=stop_words)
    
    #Tokenize tweet contents:    
    tokenizer = RegexpTokenizer(r'\w+')
        
    cleanGlish_tweets["tokens"] = cleanGlish_tweets["Content"].apply(tokenizer.tokenize)
    cleanGlish_tweets["clean_tokens"] = clean_tokenize(cleanGlish_tweets, "tokens", stop_words)
    
    #print("Token differences:")
    #print(cleanGlish_tweets["tokens"].head(5))
    #print(cleanGlish_tweets["clean_tokens"].head(5))
    
    print("Before filtering out tweets with 3 words or less, cleanGlish has %s tweets" % len(cleanGlish_tweets["Content"]))
    #Filter out tweets with less than 3 words:
    cleanGlish_tweets["num_words"] = [len(token) for token in cleanGlish_tweets["clean_tokens"]]
    cleanGlish_tweets2 = cleanGlish_tweets[cleanGlish_tweets["num_words"] >= 3].copy()
    print("After filtering, cleanGlish2 has %s tweets" % len(cleanGlish_tweets2["Content"]))
    #Determine if filtering out tweets with less than 3 words after stop word removal makes a difference
    #cleanGlish_tweets["num_words2"] = [len(token) for token in cleanGlish_tweets["tokens"]]
    #cleanGlish_tweets3 = cleanGlish_tweets[cleanGlish_tweets["num_words2"] >=3].copy()
    #print("Originally, cleanGlish2 would have had %s tweets" % len(cleanGlish_tweets3["Content"]))
    
    #print("Breakpoint")
    
    X = vec.fit_transform(cleanGlish_tweets2["Content"]).toarray()
    #print("X looks like:")
    #print(X)
    
    #Get the vocabulary and the biterms from the tweets:
    #from biterm.utility import vec_to_biterms, topic_summuary
    
    vocab = np.array(vec.get_feature_names())
    #print("Vocab is:")
    #print(vocab)
    biterms = vec_to_biterms(X)
    #print("Biterms look like:")
    #print(biterms)
    #print("The non-zero parameter we're passing looks like:")
    #print(np.count_nonzero(X, axis=1))
    #print("The sum parameter we're passing in looks like:")
    #print(np.sum(X, axis=0))
    #print("Breakpoint")
    
    
    #Create a BTM and pass the biterms to train it:
    #from biterm.btm import oBTM
    #import time
    start_time = time.time()
    
    #random.seed(1)
    btm = oBTM(num_topics=num_top, V=vocab)
    topics = btm.fit_transform(biterms, iterations=100)
    end_time = time.time()
    run_time = end_time - start_time
    print("For %s..." % company_name)
    print("BTM took %s seconds to train" % run_time)
    
    #print("First parameter:")
    #print(btm.phi_wz.T)
    #print("Topics:")
    #print(topics)
    
    ##See if formatting data in the following manner allows pyLDAvis.prepare to work:
    
    
    #Visualize the topics:
    #If HTML(vis) doesn't work, look at following link
    #Link: https://jeriwieringa.com/2018/07/17/pyLDAviz-and-Mallet/
    #import pyLDAvis
    ##!This isn't working for some reason
    #vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0))
    #pyLDAvis.display(vis)
    #pyLDAvis.show(vis)
    #from IPython.core.display import HTML
    #HTML(vis)
    #cleanGlish_tweets2["topic"] = topics.argmax()
    cleanGlish_tweets2["topic"] = [topics[i].argmax() for i in range(len(cleanGlish_tweets2["Content"]))]
    
    
    #print("\nTweets and Topics:")
    #for i in range(len(cleanGlish_tweets2["Content"])):
        #print("{} (topic: {})".format(cleanGlish_tweets2.iloc[i, cleanGlish_tweets2.columns.get_loc("Content")], topics[i].argmax()))
    #    cleanGlish_tweets2.iat[i, 22] = topics[i].argmax()
    
    #Examine topic coherence scores:
    print("\nTopic Coherence:")
    topic_summuary(btm.phi_wz.T, X, vocab, 10)
    
    #Save the tweet topics:
    respath2 = respath + str(company_name) + resEnding
    cleanGlish_tweets2.to_excel(respath2)
    

###############################################################################
global_start = time.time()

for i in range(len(files)):
    file = files[i]
    k = optKs[i]
    filepath = path + str(file) + commonEnd
    perform_BTM(filepath, k)

global_end = time.time()
global_run = global_end - global_start
print("Performing BTM across all 7 IRT companies took %s seconds" % global_run)
