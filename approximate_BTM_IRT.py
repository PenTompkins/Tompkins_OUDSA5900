# -*- coding: utf-8 -*-
#This file is being created to perform 'approximate_BTM.py' process for IRT tweets of companies having enough IRT tweets for analysis
#
#Overall, this is program: 68

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
#Save resulting figures to this folder:
figpath = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Pictures\\BTMapproximations_IRT\\'


def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r".", "") #remove/replace periods w/ nothing. Should now count acronyms as one word
    df[text_field] = df[text_field].str.replace(r"&", "and") #replace ampersands with 'and'
    df[text_field] = df[text_field].str.replace(r"http\S+", "") #remove links and replace w/ nothing
    df[text_field] = df[text_field].str.replace(r"http", "") #ensure all links have been removed
    df[text_field] = df[text_field].str.replace(r"@\S+", "") #remove @username mentions and replace with nothing
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?#@\'\`\"\_]", " ")#Remove/replace anything that's not capital/lowercase letter, number, parentheses, comma, or any of the following symbols with a space
    df[text_field] = df[text_field].str.replace(r"@", "at") #replace any remaining '@' symbols with 'at'
    df[text_field] = df[text_field].str.lower() #convert all remaining text to lowercase
    df[text_field] = df[text_field].str.replace(r"\s+", " ")#replace 2+ whitespaces with single space
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



def get_change(current, previous):
    if current == previous:
        return 0
    try:
        return ((current - previous) / previous) * 100.0
    except ZeroDivisionError:
        return -1000



def lightningBTM(num_top, vocabulary, b_terms, x1):
    btm = oBTM(num_topics=num_top, V=vocabulary) #create the btm object
    start_time = time.time()
    for i in range(0, len(b_terms), 100): #process chunks of 200 texts
        biterms_chunk = b_terms[i:i + 100]
        btm.fit(biterms_chunk, iterations=10) #only 10 iterations in this version, instead of 50
    topics = btm.transform(b_terms)
    end_time = time.time()
    run_time = end_time - start_time
    print("For k = %s topics.." % num_top)
    print("BTM online took %s seconds to train" % run_time)
    #Examine topic coherence scores:
    print("\nTopic Coherence:")
    topic_summuary(btm.phi_wz.T, x1, vocabulary, 10)



def estimate_BTM(fpath):
    #Read in the data: Below line of code will need to be reconfigured for your filepath
    company_data = pd.read_excel(fpath)
    company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]
    print("\n\n\n\n\nBeginning BTM estimation process for %s" % company_name)
    
    #Remove retweets from the company account, as they aren't technically company account tweets
    patternDel = "^RT @"
    filter1 = company_data["Content"].str.contains(patternDel)
    company_tweets2 = company_data[~filter1].copy()
    
    ##Mark tweets as OT or IRT:
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
    
    company_tweets3 = cleanSplit(company_tweets2, "Content", "IRT", "In Reply To", "Author", "OT")
    
    #For this version, extract IRT tweets only:
    company_tweets = company_tweets3[company_tweets3["IRT"] == True].copy()
    #print(company_tweets.shape)
    #print("Break")
    
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
    
    #Standardize the textual contents of tweets
    textual_tweets = standardize_text(company_tweets, "Content")
    
    #Lemmatize the textual contents of tweets:
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
    
    #from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    
    
    
    #from nltk.tokenize import RegexpTokenizer
        
    tokenizer = RegexpTokenizer(r'\w+')
        
    cleanGlish_tweets["tokens"] = cleanGlish_tweets["Content"].apply(tokenizer.tokenize)
    cleanGlish_tweets["clean_tokens"] = clean_tokenize(cleanGlish_tweets, "tokens", stop_words)
    
    #Filter out tweets with less than 3 words:
    cleanGlish_tweets["num_words"] = [len(token) for token in cleanGlish_tweets["clean_tokens"]]
    cleanGlish_tweets2 = cleanGlish_tweets[cleanGlish_tweets["num_words"] >= 3].copy()
    
    #Extract the remaining textual contents of tweets:
    cleanGlish_tweets2["clean_tokens"] = [" ".join(tok) for tok in cleanGlish_tweets2["clean_tokens"].values]
    #print(cleanGlish_tweets2["clean_tokens"].head(5))
    #print("Break")
    clean_tweets = cleanGlish_tweets2["clean_tokens"]
    x = vectorizer.fit_transform(clean_tweets)
    
    
    #import matplotlib.pyplot as plt
    #from sklearn.cluster import KMeans
    #from sklearn.metrics import silhouette_score
    
    sum_squared_dists = []
    km_silh = []
    
    #Set range of k values desired to be attempted:
    K = range(3, 21)
    
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
    #plt.show()
    figpath2 = figpath + str(company_name) + 'elbow.png'
    plt.savefig(figpath2)
    
    #######################################################################
    #See if silhouette scores are better for determining optimal k
    
    plt.figure(figsize=(7,4))
    plt.title("%s Silhouette Scores" % company_name)
    plt.scatter(x=[i for i in range(3,21)],y=km_silh,s=150,edgecolor='k')
    plt.grid(True)
    plt.xlabel("Number of clusters",fontsize=6)
    plt.ylabel("Silhouette score",fontsize=6)
    plt.xticks([i for i in range(3,21)],fontsize=8)
    plt.yticks(fontsize=8)
    #plt.show()
    figpath3 = figpath + str(company_name) + 'silhouetteScores.png'
    plt.savefig(figpath3)
    
                
    
    
    print("\nSilhouette scores:")
    for val in km_silh:
        print(val)
    
    
    #Calculate percent changes:
    changes = [0]
    for i in range(len(km_silh) - 1):
        j = i + 1
        change = get_change(km_silh[j], km_silh[i])
        changes.append(change)
    
    #Examine percent changes:
    print("\nPercent changes:")
    for val in changes:
        print(val)
        
    #Determine which k values are suitable for testing:
    potential_k = []
    
    for i in range(len(changes)):
        if changes[i] < 1 and i != 0: #if the silhouette score decreased, or only increased by less than 1% (and it's not the first obs, which always has 0% increase)
            k = i + 1 # + 1 instead of 2 because we want to grab the value before the decrease (or insignificant increase)
            potential_k.append(k)
    
    print("For %s, the k values to be tested are:" % company_name)
    print(potential_k)
    print("However, in this version, all potential k's 2-20 will be tested")
    
    
    ##############################################################################################################################################
    #BTM online training:
    
    #Bring in the vectorizer to be used for BTM and supply pre-defined stopwords
    #from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(stop_words=stop_words)
    
    #Vectorize the tweets:
    X = vec.fit_transform(cleanGlish_tweets2["Content"]).toarray()
    
    #Get the vocabulary and the biterms from the tweets:
    #from biterm.utility import vec_to_biterms, topic_summuary
    
    vocab = np.array(vec.get_feature_names())
    biterms = vec_to_biterms(X)
    
    #Create a BTM and pass the biterms to train it, per k value in potential_k:
    #from biterm.btm import oBTM
    #import time
    best_k = []
    best_coherence = []
    
    
    
    total_start = time.time()
    #Train a BTM model on each potential k: (Not quite)
    #Train a BTM model on all potential k's 2-15:
    for k in K:
        lightningBTM(k, vocab, biterms, X)
        
    total_end = time.time()
    total_time = total_end - total_start
    print("For %s, total BTM estimation run-time was %s" % (company_name, total_time))
    #Unfortunately, I don't know how to save average topic coherence within speedyBTM
    #Results will have to be inspected manually to determine which value of k produced the best average topic coherence    
    
    
#############################################################################
global_start = time.time()

for file in files:
    filepath = path + file + commonEnd
    estimate_BTM(filepath)
    
global_end = time.time()
global_run = global_end - global_start
print("Whole shabang took %s seconds" % global_run)

