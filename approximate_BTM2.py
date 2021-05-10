# -*- coding: utf-8 -*-
#This file is being created to overlay average topic coherence scores on plots produced in 'approximate_BTM.py'
#Above file has been copied/pasted here, I'm adding in coherence score results and code to overlay those in the elbow and silhouette plots
#Otherwise, nothing else should be changed (except additional code to get coherence arrays input along with company data)
#
#Overall, this is program: 46

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
files = ['Amazon', 'BMW', 'CocaCola', 'Disney', 'Google', 'McDonalds', 'MercedesBenz', 'Microsoft', 'Samsung', 'Toyota']
##Temporarily changing below such that original results are preserved (originally didn't have 'Again' at the end)
figpath = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Pictures\\BTMapproximationsAgain\\'

##Adding in topic coherence vectors for each company:
Amazon_arr = [-78.06, -88.37, -90.84, -86.50, -90.99, -103.27, -98.62, -89.12, -93.11, -94.73, -103.60, -98.19, -107.02, -95.69, -97.51, -94.38, -98.47, -105.18, -98.99]
BMW_arr = [-64.44, -42.60, -39.63, -79.51, -54.37, -74.18, -78.48, -69.39, -78.03, -86.05, -76.48, -85.11, -84.81, -78.66, -89.28, -77.97, -80.40, -84.41, -83.89]
CocaCola_arr = [-35.87, -45.55, -44.12, -54.11, -53.52, -57.27, -65.61, -67.02, -67.96, -50.29, -58.02, -61.68, -61.89, -65.73, -62.07, -72.14, -62.88, -71.65, -73.24]
Disney_arr = [-107.59, -109.17, -115.10, -120.39, -114.00, -118.26, -117.24, -122.72, -119.09, -114.01, -112.03, -116.59, -115.73, -117.06, -109.22, -115.96, -116.32, -117.98, -115.89]
Google_arr = [-50.46, -34.66, -49.45, -59.34, -72.88, -64.63, -69.74, -75.18, -74.21, -54.80, -53.01, -70.30, -80.10, -72.79, -68.05, -72.67, -70.98, -70.44, -78.49]
McDonalds_arr = [-84.72, -79.73, -98.54, -100.74, -77.59, -88.08, -84.18, -97.57, -89.50, -95.81, -88.67, -104.81, -95.58, -94.38, -92.15, -101.70, -95.81, -91.57, -86.17]
Mercedes_arr = [-78.50, -91.08, -93.27, -95.95, -90.80, -93.06, -96.73, -94.31, -89.62, -90.37, -97.95, -83.42, -93.79, -87.70, -98.70, -87.49, -89.13, -97.41, -88.29]
Microsoft_arr = [-107.01, -112.41, -111.66, -111.70, -115.89, -113.69, -113.18, -113.87, -112.41, -114.14, -114.30, -114.90, -111.40, -109.64, -111.20, -109.89, -111.55, -113.75, -116.09]
Samsung_arr = [-116.44, -128.63, -125.57, -129.02, -124.76, -129.40, -126.77, -133.11, -131.20, -129.41, -134.89, -133.38, -128.93, -132.93, -134.47, -135.80, -133.08, -135.41, -134.35]
Toyota_arr = [-132.40, -128.97, -126.31, -127.21, -131.09, -126.80, -130.06, -121.94, -126.25, -122.89, -124.62, -129.00, -120.05, -120.02, -121.51, -122.73, -122.59, -119.99, -122.73]
all_arrs = []
all_arrs.append(Amazon_arr)
all_arrs.append(BMW_arr)
all_arrs.append(CocaCola_arr)
all_arrs.append(Disney_arr)
all_arrs.append(Google_arr)
all_arrs.append(McDonalds_arr)
all_arrs.append(Mercedes_arr)
all_arrs.append(Microsoft_arr)
all_arrs.append(Samsung_arr)
all_arrs.append(Toyota_arr)

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



def estimate_BTM(fpath, arr):
    #Read in the data: Below line of code will need to be reconfigured for your filepath
    company_data = pd.read_excel(fpath)
    company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]
    print("\n\n\n\n\nBeginning BTM estimation process for %s" % company_name)
    
    #Remove retweets from the company account, as they aren't technically company account tweets
    patternDel = "^RT @"
    filter1 = company_data["Content"].str.contains(patternDel)
    company_tweets = company_data[~filter1].copy()
    #company_tweets2 = company_data[~filter1].copy()
    
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
    #def standardize_text(df, text_field):
    #    df[text_field] = df[text_field].str.replace(r".", "") #remove/replace periods w/ nothing. Should now count acronyms as one word
    #    df[text_field] = df[text_field].str.replace(r"&", "and") #replace ampersands with 'and'
    #    df[text_field] = df[text_field].str.replace(r"http\S+", "") #remove links and replace w/ nothing
    #    df[text_field] = df[text_field].str.replace(r"http", "") #ensure all links have been removed
    #    df[text_field] = df[text_field].str.replace(r"@\S+", "") #remove @username mentions and replace with nothing
    #    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?#@\'\`\"\_]", " ")#Remove/replace anything that's not capital/lowercase letter, number, parentheses, comma, or any of the following symbols with a space
    #    df[text_field] = df[text_field].str.replace(r"@", "at") #replace any remaining '@' symbols with 'at'
    #    df[text_field] = df[text_field].str.lower() #convert all remaining text to lowercase
    #    #remove double spaces and replace with single space
    #    df[text_field] = df[text_field].str.replace(r"\s+", " ")
    #    return df
    
    textual_tweets = standardize_text(company_tweets, "Content")
    
    #Examine tweets after standardization has been performed:
    #print(textual_tweets["Content"].head(5))
    
    #Perform lemmatization on the textual contents of the tweets:
    ##! Code for this function derived from the following link: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    #from textblob import TextBlob, Word
    
    #def lem_with_postag(df, text_field):
    #    tag_dict = {"J": 'a',
    #                "N": 'n',
    #                "V": 'v',
    #                "R": 'r'}
    #    output = []
    #    for tweet in df[text_field]:
    #        sent = TextBlob(tweet)
    #        words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]
    #        lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    #        lemTweet = " ".join(lemmatized_list)
    #        output.append(lemTweet)
    #    return output
    
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
    
    ##Filter out tweets w/ less than 3 words after stop word removal:
    #def clean_tokenize(df, text_field, stop_set):
    #    output = []
    #    for tweet in df[text_field]:
    #        clean_toks = []
    #        for tok in tweet:
    #            if tok not in stop_set:
    #                clean_toks.append(tok)
    #        output.append(clean_toks)
    #    return output
    
    
    #from nltk.tokenize import RegexpTokenizer
        
    tokenizer = RegexpTokenizer(r'\w+')
        
    cleanGlish_tweets["tokens"] = cleanGlish_tweets["Content"].apply(tokenizer.tokenize)
    cleanGlish_tweets["clean_tokens"] = clean_tokenize(cleanGlish_tweets, "tokens", stop_words)
    
    #Filter out tweets with less than 3 words:
    cleanGlish_tweets["num_words"] = [len(token) for token in cleanGlish_tweets["clean_tokens"]]
    cleanGlish_tweets2 = cleanGlish_tweets[cleanGlish_tweets["num_words"] >= 3].copy()
    
    #Extract the remaining textual contents of tweets:
    #clean_tokens = cleanGlish_tweets2["clean_tokens"]
    #Doesn't hurt to examine some of them:
    #print(cleanGlish_tweets2["clean_tokens"].head(5))
    #print("Break")
    
    #x = vectorizer.fit_transform(clean_tokens)
    #x = vectorizer.fit_transform(cleanGlish_tweets2["clean_tokens"])
    #x = vectorizer.fit_transform(str(clean_tokens))
    #clean_tokens = [clean_tokens]
    #x = vectorizer.fit_transform(clean_tokens)
    #x = vectorizer.fit_transform(str(clean_tokens))
    #x = vectorizer.fit_transform(cleanGlish_tweets2["clean_tokens"].str)
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
    #Considering I have yet to see a best k greater than 13, I'm reducing K's range from (2, 21) to (2, 16)
    ##However, since this is lightning BTM, might as well revert back to K = range(2, 21)
    ##Might do that for an overnight run or something, reverting back to (2,16) for now
    ##Actually, might as well get this going
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
    #plt.show()
    for i in range(len(K)):
        label = "{:.2f}".format(arr[i])
        plt.annotate(label,
                     (K[i], sum_squared_dists[i]),
                     textcoords = "offset points",
                     xytext = (3, 5),
                     ha='center',
                     fontsize=5)    
    figpath2 = figpath + str(company_name) + 'elbow.png'
    plt.savefig(figpath2)
    
    #######################################################################
    #See if silhouette scores are better for determining optimal k
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler()
    #Actually, think this can all be done above as well
    
    plt.figure(figsize=(7,4))
    plt.title("%s Silhouette Scores" % company_name)
    plt.scatter(x=[i for i in range(2,21)],y=km_silh,s=150,edgecolor='k')
    plt.grid(True)
    plt.xlabel("Number of clusters",fontsize=6)
    plt.ylabel("Silhouette score",fontsize=6)
    plt.xticks([i for i in range(2,21)],fontsize=8)
    plt.yticks(fontsize=8)
    #plt.show()
    for i in range(len(K)):
        label = "{:.2f}".format(arr[i])
        plt.annotate(label,
                     (K[i], km_silh[i]),
                     textcoords = "offset points",
                     xytext = (0, 9),
                     ha='center',
                     fontsize=6)    
    figpath3 = figpath + str(company_name) + 'silhouetteScores.png'
    plt.savefig(figpath3)
    
    #plt.figure(figsize=(10,10))
    #plt.title("%s Silhouette Scores" % company_name)
    #plt.xlabel('k')
    #plt.ylabel('Silhouette Score')
    #plt.scatter(x=rangemax(K)), y = km_silh)
    #plt.scatter(x=[i for i in range(2, np.max(K))], y = km_silh)
    #figpath4 = figpath + str(company_name) + '_silhouetteScoresAgain.png'
    #plt.savefig(figpath4)
                
    
    
    print("\nSilhouette scores:")
    for val in km_silh:
        print(val)
    
    #Function to calculate percent change in silhouette scores
    #Code derived from: https://stackoverflow.com/questions/30926840/how-to-check-change-between-two-values-in-percent
    #def get_change(current, previous):
    #    if current == previous:
    #        return 0
    #    try:
    #        return ((current - previous) / previous) * 100.0
    #    except ZeroDivisionError:
    #        return -1000
    
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
    
    #Function to perform online BTM training
    #def speedyBTM(num_top, vocabulary, b_terms):
    #    btm = oBTM(num_topics=num_top, V=vocabulary) #create the btm object
    #    start_time = time.time()
    #    for i in range(0, len(b_terms), 100): #process chunks of 200 texts
    #        biterms_chunk = biterms[i:i + 100]
    #        btm.fit(biterms_chunk, iterations=50)
    #    topics = btm.transform(biterms)
    #    end_time = time.time()
    #    run_time = end_time - start_time
    #    print("For k = %s topics.." % num_top)
    #    print("BTM online took %s seconds to train" % run_time)
    #    #Examine topic coherence scores:
    #    print("\nTopic Coherence:")
    #    topic_summuary(btm.phi_wz.T, X, vocab, 10)
    
    
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
eye = 0 #to keep track of coherence arrays

for file in files:
    filepath = path + file + commonEnd
    el_arr = all_arrs[eye]
    estimate_BTM(filepath, el_arr)
    eye = eye + 1
    
global_end = time.time()
global_run = global_end - global_start
print("Whole shabang took %s seconds" % global_run)

