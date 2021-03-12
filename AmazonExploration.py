#Initial text analysis and exploration of Amazon tweets
#I used the following link to help out with this code: https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
#Overall, this is program: 3

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from scipy.stats import multivariate_normal as mvn
import nltk
import os
import random
import string

import os, sys, email,re

#Read in the Amazon data: Below line of code will need to be reconfigured for your filepath
amazon_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Amazon_Dec_1_2020.xlsx')
#print(amazon_data.shape)

#Remove retweets from the Amazon account, as they aren't technically Amazon account tweets
patternDel = "^RT @"
filter1 = amazon_data["Content"].str.contains(patternDel)
#print(filter1)

amazon_tweets = amazon_data[~filter1].copy()

#First, I'm going to try to see the most frequently occuring words throughout Amazon's tweets
#However, to do that, it would be more informative to first remove stop words
#nltk.download('stopwords')
#nltk.download('punkt')
stop_words = set(stopwords.words("english"))

#Perform word tokenization on Amazon's tweets
tweets = amazon_tweets["Content"]
amzn_word_toks = tweets.apply(word_tokenize)

#Remove the stopwords from it
filtered_toks = []

#print(amzn_word_toks)
#print(stop_words)
#for i in stop_words:
#    print(i)
#for i in amzn_word_toks:
#    for j in i:
#        print(j)

#for w in amzn_word_toks: #tweet tokens
#    for j in w: #word tokens within each tweet
#        if j not in stop_words:
#            filtered_toks.append(j)

#Examine the word frequencies
from nltk.probability import FreqDist
#fdist = FreqDist(filtered_toks)

#print(fdist.most_common(10))
#Without adding to the stop_words set, the top 10 most common words are:
#1) !, 3985
#2) @, 3103
#3) , , 2150
#4) : , 2000
#5) We, 1993
#6) . , 1989
#7) re, 1429
#8) https, 1189
#9) us, 1000
#10) love, 727

#I'm removing (!), (@), (,), (:), (We), (.), (re), and (https) then trying again

filtered_toks2 = []

#Adding the above characters/words to stop_words
stop_words.add('!')
stop_words.add('@')
stop_words.add(',')
stop_words.add(':')
stop_words.add('We')
stop_words.add('.')
stop_words.add('re')
stop_words.add('https')

#for w in amzn_word_toks: #tweet tokens
#    for j in w: #word tokens within each tweet
#        if j not in stop_words:
#            filtered_toks2.append(j)
            
#fdist2 = FreqDist(filtered_toks2)
#print(fdist2.most_common(10))
#Now, the top 10 occurring words are:
#1) 're, 1429 (didn't quite get rid of re)
#2) us, 1000
#3) love, 727
#4) #, 723
#5) details, 627
#6) send, 622
#7) 'd, 559
#8) 's, 547
#9) ?, 531
#10) like, 475

#Removing ('re), ('d), ('s), and (?) then trying again
stop_words.add('\'re')
stop_words.add('\'d')
stop_words.add('\'s')
stop_words.add('?')

filtered_toks3 = []

for w in amzn_word_toks: #tweet tokens
    for j in w: #word tokens within each tweet
        if j not in stop_words:
            filtered_toks3.append(j)

fdist3 = FreqDist(filtered_toks3)
print(fdist3.most_common(10))
#Now, top 10 occurring words are:
#1) us, 1000
#2) love, 727
#3) #, 723
#4) details, 627
#5) send, 622
#6) like, 475
#7) DeliveringSmiles, 440
#8) holiday, 408
#9) Thanks, 401
#10) season, 355

