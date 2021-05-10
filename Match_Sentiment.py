#This file is being created to match outlier free data with their sentiment scores
#
#Overall, this is program: 77

import pandas as pd
import numpy as np
import nltk, os, sys, email, re


#For saving files:
path = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\OutlierFree_data\\withSentiment\\'
commonEnd = '_outsRem_ws.xlsx'

#Read in outlier free data:
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\OutlierFree_data\\Toyota_outsRemoved.xlsx')
company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]

#Read in sentiment assigned data:
sent_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Pictures\\SentimentResults\\Toyota_sentResults.xlsx')

#Perhaps not the quickest method, but I think this should get the trick done:

#Create a sentiment variable in outlier free data. Initially assign all tweets to sentiment = 2.5
init_sent = [2.5] * len(company_data)
company_data["Sentiment"] = init_sent

print("Entering the long for loop")

for i in range(len(sent_data)):
    orig_loc = sent_data.iloc[i, sent_data.columns.get_loc("Original Location")] #save original location of sentiment assigned tweet
    orig_sent = sent_data.iloc[i, sent_data.columns.get_loc("Sentiment")] #save the tweet's sentiment
    for j in range(len(company_data)):
        if company_data.iloc[j, company_data.columns.get_loc("Original Location")] == orig_loc: #if we've found matching tweets
            company_data.iat[j, 24] = orig_sent #then match their sentiments
            break #and only 1 match can be true, no need to keep looking

print("Before removing tweets not assigned sentiment, we had %s tweets" % len(company_data))            
#Remove tweets not assigned a sentiment score:
company_data2 = company_data[company_data["Sentiment"] != 2.5].copy()
print("After removing tweets not assigned sentiment, %s tweets remain" % len(company_data2))

#Save results:
fpath = path + str(company_name) + commonEnd
company_data2.to_excel(fpath)
            