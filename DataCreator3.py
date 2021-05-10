#This file's purpose is to put together datasets for modeling (removing tweets not able to be assigned topic or sentiment)
#Original data will be loaded in, retweets will be removed, tweets will be marked OT/IRT, and only 'en' and 'und' tweets will be kept
#Furthermore, when possible, tweets will be assigned to their topics from 'BTM_model_OT.py' or 'BTM_model_IRT.py' (depending on tweet type)
#When not possible, tweets will be removed
#When possible, tweets will be assigned sentiment stemming from 'IBM_Sent.py'. When not, tweets will be removed
#Beginning creation of this file by copying/pasting 'Data_creator2.py'
##Note: Depending on whether dataset may be split OT and IRT, or whether dataset only contains enough of one or the either will determine what needs to be commented/uncommented in this file
##Currently, file is set up to just work for IRT data, uncommenting any line with 'OT' present will allow for creation of two datasets simultaneously
#
#Overall, this is program: 82

import pandas as pd
import numpy as np
import nltk, os, sys, email, re


#For saving files:
#path_OT = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Analysis_Data_OT\\'
#commonEnd_OT = '_relTweets_OT.xlsx'
path_IRT = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Analysis_Data_IRT\\'
commonEnd_IRT = '_relTweets_IRT.xlsx'

#Read in original data:
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\McDonalds_Dec_1_2020.xlsx')
company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]

#Read in data stemming from 'BTM_model_OT.py'
#topic_data_OT = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Pictures\\BTMresults_OT\\Toyota_BTMresults_OT.xlsx')

#Read in data stemming from 'BTM_model_IRT.py'
topic_data_IRT = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Pictures\\BTMresults_IRT\\McDonalds_BTMresults_IRT.xlsx')

#Read in data stemming from 'IBM_Sent.py'
sent_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Pictures\\SentimentResults\\McDonalds_sentResults.xlsx')

#Save the number of topics to a variable:
#num_topics = np.max(topic_data_OT["topic"]) + 1
num_topics = np.max(topic_data_IRT["topic"]) + 1
#num_topics_OT = np.max(topic_data_OT["topic"]) + 1
#num_topics_IRT = np.max(topic_data_IRT["topic"]) + 1
#both_tops = [num_topics_OT, num_topics_IRT]
#num_topics = np.max(both_tops)
print("Max topics is %s" % num_topics)

#Create a topic variable in original data, initially assign all tweets to inherent, additional topic:
inh_topic = [num_topics] * len(company_data)
company_data["topic"] = inh_topic
print(company_data["topic"].head(5))
   
    
#Where possible, change official tweet topic to real assignment:
#for i in range(len(topic_data_OT)):
#    orig_loc = topic_data_OT.iloc[i, topic_data_OT.columns.get_loc("Original Location")] #save original location of topic assigned tweet
#    orig_top = topic_data_OT.iloc[i, topic_data_OT.columns.get_loc("topic")] #save the topic the tweet was assigned to
#    company_data.iat[orig_loc, 19] = orig_top


#Where possible, change IRT tweet topic to real assignment:
for i in range(len(topic_data_IRT)):
    orig_loc = topic_data_IRT.iloc[i, topic_data_IRT.columns.get_loc("Original Location")] #save original location of topic assigned tweet
    orig_top = topic_data_IRT.iloc[i, topic_data_IRT.columns.get_loc("topic")] #save the topic the tweet was assigned to
    company_data.iat[orig_loc, 19] = orig_top
    
#See if that worked properly:
print(company_data["topic"].head(5))


#Create a sentiment variable in the data. Initially assign all tweets to sentiment = 2.5
init_sent = [2.5] * len(company_data)
company_data["Sentiment"] = init_sent

#Where possible, change tweet sentiment to real results:
for i in range(len(sent_data)):
    orig_loc = sent_data.iloc[i, sent_data.columns.get_loc("Original Location")] #save original location of sentiment assigned tweet
    orig_sent = sent_data.iloc[i, sent_data.columns.get_loc("Sentiment")] #save the sentiment the tweet was assigned
    print(orig_sent)
    company_data.iat[orig_loc, 20] = orig_sent
    
#Examine results
print(company_data["Sentiment"].head(5))

    

#Save the position in which tweets were read in as a variable:
read_inLoc = list(range(len(company_data)))
#print(read_inLoc)

#company_data["read_inLoc"] = read_inLoc
#print("Within data, it looks like:")
#print(company_data["read_inLoc"].head(5))

#Remove retweets from the company account, as they aren't technically company account tweets
patternDel = "^RT @"
filter1 = company_data["Content"].str.contains(patternDel)
company_tweets = company_data[~filter1].copy()

##Mark tweets as OT or IRT:
#Perform initial separation based on "^@" regex:
initIRT = [bool(re.search("^@", i)) for i in company_tweets["Content"]]
initOT = [not elem for elem in initIRT]
#print(initOT)

#Create IRT and OT variables in the data:
company_tweets["IRT"] = initIRT
company_tweets["OT"] = initOT

#print(company_tweets["IRT"])
#print(company_tweets["OT"])

#Fill in NAs under the 'In Reply To' field with "OT":
company_tweets["In Reply To"] = company_tweets["In Reply To"].replace(np.nan, "OT", regex=True)
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
                    tweets.iat[i, 21] = False #then the original tweet is part of an official thread, not true IRT
                    tweets.iat[i, 22] = True
                    #print(tweets.iloc[i, tweets.columns.get_loc(text1)])
    return tweets

company_tweets = cleanSplit(company_tweets, "Content", "IRT", "In Reply To", "Author", "OT")



#Only keep tweets meant for the English audience (en and und). Other audiences may behave differently
English_tweets = company_tweets[company_tweets["Language"] == "en"]
und_tweets = company_tweets[company_tweets["Language"] == "und"]
company_tweets2 = pd.concat([English_tweets, und_tweets], axis=0)
print("After removing retweets and non-Enlish tweets, we have below amount remaining")
print(company_tweets2.shape)

#Remove tweets not able to be assigned a topic:
company_tweets3 = company_tweets2[company_tweets2["topic"] != num_topics].copy()
print("After removing tweets not able to be assigned a topic, we have below amount remaining")
print(company_tweets3.shape)

#Probably already all removed, but remove tweets not able to be assigned a sentiment
company_tweets4 = company_tweets3[company_tweets3["Sentiment"] != 2.5].copy()
print("After removing tweets not able to be assigned sentiment, we have below amount remaining")
print(company_tweets4.shape)

#Separate into OT and IRT datasets:
#company_tweets_OT = company_tweets4[company_tweets4["OT"] == True].copy()
#print("We have %s official tweets" % len(company_tweets_OT))
company_tweets_IRT = company_tweets4[company_tweets4["IRT"] == True].copy()
print("We have %s IRT tweets" % len(company_tweets_IRT))

#########################################################################
#Save the resulting official tweet file:
#fpath_OT = path_OT + str(company_name) + commonEnd_OT
#company_tweets_OT.to_excel(fpath_OT)

#Save resulting IRT tweet file:
fpath_IRT = path_IRT + str(company_name) + commonEnd_IRT
company_tweets_IRT.to_excel(fpath_IRT)
print("All done")


