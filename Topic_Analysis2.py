# -*- coding: utf-8 -*-
#This file is being created to improve upon 'Topic_Analysis.py' (whose purpose was to analyze results from 'BTM_model2.py'
#I'm beginning by copying/pasting said file in here, then adding code to create stacked bar charts
##I may also eventually add code to perform statistical tests indicating whether there's significant differences in performance between topics
#
#Overall, this is program: 45.2

import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import keras
import random
import statistics
import math
import matplotlib.pyplot as plt

#Shouldn't be needed, but it never hurts to be seeded
random.seed(1)
np.random.seed(2)

#For times when a statistic cannot be computed:
NA = 'NA'

##Creating function to produce stacked bar charts:
#Code derived from: https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
#Actually, commented most of that out. New code from: https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
def plot_stacked_bar(df, data, series_labels, count_vals, category_labels=None, 
                     show_values=True, value_format="{}", y_label=None, 
                     colors=None, grid=True, reverse=False, title=None):
    
    
    ny = len(data[0])
    ind = list(range(ny))
    ny2 = len(category_labels)
    ind2 = list(range(ny2))
    
    axes = []
    cum_size = np.zeros(ny)
    
    data = np.array(data)
    
    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)
        
    
    fig = plt.figure()
    #ax = fig.add_axes([0, 0, 1, 1])
    ax = fig.add_subplot(111)
    j = 0
    
    for row in data:
        floorTracker = 0
        for i in range(0, len(row)):
            ax.bar(j, row[i], 0.35, bottom=floorTracker, color=str(colors[i]))
            floorTracker = floorTracker + row[i]
        j = j + 1
     
    ax.set_ylabel(y_label)
    ax.set_xlabel('Category')
    ax.set_title(title)
    #ax.set_xticks(ind, category_labels)
    ax.set_xticks(ind2)
    ax.set_xticklabels(category_labels, fontsize='x-small')
    ax.set_yticks(np.arange(0, 1.1, 1))
    ax.legend(labels=series_labels, bbox_to_anchor=(1.0, 1.0), loc=2, borderaxespad=0.)
    plt.tight_layout()
    #plt.show()
    
    
    j = 0
    if show_values:
        for row in data:
            floorTracker = 0
            no_monopolies = True
            for i in range(0, len(row)):
                if row[i] > 0.99:
                    no_monopolies = False
            if no_monopolies:
                plt.text(j, 1.01, count_vals[j], ha="center", va="center", fontsize="x-small")
            else:
                plt.text(j+0.18, 0.98, count_vals[j], fontsize="x-small")
            for i in range(0, len(row)):
                #ax.bar(j, row[i], 0.35, bottom=floorTracker, color=str(colors[i]))
                w = 0.35
                h = row[i]
                if round(h, 2) > 0.01:
                    plt.text(j, floorTracker + h/2, value_format.format(h), ha="center", va="center", fontsize='xx-small')
                floorTracker = floorTracker + row[i]
            j = j + 1        

    plt.show()
##########################################################################


#Read in the data: Below line of code will need to be reconfigured for your filepath
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Pictures\\BTMresults2\\Toyota Motor Corp._BTMresults2.xlsx')
company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]

#No need to remove retweets here, already taken care of in 'BTM_model2.py'
#Essentially, everything has been taken care of already, and this file should only need to compare topic performances

num_topics = np.max(company_data["topic"]) + 1
total_obs = len(company_data)
print("For %s, there are %s total tweets input for analysis" % (company_name, len(company_data)))
print("And there are a total of %s topics" % num_topics)

#Assign a color to each topic:
all_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'lime', 'b3', 'g3', 'r3', 'c3', 'm3', 'y3', 'tab:brown', 'b9', 'g9', 'r9', 'c9', 'm9']
topic_colors = all_colors[0:num_topics]


#Initialize variable to save all tweet topic proportions into
all_proportions = []
all_counts = [] #to store number observations, when needed
all_counts.append(total_obs)

#Initialize variable to save tweet topic proportions (across all tweets) into:
all_props = []


#And these will make displaying results a much less painful process:
avgLikes = []
avgRT = []
stdevLikes = []
stdevRT = []
topic_count = []

for i in range(0, num_topics):
    #print("Calculating stats for topic %s" % i)
    topical = company_data[company_data["topic"] == i].copy()
    print("Within topic %s, there are a total of %s tweets" % (i, len(topical)))
    topic_obs = len(topical)
    topic_prop = topic_obs / total_obs
    #Save the proportion of tweets belonging to this topic, across all tweets:
    all_props.append(topic_prop)
    print("This constitutes %s of the data" % topic_prop)
    print("These tweets average %s likes, with a standard deviation of %s" % (np.mean(topical["Number of Likes"]), statistics.stdev(topical["Number of Likes"])))
    print("These tweets average %s RTs, with a standard deviation of %s" % (np.mean(topical["Number of Retweets"]), statistics.stdev(topical["Number of Retweets"])))
    avgLikes.append(np.mean(topical["Number of Likes"]))
    stdevLikes.append(statistics.stdev(topical["Number of Likes"]))
    avgRT.append(np.mean(topical["Number of Retweets"]))
    stdevRT.append(statistics.stdev(topical["Number of Retweets"]))
    topic_count.append(len(topical))
    
    
    
#Save tweet topic proportions, across all tweets:
all_proportions.append(all_props)


#Create another variable for ease of displaying results:
numeric_topics = [i for i in range(0, num_topics)]

#Then, create the excel file for displaying results:
easy_table = pd.DataFrame({'Topic':numeric_topics, 'Number Obs': topic_count, 'Data Proportion': all_props, 'Avg. Likes': avgLikes,
                           'Stdev. Likes': stdevLikes, 'Avg. Retweets': avgRT, 'Stdev. Retweets': stdevRT})

easy_table.to_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Pictures\\Easy_Tables\\easy_table.xlsx')



##Examine topic distributions between top 25% performing, median 50% performing, and bottom 25% performing tweets (based on likes)
company_tweets = company_data.sort_values(by = "Number of Likes", ascending=False)

#Extract top 25% performing tweets:
nr = math.ceil(0.25 * len(company_tweets))
top_tweets = company_tweets.head(nr).copy()
all_counts.append(nr)

#Initialize variable to save tweet topic proportions, across top 25% liked tweets, into:
topLiked_props = []

#Examine topic distributions of top 25% performing tweets (based on likes):
print("\n\n\nWithin top liked tweets, there are %s observations" % len(top_tweets))
for i in range(0, num_topics):
    topical_top = top_tweets[top_tweets["topic"] == i]
    tprop = len(topical_top) / len(top_tweets)
    #Save proportion of tweets belonging to this topic across top 25% (based on likes) of all tweets:
    topLiked_props.append(tprop)
    print("There are %s tweets belonging to topic %s, which accounts for %s of top liked tweets" % (len(topical_top), i, tprop))

#Save tweet topic proportions (across top 25% of tweets, based on likes):
all_proportions.append(topLiked_props)
    
#Extract median 50% performing tweets:
desired_num = int(round(0.5 * len(company_tweets)))
nr2 = nr + 1
endPoint = int(nr + desired_num)
if len(company_tweets) % 4 == 0:
    endPoint = endPoint + 1
mid_tweets = company_tweets.iloc[nr2:endPoint]
all_counts.append(len(mid_tweets))

#Initialize variable to save tweet topic distributions (across median liked tweets, based on likes):
midLiked_props = []

#Examine topic distributions of median 50% performing tweets (based on likes):
print("\nWithin median liked tweets, there are %s observations" % len(mid_tweets))
for i in range(0, num_topics):
    topical_mid = mid_tweets[mid_tweets["topic"] == i]
    tprop = len(topical_mid) / len(mid_tweets)
    #Save the proportion of tweets belonging to each topic, across median 50% performing of all tweets (based on likes)
    midLiked_props.append(tprop)
    print("There are %s tweets belonging to topic %s, which accounts for %s of median liked tweets" % (len(topical_mid), i, tprop))

#Save the tweet topic distributions across median 50% performing of all tweets (based on likes):
all_proportions.append(midLiked_props)

#Extract bottom 25% performing tweets:
bot_tweets = company_tweets.tail(nr).copy()
all_counts.append(nr)

#Initialize variable to save above tweet topic proportions:
botLiked_props = []

#Examine topic distributions across bottom 25% performing tweets (based on likes):
print("\nWithin bottom liked tweets, there are %s observations" % len(bot_tweets))
for i in range(0, num_topics):
    topical_bot = bot_tweets[bot_tweets["topic"] == i]
    tprop = len(topical_bot) / len(bot_tweets)
    botLiked_props.append(tprop)
    print("There are %s tweets belonging to topic %s, which accounts for %s of bottom liked tweets" % (len(topical_bot), i, tprop))


#Save the results above:
all_proportions.append(botLiked_props)

##Examine topic distributions between top 25% performing, median 50% performing, and bottom 25% performing tweets (based on retweets)
company_tweets2 = company_data.sort_values(by = "Number of Retweets", ascending=False)

#Extract top 25% performing tweets:
nr = math.ceil(0.25 * len(company_tweets2))
top_tweets2 = company_tweets2.head(nr).copy()
all_counts.append(nr)

#Initialize proportion variable:
topRT_props = []

#Examine topic distributions of top 25% performing tweets (based on retweets):
print("\n\n\nWithin top RT'd tweets, there are %s observations" % len(top_tweets2))
for i in range(0, num_topics):
    topical_top = top_tweets2[top_tweets2["topic"] == i]
    tprop = len(topical_top) / len(top_tweets2)
    topRT_props.append(tprop)
    print("There are %s tweets belonging to topic %s, which accounts for %s of top RT'd tweets" % (len(topical_top), i, tprop))

#Save results here:
all_proportions.append(topRT_props)
    
#Extract median 50% performing tweets:
desired_num = int(round(0.5 * len(company_tweets2)))
nr2 = nr + 1
endPoint = int(nr + desired_num)
if len(company_tweets2) % 4 == 0:
    endPoint = endPoint + 1
mid_tweets2 = company_tweets2.iloc[nr2:endPoint]
all_counts.append(len(mid_tweets2))

#Initialize variable to save proportions into:
midRT_props = []

#Examine topic distributions of median 50% performing tweets (based on retweets):
print("\nWithin median RT'd tweets, there are %s observations" % len(mid_tweets2))
for i in range(0, num_topics):
    topical_mid = mid_tweets2[mid_tweets2["topic"] == i]
    tprop = len(topical_mid) / len(mid_tweets2)
    midRT_props.append(tprop)
    print("There are %s tweets belonging to topic %s, which accounts for %s of median RT'd tweets" % (len(topical_mid), i, tprop))

#Save results here:
all_proportions.append(midRT_props)
    
#Extract bottom 25% performing tweets:
bot_tweets2 = company_tweets2.tail(nr).copy()
all_counts.append(nr)

#Initialize variable to save results:
botRT_props = []

#Examine topic distributions across bottom 25% performing tweets (based on retweets):
print("\nWithin bottom RT'd tweets, there are %s observations" % len(bot_tweets2))
for i in range(0, num_topics):
    topical_bot = bot_tweets2[bot_tweets2["topic"] == i]
    tprop = len(topical_bot) / len(bot_tweets2)
    botRT_props.append(tprop)
    print("There are %s tweets belonging to topic %s, which accounts for %s of bottom RT'd tweets" % (len(topical_bot), i, tprop))

#Save results:
all_proportions.append(botRT_props)

##Create stacked barchart across set of all company tweets:

#Create variable to save category labels:
cat_labs = ['All Tweets', 'Top 25%\nLiked', 'Median 50%\nLiked', 'Bottom 25%\nLiked', 'Top 25%\nRetweeted', 'Median 50%\nRetweeted', 'Bottom 25%\nRetweeted']

#Save topic labels:
topic_labs = []
topic_tag = 'Topic '

for i in range(0, num_topics):
    t = topic_tag + str(i)
    topic_labs.append(t)
    
allTweet_title = str(company_name) + ': All Tweets'

plot_stacked_bar(company_data, all_proportions, topic_labs, all_counts,  category_labels=cat_labs, show_values=True, value_format="{:.2f}", y_label="Relevant Categorical Proportion", colors=topic_colors, title=allTweet_title)
#plt.show()
#print("Break")





    
print("\nPreparing for split analysis")
official_tweets = company_data[company_data["OT"] == True].copy()
IRT_tweets = company_data[company_data["IRT"] == True].copy()

num_OT = len(official_tweets)
num_IRT = len(IRT_tweets)

#Setting 75 as the minimum number of tweets required for analysis
minThresh = 75

#If there are enough tweets in both categories for separate analysis:
if num_OT >= minThresh and num_IRT >= minThresh:
    print("%s seems to have enough tweets of both types for split analysis" % company_name)
    
    #Initialize variables needed for plotting stacked bar charts:
    official_proportions = []
    official_props = []
    official_counts = []
    
    #Reset variables to be used in displaying results:
    avgLikes = []
    avgRT = []
    stdevLikes = []
    stdevRT = []
    topic_count = []    
    
    print("\nBeginning official tweet analysis")
    print("There are a total of %s official tweets" % num_OT)
    for i in range(0, num_topics):
        #print("Calculating stats for topic %s" % i)
        topical = official_tweets[official_tweets["topic"] == i].copy()
        print("Within topic %s, there are a total of %s tweets" % (i, len(topical)))
        topic_obs = len(topical)
        topic_prop = topic_obs / num_OT
        official_props.append(topic_prop)
        print("This constitutes %s of all official tweets" % topic_prop)
        try:
            print("These tweets average %s likes, with a standard deviation of %s" % (np.mean(topical["Number of Likes"]), statistics.stdev(topical["Number of Likes"])))
        except statistics.StatisticsError:
            print("These tweets average %s likes, with a standard deviation of %s" % (np.mean(topical["Number of Likes"]), NA))
        try:
            print("These tweets average %s RTs, with a standard deviation of %s" % (np.mean(topical["Number of Retweets"]), statistics.stdev(topical["Number of Retweets"])))
        except statistics.StatisticsError:
            print("These tweets average %s RTs, with a standard deviation of %s" % (np.mean(topical["Number of Retweets"]), NA))
        #Save variables for ease of displaying results:
        try:
            avgLikes.append(np.mean(topical["Number of Likes"]))
        except statistics.StatisticsError:
            avgLikes.append(NA)
        try:
            stdevLikes.append(statistics.stdev(topical["Number of Likes"]))
        except statistics.StatisticsError:
            stdevLikes.append(NA)
        try:
            avgRT.append(np.mean(topical["Number of Retweets"]))
        except statistics.StatisticsError:
            avgRT.append(NA)
        try:
            stdevRT.append(statistics.stdev(topical["Number of Retweets"]))
        except statistics.StatisticsError:
            stdevRT.append(NA)
        topic_count.append(len(topical))
        
    
    #Save results for easy displaying:
    easy_table2 = pd.DataFrame({'Topic':numeric_topics, 'Number Obs': topic_count, 'Data Proportion': official_props, 'Avg. Likes': avgLikes,
                               'Stdev. Likes': stdevLikes, 'Avg. Retweets': avgRT, 'Stdev. Retweets': stdevRT})
    
    easy_table2.to_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Pictures\\Easy_Tables\\easy_table_OT.xlsx')
    
    
    
    #Save topic proportions within official tweets:
    official_proportions.append(official_props)    
    ##Examine topic distributions between top 25% performing, median 50% performing, and bottom 25% performing official tweets (based on likes)
    #Just changing line below should allow for code to run without changing anything else besides comments, print statements, and setting of company_tweets2
    company_tweets = official_tweets.sort_values(by = "Number of Likes", ascending=False)
    official_counts.append(len(company_tweets))
    
    #Extract top 25% performing tweets:
    nr = math.ceil(0.25 * len(company_tweets))
    top_tweets = company_tweets.head(nr).copy()
    official_counts.append(len(top_tweets))
    
    topOT_props = []
    #Examine topic distributions of top 25% performing official tweets (based on likes):
    print("\n\n\nWithin top liked official tweets, there are %s observations" % len(top_tweets))
    for i in range(0, num_topics):
        topical_top = top_tweets[top_tweets["topic"] == i]
        tprop = len(topical_top) / len(top_tweets)
        topOT_props.append(tprop)
        print("There are %s tweets belonging to topic %s, which accounts for %s of top liked official tweets" % (len(topical_top), i, tprop))
    
    official_proportions.append(topOT_props)    
    #Extract median 50% performing official tweets:
    desired_num = int(round(0.5 * len(company_tweets)))
    nr2 = nr + 1
    endPoint = int(nr + desired_num)
    if len(company_tweets) % 4 == 0:
        endPoint = endPoint + 1
    mid_tweets = company_tweets.iloc[nr2:endPoint]
    official_counts.append(len(mid_tweets))
    
    midOT_props = []
    #Examine topic distributions of median 50% performing official tweets (based on likes):
    print("\nWithin median liked official tweets, there are %s observations" % len(mid_tweets))
    for i in range(0, num_topics):
        topical_mid = mid_tweets[mid_tweets["topic"] == i]
        tprop = len(topical_mid) / len(mid_tweets)
        midOT_props.append(tprop)
        print("There are %s tweets belonging to topic %s, which accounts for %s of median liked official tweets" % (len(topical_mid), i, tprop))
    
    official_proportions.append(midOT_props)    
    #Extract bottom 25% performing official tweets:
    bot_tweets = company_tweets.tail(nr).copy()
    official_counts.append(len(bot_tweets))
    
    botOT_props = []
    #Examine topic distributions across bottom 25% performing official tweets (based on likes):
    print("\nWithin bottom liked official tweets, there are %s observations" % len(bot_tweets))
    for i in range(0, num_topics):
        topical_bot = bot_tweets[bot_tweets["topic"] == i]
        tprop = len(topical_bot) / len(bot_tweets)
        botOT_props.append(tprop)
        print("There are %s tweets belonging to topic %s, which accounts for %s of bottom liked official tweets" % (len(topical_bot), i, tprop))
    
    
    official_proportions.append(botOT_props)
    ##Examine topic distributions between top 25% performing, median 50% performing, and bottom 25% performing official tweets (based on retweets)
    company_tweets2 = official_tweets.sort_values(by = "Number of Retweets", ascending=False)
    
    #Extract top 25% performing tweets:
    nr = math.ceil(0.25 * len(company_tweets2))
    top_tweets2 = company_tweets2.head(nr).copy()
    official_counts.append(len(top_tweets2))
    
    topOT_props2 = []
    #Examine topic distributions of top 25% performing official tweets (based on retweets):
    print("\n\n\nWithin top RT'd official tweets, there are %s observations" % len(top_tweets2))
    for i in range(0, num_topics):
        topical_top = top_tweets2[top_tweets2["topic"] == i]
        tprop = len(topical_top) / len(top_tweets2)
        topOT_props2.append(tprop)
        print("There are %s tweets belonging to topic %s, which accounts for %s of top RT'd official tweets" % (len(topical_top), i, tprop))
    
    official_proportions.append(topOT_props2)    
    #Extract median 50% performing official tweets:
    desired_num = int(round(0.5 * len(company_tweets2)))
    nr2 = nr + 1
    endPoint = int(nr + desired_num)
    if len(company_tweets2) % 4 == 0:
        endPoint = endPoint + 1
    mid_tweets2 = company_tweets2.iloc[nr2:endPoint]
    official_counts.append(len(mid_tweets2))
    
    midOT_props2 = []
    #Examine topic distributions of median 50% performing official tweets (based on retweets):
    print("\nWithin median RT'd official tweets, there are %s observations" % len(mid_tweets2))
    for i in range(0, num_topics):
        topical_mid = mid_tweets2[mid_tweets2["topic"] == i]
        tprop = len(topical_mid) / len(mid_tweets2)
        midOT_props2.append(tprop)
        print("There are %s tweets belonging to topic %s, which accounts for %s of median RT'd official tweets" % (len(topical_mid), i, tprop))
     
    official_proportions.append(midOT_props2)   
    #Extract bottom 25% performing official tweets:
    bot_tweets2 = company_tweets2.tail(nr).copy()
    official_counts.append(len(bot_tweets2))
    
    botOT_props2 = []
    #Examine topic distributions across bottom 25% performing official tweets (based on retweets):
    print("\nWithin bottom RT'd official tweets, there are %s observations" % len(bot_tweets2))
    for i in range(0, num_topics):
        topical_bot = bot_tweets2[bot_tweets2["topic"] == i]
        tprop = len(topical_bot) / len(bot_tweets2)
        botOT_props2.append(tprop)
        print("There are %s tweets belonging to topic %s, which accounts for %s of bottom RT'd official tweets" % (len(topical_bot), i, tprop))
    
    official_proportions.append(botOT_props2)
    
    ##Create stacked barchart for tweets belonging to official category:
    #Create variable to save category labels:
    cat_labs = ['All Official\nTweets', 'Top 25%\nLiked', 'Median 50%\nLiked', 'Bottom 25%\nLiked', 'Top 25%\nRetweeted', 'Median 50%\nRetweeted', 'Bottom 25%\nRetweeted']
    
    #Save topic labels:
    topic_labs = []
    topic_tag = 'Topic '
    
    for i in range(0, num_topics):
        t = topic_tag + str(i)
        topic_labs.append(t)
        
    otTweet_title = str(company_name) + ': Official Tweets'
    
    plot_stacked_bar(company_data, official_proportions, topic_labs, official_counts,  category_labels=cat_labs, show_values=True, value_format="{:.2f}", y_label="Relevant Categorical Proportion", colors=topic_colors, title=otTweet_title)
    
    ################################################################################################################################################
    #Create variables for plotting stacked bars:
    IRT_proportions = []
    IRT_props = []
    IRT_counts = []
    
    #Reset variables to be used in displaying results:
    avgLikes = []
    avgRT = []
    stdevLikes = []
    stdevRT = []
    topic_count = []     
    
    
    print("\n\n\nBeginning IRT tweet analysis")
    print("There are a total of %s IRT tweets" % num_IRT)
    for i in range(0, num_topics):
        #print("Calculating stats for topic %s" % i)
        topical = IRT_tweets[IRT_tweets["topic"] == i].copy()
        print("Within topic %s, there are a total of %s IRT tweets" % (i, len(topical)))
        topic_obs = len(topical)
        topic_prop = topic_obs / num_IRT
        IRT_props.append(topic_prop)
        print("This constitutes %s of all IRT tweets" % topic_prop)
        try:
            print("These tweets average %s likes, with a standard deviation of %s" % (np.mean(topical["Number of Likes"]), statistics.stdev(topical["Number of Likes"])))
        except statistics.StatisticsError:
            print("These tweets average %s likes, with a standard deviation of %s" % (np.mean(topical["Number of Likes"]), NA))
        try:
            print("These tweets average %s RTs, with a standard deviation of %s" % (np.mean(topical["Number of Retweets"]), statistics.stdev(topical["Number of Retweets"])))
        except statistics.StatisticsError:
            print("These tweets average %s RTs, with a standard deviation of %s" % (np.mean(topical["Number of Retweets"]), NA))
        #Save variables for ease of displaying results:
        try:
            avgLikes.append(np.mean(topical["Number of Likes"]))
        except statistics.StatisticsError:
            avgLikes.append(NA)
        try:
            stdevLikes.append(statistics.stdev(topical["Number of Likes"]))
        except statistics.StatisticsError:
            stdevLikes.append(NA)
        try:
            avgRT.append(np.mean(topical["Number of Retweets"]))
        except statistics.StatisticsError:
            avgRT.append(NA)
        try:
            stdevRT.append(statistics.stdev(topical["Number of Retweets"]))
        except statistics.StatisticsError:
            stdevRT.append(NA)
        topic_count.append(len(topical))
    
    
    #Save results for easy displaying:
    easy_table3 = pd.DataFrame({'Topic':numeric_topics, 'Number Obs': topic_count, 'Data Proportion': IRT_props, 'Avg. Likes': avgLikes,
                               'Stdev. Likes': stdevLikes, 'Avg. Retweets': avgRT, 'Stdev. Retweets': stdevRT})
    
    easy_table3.to_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Pictures\\Easy_Tables\\easy_table_IRT.xlsx')
    
         
            
    #Save tweet topic proportions across all IRT tweets:      
    IRT_proportions.append(IRT_props)   
    ##Examine topic distributions between top 25% performing, median 50% performing, and bottom 25% performing IRT tweets (based on likes)
    #Just changing line below should allow for code to run without changing anything else besides comments, print statements, and setting of company_tweets2
    company_tweets = IRT_tweets.sort_values(by = "Number of Likes", ascending=False)
    IRT_counts.append(len(company_tweets))
    
    #Extract top 25% performing IRT tweets:
    nr = math.ceil(0.25 * len(company_tweets))
    top_tweets = company_tweets.head(nr).copy()
    IRT_counts.append(len(top_tweets))
    
    topIRT_props = []
    #Examine topic distributions of top 25% performing IRT tweets (based on likes):
    print("\n\n\nWithin top liked IRT tweets, there are %s observations" % len(top_tweets))
    for i in range(0, num_topics):
        topical_top = top_tweets[top_tweets["topic"] == i]
        tprop = len(topical_top) / len(top_tweets)
        topIRT_props.append(tprop)
        print("There are %s tweets belonging to topic %s, which accounts for %s of top liked IRT tweets" % (len(topical_top), i, tprop))
    
    IRT_proportions.append(topIRT_props)    
    #Extract median 50% performing IRT tweets:
    desired_num = int(round(0.5 * len(company_tweets)))
    nr2 = nr + 1
    endPoint = int(nr + desired_num)
    if len(company_tweets) % 4 == 0:
        endPoint = endPoint + 1
    mid_tweets = company_tweets.iloc[nr2:endPoint]
    IRT_counts.append(len(mid_tweets))
    
    midIRT_props = []
    #Examine topic distributions of median 50% performing IRT tweets (based on likes):
    print("\nWithin median liked IRT tweets, there are %s observations" % len(mid_tweets))
    for i in range(0, num_topics):
        topical_mid = mid_tweets[mid_tweets["topic"] == i]
        tprop = len(topical_mid) / len(mid_tweets)
        midIRT_props.append(tprop)
        print("There are %s tweets belonging to topic %s, which accounts for %s of median liked IRT tweets" % (len(topical_mid), i, tprop))
    
    IRT_proportions.append(midIRT_props)   
    #Extract bottom 25% performing IRT tweets:
    bot_tweets = company_tweets.tail(nr).copy()
    IRT_counts.append(len(bot_tweets))
    
    botIRT_props = []
    #Examine topic distributions across bottom 25% performing IRT tweets (based on likes):
    print("\nWithin bottom liked IRT tweets, there are %s observations" % len(bot_tweets))
    for i in range(0, num_topics):
        topical_bot = bot_tweets[bot_tweets["topic"] == i]
        tprop = len(topical_bot) / len(bot_tweets)
        botIRT_props.append(tprop)
        print("There are %s tweets belonging to topic %s, which accounts for %s of bottom liked IRT tweets" % (len(topical_bot), i, tprop))
    
    
    IRT_proportions.append(botIRT_props)
    ##Examine topic distributions between top 25% performing, median 50% performing, and bottom 25% performing IRT tweets (based on retweets)
    company_tweets2 = IRT_tweets.sort_values(by = "Number of Retweets", ascending=False)
    
    
    #Extract top 25% performing IRT tweets:
    nr = math.ceil(0.25 * len(company_tweets2))
    top_tweets2 = company_tweets2.head(nr).copy()
    IRT_counts.append(len(top_tweets2))
    
    topIRT_props2 = []
    #Examine topic distributions of top 25% performing IRT tweets (based on retweets):
    print("\n\n\nWithin top RT'd IRT tweets, there are %s observations" % len(top_tweets2))
    for i in range(0, num_topics):
        topical_top = top_tweets2[top_tweets2["topic"] == i]
        tprop = len(topical_top) / len(top_tweets2)
        topIRT_props2.append(tprop)
        print("There are %s tweets belonging to topic %s, which accounts for %s of top RT'd IRT tweets" % (len(topical_top), i, tprop))
    
    IRT_proportions.append(topIRT_props2)
    #Extract median 50% performing IRT tweets:
    desired_num = int(round(0.5 * len(company_tweets2)))
    nr2 = nr + 1
    endPoint = int(nr + desired_num)
    if len(company_tweets2) % 4 == 0:
        endPoint = endPoint + 1
    mid_tweets2 = company_tweets2.iloc[nr2:endPoint]
    IRT_counts.append(len(mid_tweets2))
    
    midIRT_props2 = []
    #Examine topic distributions of median 50% performing IRT tweets (based on retweets):
    print("\nWithin median RT'd IRT tweets, there are %s observations" % len(mid_tweets2))
    for i in range(0, num_topics):
        topical_mid = mid_tweets2[mid_tweets2["topic"] == i]
        tprop = len(topical_mid) / len(mid_tweets2)
        midIRT_props2.append(tprop)
        print("There are %s tweets belonging to topic %s, which accounts for %s of median RT'd IRT tweets" % (len(topical_mid), i, tprop))
    
    IRT_proportions.append(midIRT_props2) 
    #Extract bottom 25% performing IRT tweets:
    bot_tweets2 = company_tweets2.tail(nr).copy()
    IRT_counts.append(len(bot_tweets2))
    
    botIRT_props2 = []
    #Examine topic distributions across bottom 25% performing IRT tweets (based on retweets):
    print("\nWithin bottom RT'd IRT tweets, there are %s observations" % len(bot_tweets2))
    for i in range(0, num_topics):
        topical_bot = bot_tweets2[bot_tweets2["topic"] == i]
        tprop = len(topical_bot) / len(bot_tweets2)
        botIRT_props2.append(tprop)
        print("There are %s tweets belonging to topic %s, which accounts for %s of bottom RT'd IRT tweets" % (len(topical_bot), i, tprop))
    
    IRT_proportions.append(botIRT_props2)
    
    ##Create stacked barchart for IRT tweets:
    #Create variable to save category labels:
    cat_labs = ['All IRT Tweets', 'Top 25%\nLiked', 'Median 50%\nLiked', 'Bottom 25%\nLiked', 'Top 25%\nRetweeted', 'Median 50%\nRetweeted', 'Bottom 25%\nRetweeted']
    
    #Save topic labels:
    topic_labs = []
    topic_tag = 'Topic '
    
    for i in range(0, num_topics):
        t = topic_tag + str(i)
        topic_labs.append(t)
        
    irtTweet_title = str(company_name) + ': IRT Tweets'
    
    plot_stacked_bar(company_data, IRT_proportions, topic_labs, IRT_counts,  category_labels=cat_labs, show_values=True, value_format="{:.2f}", y_label="Relevant Categorical Proportion", colors=topic_colors, title=irtTweet_title)
    
    
        
else:
    if num_OT < minThresh:
        print("It seems that %s doesn't have enough official tweets for split analysis. There are only %s OT tweets." % (company_name, num_OT))
    if num_IRT < minThresh:
        print("It seems that %s doesn't have enough IRT tweets for split analysis. There are only %s IRT tweets." % (company_name, num_IRT))