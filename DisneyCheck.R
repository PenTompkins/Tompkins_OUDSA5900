#Making this script to see why 16 (I think) rows of Disney data get marked as IRT
#when they should all be official tweets. I think this is because there are instances where
#Disney replies to itself, but starts the reply with another @username mention.
#This file is also used to ensure that Google tweets including 'ar' and 'na' legitimately include those terms, and this is not simply a..
#..byproduct of improper textual pre-processing

#Overall, this is program: 11

library(readxl)
#read in the data: Below line of code will have to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/Disney_Dec_1_2020.xlsx")

#Filter out retweets
tweets = data[-grep("^RT @", data$Content),]

#Split data into OT vs. IRT
official_tweets = tweets[-grep("^@", tweets$Content),]
IRT_tweets = tweets[grep("^@", tweets$Content),]

#Checking to make sure that 'noIRT_Count.py' results make sense:
episodes = tweets[grep("\\b[Ee]pisodes?\\b", tweets$Content),]
mean(episodes[[6]]) #Disney tweets including 'episode(s)' avg 1150.404 likes

anniversary = tweets[grep("\\b[Aa]nniversary\\b", tweets$Content),]
mean(anniversary[[6]]) #Disney tweets including 'anniversary' avg. 4488.911 likes

##################################################################################################

#Checking Google data for the 'ar' term:
#read in the data: Below line of code will have to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/Google_Dec_1_2020.xlsx")

#Filter out retweets
tweets = data[-grep("^RT @", data$Content),]

#Find tweets including 'ar':
goog_ar = tweets[grep("\\b[Aa][Rr]\\b", tweets$Content),]

#Find tweets including 'na':
goog_na = tweets[grep("\\b[Nn][Aa]\\b", tweets$Content),]
#Seems that Google tweets do legitimately include the terms 'ar' and 'na'




























