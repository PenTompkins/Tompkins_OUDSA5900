#Initial purpose of this file was to examine/explore the other 9 non-Amazon datasets
#However, only BMW data ended up being examined in this file
#Overall, this is program: 9

library(readxl)

##################################################################################################
#Examination of BMW data:
#Read in the data: Below line of code will need to be reconfigured for your filepath
bmw_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/BMW_Dec_1_2020.xlsx")

RT_likes = data.frame(bmw_data["Number of Retweets"], bmw_data["Number of Likes"])
plot(RT_likes)
#There appears to be the issue of retweets, same as in the Amazon data
#Filtering out retweets from the BWM data:
bmw_tweets = bmw_data[-grep("^RT @", bmw_data$Content),]
#Seems to have removed 97 retweets from the data

#Replot Retweets vs. Likes
RT_likes = data.frame(bmw_tweets["Number of Retweets"], bmw_tweets["Number of Likes"])
plot(RT_likes)
#appears much more linear, but not perfectly so

#How many likes/retweets does BMW average across all tweets?
mean(bmw_tweets[[6]]) #Overall, BMW averages 273.58 likes per tweet
mean(bmw_tweets[[7]]) #Overall, BMW averages 28.66 RT per tweet
#273.58/28.66 = 9.54, no longer seeing the likes/RT = 6 relationship (like in Amazon data)
#However, it's possible that this 9.5 remains fairly constant for BMW data

#For BMW, let's examine the differences between OT and IRT categories:
bmw_OT = bmw_tweets[-grep("^@", bmw_tweets$Content),] #806 official BMW tweets
bmw_IRT = bmw_tweets[grep("^@", bmw_tweets$Content),] #2347 IRT BMW tweets

mean(bmw_OT[[6]])#BMW's official tweets avg 1063.79 likes
mean(bmw_OT[[7]])#BMW's official tweets avg 111.72 RT
mean(bmw_IRT[[6]])#BMW's IRT tweets average 2.21 likes
mean(bmw_IRT[[7]])#BMW's IRT tweets average 0.1398 RT

#Examining effect of including link for BWM tweets:
bmw_link = bmw_tweets[grep("https://", bmw_tweets$Content),]
bmw_noLink = bmw_tweets[-grep("https://", bmw_tweets$Content),]

mean(bmw_link[[6]])#BMW tweets with link avg 685.05 likes
mean(bmw_link[[7]])#BMW tweets with link avg 72.41 RT
mean(bmw_noLink[[6]])#BMW tweets without link avg 48.78 likes
mean(bmw_noLink[[7]])#BMW tweets without link avg 4.76 RT

#Split based on link and OT vs. IRT:
bmw_OTlink = bmw_OT[grep("https://", bmw_OT$Content),]
bmw_OTnoLink = bmw_OT[-grep("https://", bmw_OT$Content),]
bmw_IRTlink = bmw_IRT[grep("https://",bmw_IRT$Content),]
bmw_IRTnoLink = bmw_IRT[-grep("https://",bmw_IRT$Content),]

mean(bmw_OTlink[[6]])#BMW official tweets w/ link avg 1281.23 likes
mean(bmw_OTlink[[7]])#BMW official tweets w/ link avg 135.46 RT
mean(bmw_OTnoLink[[6]])#BMW official tweets w/ NO link avg 450.64 likes
mean(bmw_OTnoLink[[7]])#BMW official tweets w/ NO link avg 44.72 RT
mean(bmw_IRTlink[[6]])#BMW IRT tweets w/ link avg 1.56 likes
mean(bmw_IRTlink[[7]])#BMW IRT tweets w/ link avg 0.12 RT
mean(bmw_IRTnoLink[[6]])#BMW IRT tweets w/ NO link avg 2.3961 likes
mean(bmw_IRTnoLink[[7]])#BMW IRT tweets w/ NO link avg 0.1461 RT

##################################################################################
#What is 'iaa19'?
iaa19 = bmw_tweets[grep("[Ii][Aa][Aa]19", bmw_tweets$Content),]
#What about these hashtags?
the7 = bmw_tweets[grep("#[Tt][Hh][Ee]7", bmw_tweets$Content),]
hashtag_the = bmw_tweets[grep("#[Tt]he", bmw_tweets$Content),]
thex7 = bmw_tweets[grep("#[Tt]he[Xx]7", bmw_tweets$Content),]

























