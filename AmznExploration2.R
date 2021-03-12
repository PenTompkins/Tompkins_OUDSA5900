#This R script is to examine the top 10 most common words found in AmazonExploration.py
#Overall, this is program: 4

library(readxl)
#Read in the Amazon data from excel: Below line of code will have to be configured for your filepath
amazon_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/Amazon_Dec_1_2020.xlsx")

#Remove Amazon's retweets from the data
amazon_tweets = amazon_data[-grep("^RT @", amazon_data$Content),]
#Might as well examine the average likes and retweets across all of Amazon's tweets:
mean(amazon_tweets[[6]])#Amazon tweets average 30.65 likes
mean(amazon_tweets[[7]])#Amazon tweets average 5.14 RT

#Create baseline CS and OT datasets:
all_CS = amazon_tweets[grep("^@", amazon_tweets$Content),]
all_OT = amazon_tweets[-grep("^@", amazon_tweets$Content),]
mean(all_CS[[6]]) #CS tweets avg 3.45 likes
mean(all_CS[[7]]) #CS tweets avg 0.5717 RT
mean(all_OT[[6]]) #OT tweets avg 354.39 likes
mean(all_OT[[7]]) #OT tweets avg 59.48 RT
#Interesting that mean(CS likes, CS RT)*100 ~=~ mean(OT likes, OT RT)

#Creating a dataset for each of the top 10 occurring words:
with_us = amazon_tweets[grep("\\b[Uu]s\\b", amazon_tweets$Content),]
with_love = amazon_tweets[grep("\\b[Ll]ove\\b", amazon_tweets$Content),]
with_hashtag = amazon_tweets[grep("#", amazon_tweets$Content),]
with_details = amazon_tweets[grep("\\b[Dd]etails\\b", amazon_tweets$Content),]
with_send = amazon_tweets[grep("\\b[Ss]end\\b", amazon_tweets$Content),]
with_like = amazon_tweets[grep("\\b[Ll]ike\\b", amazon_tweets$Content),]
with_DeliveringSmiles = amazon_tweets[grep("\\bDeliveringSmiles\\b", amazon_tweets$Content),]
with_holiday = amazon_tweets[grep("\\b[Hh]oliday\\b", amazon_tweets$Content),]
with_thanks = amazon_tweets[grep("\\b[Tt]hanks\\b", amazon_tweets$Content),]
with_season = amazon_tweets[grep("\\b[Ss]eason\\b", amazon_tweets$Content),]

mean(with_us[[6]])#avg of 13.03 likes
mean(with_us[[7]])#avg of 2.33 RT
mean(with_love[[6]])#avg of 10.85 likes
mean(with_love[[7]])#avg of 1.53 RT
mean(with_hashtag[[6]])#avg of 40.28 likes
mean(with_hashtag[[7]])#avg of 7.09 RT
mean(with_details[[6]])#avg of 2.64 likes
mean(with_details[[7]])#avg of 0.43 RT
mean(with_send[[6]])#avg of 2.88 likes
mean(with_send[[7]])#avg of 0.46 RT
mean(with_like[[6]])#avg of 21.35 likes
mean(with_like[[7]])#avg of 3.41 RT
mean(with_DeliveringSmiles[[6]])#avg of 9.8 likes
mean(with_DeliveringSmiles[[7]])#avg of 1.53 RT
mean(with_holiday[[6]])#avg of 12.80 likes
mean(with_holiday[[7]])#avg of 2.26 RT
mean(with_thanks[[6]])#avg of 6.92 likes
mean(with_thanks[[7]])#avg of 1.28 RT
mean(with_season[[6]])#avg of 5.23 likes
mean(with_season[[7]])#avg of 0.93 RT

#However, I haven't accounted for customer service vs official tweets yet:
#with_us broken down further:
with_usCS = with_us[grep("^@", with_us$Content),]
with_usOT = with_us[-grep("^@", with_us$Content),]
mean(with_usCS[[6]])#CS tweets including 'us' avg 1.63 likes
mean(with_usCS[[7]])#CS tweets including 'us' avg 0.34 RT
mean(with_usOT[[6]])#Official tweets including 'us' avg 546.79 likes
mean(with_usOT[[7]])#Official tweets including 'us' avg 95.63 RT

#with_love broken down further:
with_loveCS = with_love[grep("^@", with_love$Content),]
with_loveOT = with_love[-grep("^@", with_love$Content),]
mean(with_loveCS[[6]])#CS tweets including 'love' avg 4.82 likes
mean(with_loveCS[[7]])#CS tweets including 'love' avg 0.51 RT
mean(with_loveOT[[6]])#Official tweets including 'love' avg 517.25 likes
mean(with_loveOT[[7]])#Official tweets including 'love' avg 87.25 RT
#Seems that using the word 'love' has more of an impact on CS tweets than official tweets
#Haven't yet accounted for tweets belonging to multiple groups
#I image that some of tweets including 'love' and 'us' overlap, for example

#with_hashtag broken down further:
with_hashtagCS = with_hashtag[grep("^@", with_hashtag$Content),]
with_hashtagOT = with_hashtag[-grep("^@", with_hashtag$Content),]
mean(with_hashtagCS[[6]])#CS tweets including '#' avg 4.34 likes
mean(with_hashtagCS[[7]])#CS tweets including '#' avg 0.39 RT
mean(with_hashtagOT[[6]])#Official tweets including '#' avg 238.13 likes
mean(with_hashtagOT[[7]])#Official tweets including '#' avg 43.96 RT
#Including hashtag may improve CS likes, but doesn't seem to improve anything else

#with_details broken down further:
with_detailsCS = with_details[grep("^@", with_details$Content),]
with_detailsOT = with_details[-grep("^@", with_details$Content),]
mean(with_detailsCS[[6]])#CS tweets including 'details' avg 1.32 likes
mean(with_detailsCS[[7]])#CS tweets including 'details' avg 0.22 RT
mean(with_detailsOT[[6]])#Official tweets including 'details' avg 207.5 likes
mean(with_detailsOT[[7]])#Official tweets including 'details' avg 32.75 RT
#Tweets including the word details seem to perform worse in all categories
#For CS, that may be because 'details' most commonly occurs when people are having
#issues and Amazon wants them to send them the details

#with_send broken down further:
with_sendCS = with_send[grep("^@", with_send$Content),]
with_sendOT = with_send[-grep("^@", with_send$Content),]
mean(with_sendCS[[6]])#CS tweets including 'send' avg 1.32 likes
mean(with_sendCS[[7]])#CS tweets including 'send' avg 0.20 RT
mean(with_sendOT[[6]])#Official tweets including 'send' avg 948 likes
mean(with_sendOT[[7]])#Official tweets including 'send' avg 154 RT
#Same can probably be said about CS tweets including 'send'
#with_sendOT only had one observation though, too small of sample size

#with_like broken down further:
with_likeCS = with_like[grep("^@", with_like$Content),]
with_likeOT = with_like[-grep("^@", with_like$Content),]
mean(with_likeCS[[6]])#CS tweets including 'like' avg 1.63 likes
mean(with_likeCS[[7]])#CS tweets including 'like' avg 0.30 RT
mean(with_likeOT[[6]])#Official tweets including 'like' avg 618.5 likes
mean(with_likeOT[[7]])#Official tweets including 'like' avg 97.71 RT
#Including the word like seems to be negative for CS tweets
#However, it seems to be positive for OT tweets

#with_DeliveringSmiles broken down further:
with_DeliveringSmilesCS = with_DeliveringSmiles[grep("^@", with_DeliveringSmiles$Content),]
with_DeliveringSmilesOT = with_DeliveringSmiles[-grep("^@", with_DeliveringSmiles$Content),]
mean(with_DeliveringSmilesCS[[6]])#CS tweets including 'DeliveringSmiles' avg 0.86 likes
mean(with_DeliveringSmilesCS[[7]])#CS tweets including 'DeliveringSmiles' avg 0.06 RT
mean(with_DeliveringSmilesOT[[6]])#Official tweets including 'DeliveringSmiles' avg 304.69 likes
mean(with_DeliveringSmilesOT[[7]])#Official tweets including 'DeliveringSmiles' avg 50.31 RT
#Seems that this was an unpopular hashtag, for both CS and OT tweets (especially CS)

#with_holiday broken down further:
with_holidayCS = with_holiday[grep("^@", with_holiday$Content),]
with_holidayOT = with_holiday[-grep("^@", with_holiday$Content),]
mean(with_holidayCS[[6]])#CS tweets including 'holiday' avg 0.98 likes
mean(with_holidayCS[[7]])#CS tweets including 'holiday' avg 0.11 RT
mean(with_holidayOT[[6]])#Official tweets including 'holiday' avg 219.38 likes
mean(with_holidayOT[[7]])#Official tweets including 'holiday' avg 39.81 RT
#For some reason, including 'holiday' doesn't seem to help any categories

#with_thanks broken down further:
with_thanksCS = with_thanks[grep("^@", with_thanks$Content),]
with_thanksOT = with_thanks[-grep("^@", with_thanks$Content),]
mean(with_thanksCS[[6]])#CS tweets including 'thanks' avg 3.15 likes
mean(with_thanksCS[[7]])#CS tweets including 'thanks' avg 0.54 RT
mean(with_thanksOT[[6]])#Official tweets including 'thanks' avg 224.86 likes
mean(with_thanksOT[[7]])#Official tweets including 'thanks' avg 44.14 RT
#Thanks does seem to be important for CS tweets, but keep in mind that
#IRTFP tweets are there too. They said thanks to @Nick29T and got 508 likes or something
#So you need to remember there might be outliers present
#Actually, these tweets seem to perform slightly worse in all categories as well

#with_season broken down further:
with_seasonCS = with_season[grep("^@", with_season$Content),]
with_seasonOT = with_season[-grep("^@", with_season$Content),]
mean(with_seasonCS[[6]])#CS tweets including 'season' avg 0.79 likes
mean(with_seasonCS[[7]])#CS tweets including 'season' avg 0.07 RT
mean(with_seasonOT[[6]])#Official tweets including 'season' avg 156.5 likes
mean(with_seasonOT[[7]])#Official tweets including 'season' avg 30.2 RT
#Same for season





