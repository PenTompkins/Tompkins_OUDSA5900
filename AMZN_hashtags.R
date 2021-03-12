#This script is to examine the effect that num_hashtags has on likes and RT for Amazon data
#Overall, this is program: 7

library(readxl)
#Read in the Amazon data from excel: Below line of code will need to be reconfigured for your filepath
amazon_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/Amazon_Dec_1_2020.xlsx")

#Remove Amazon's retweets from the data
amazon_tweets = amazon_data[-grep("^RT @", amazon_data$Content),]

#Determine the max number of hashtags contained throughout all tweets:
which.max(amazon_tweets$num_hashtags)
#Row 1022 contains 3 hashtags
#However, it seems to only really contain 2...they use '#1' in place of 'number one'

zero_hash = amazon_tweets[amazon_tweets$num_hashtags == 0,]
one_hash = amazon_tweets[amazon_tweets$num_hashtags == 1,]
two_hash = amazon_tweets[amazon_tweets$num_hashtags == 2,]
three_hash = amazon_tweets[amazon_tweets$num_hashtags == 3,]
#Appears that there was one tweet with 3 legit hashtags

#I'm moving the tweet with '#1' from group 3 to group 2
two_hash = rbind(two_hash, three_hash[three_hash$`Number of Likes` == 310,])
three_hash = three_hash[-(three_hash$`Number of Likes` == 310),]

#Without separating OT and IRT, examine mean likes/RT of different hash groups:
mean(zero_hash[[6]])#tweets w/ 0 hashtags avg 27.88 likes
mean(zero_hash[[7]])#tweets w/ 0 hashtags avg 4.58 RT
mean(one_hash[[6]]) #tweets w/ 1 hashtag avg 37.73 likes
mean(one_hash[[7]]) #tweets w/ 1 hashtag avg 6.61 RT
mean(two_hash[[6]]) #tweets w/ 2 hashtags avg 191.82 likes
mean(two_hash[[7]]) #tweets w/ 2 hashtags avg 36.91 RT
mean(three_hash[[6]]) #the tweet w/ 3 hashtags got 148 likes
mean(three_hash[[7]]) #the tweet w/ 3 hashtags got 13 RT

#Time to separate OT from IRT:
zero_hashOT = zero_hash[-grep("^@", zero_hash$Content),]
zero_hashIRT = zero_hash[grep("^@", zero_hash$Content),]
one_hashOT = one_hash[-grep("^@", one_hash$Content),]
one_hashIRT = one_hash[grep("^@", one_hash$Content),]
two_hashOT = two_hash[-grep("^@", two_hash$Content),]
two_hashIRT = two_hash[grep("^@", two_hash$Content),]
#no need to do it for the one, 3 hashtag tweet

#Examine effects again:
mean(zero_hashOT[[6]]) #OT tweets with 0 hash avg 446.89 likes
mean(zero_hashOT[[7]]) #OT tweets with 0 hash avg 71.82 RT
mean(zero_hashIRT[[6]]) #CS tweets with 0 hash avg 3.23 likes
mean(zero_hashIRT[[7]]) #CS tweets with 0 hash avg 0.62 RT
mean(one_hashOT[[6]]) #OT tweets with 1 hash avg 241.84 likes
mean(one_hashOT[[7]]) #OT tweets with 1 hash avg 44.62 RT
mean(one_hashIRT[[6]]) #CS tweets with 1 hash avg 4.34 likes
mean(one_hashIRT[[7]]) #CS tweets with 1 hash avg 0.39 RT
mean(two_hashOT[[6]]) #OT tweets with 2 hash avg 210.8 likes
mean(two_hashOT[[7]]) #OT tweets with 2 hash avg 40.6 likes
mean(two_hashIRT[[6]]) #the CS tweet with 2 hash had 2 likes
mean(two_hashIRT[[7]]) #the CS tweet with 2 hash had 0 RT




