#This file is being created so that I don't have to make changes to 'Generic_Exploration.R' for Disney data
#It will perform the same analysis as Generic_Exploration, but appropriate to Disney data
#Overall, this is program: 15

library(readxl)
#read in the data: Below line of code will need to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/Disney_Dec_1_2020.xlsx")

#Filter out retweets
tweets = data[-grep("^RT @", data$Content),]

#!!!Not part of template.
idx = grepl("^RT @", data$Content)
sum(idx)

#Optional: Filter to only keep tweets marked 'en' (English)
#!!! Don't run the below line unless you mean to! (below line has now been commented out)
#tweets = tweets[which(tweets$Language == 'en'),]

#Plot Retweets vs. likes:
RT_likes = data.frame(tweets["Number of Retweets"], tweets["Number of Likes"])
plot(RT_likes)
#appears linear, but seems that data may contain outliers

#Checking outliers:
library(outliers)
set.seed(1)
outlier(tweets$`Number of Retweets`)
numOuts = scores(tweets$`Number of Retweets`, type = "chisq", prob = 0.999)
sum(numOuts)#25 observations past the 99th percentile, 16 past the 99.9th percentile
#However, I wouldn't consider all those to really be outliers


#Creating a dataset without the 4 extreme outliers:
which.max(tweets$`Number of Retweets`)#row 2625 of Disney data contains largest outlier
tweets2 = tweets[-which(tweets$`Number of Likes` == 221512),]
which.max(tweets2$`Number of Retweets`)#row 2430 of tweets2 contains next most extreme outlier
tweets2 = tweets2[-which(tweets2$`Number of Likes` == 204037),]
which.max(tweets2$`Number of Retweets`)#row 2681 of tweets2 contains next most extreme outlier
tweets2 = tweets2[-which(tweets2$`Number of Likes` == 129722),]
which.max(tweets2$`Number of Retweets`)#row 2048 of tweets2 contains next most extreme outlier
tweets2 = tweets2[-which(tweets2$`Number of Likes` == 101341),]

#Re-plot retweets vs. likes:
RT_likes2 = data.frame(tweets2["Number of Retweets"], tweets2["Number of Likes"])
plot(RT_likes2)


#Examine Disney tweet statistics for all data
#Find overall average for company likes and retweets:
mean(tweets[[6]]) #likes
mean(tweets[[7]]) #retweets

#Examine effect of including a link across all tweets:
all_link = tweets[grep("https://", tweets$Content),]
all_noLink = tweets[-grep("https://", tweets$Content),]

mean(all_link[[6]]) #avg likes across all tweets containing link
mean(all_link[[7]]) #avg RT across all tweets containing link

mean(all_noLink[[6]]) #avg likes across all tweets NOT containing any links
mean(all_noLink[[7]]) #avg RT across all tweets NOT containing any links


#Do the same for Disney data with 4 outliers removed:
mean(tweets2[[6]])
mean(tweets2[[7]])
outRem_allLink = tweets2[grep("https://", tweets2$Content),]
outRem_noLink = tweets2[-grep("https://", tweets2$Content),]
mean(outRem_allLink[[6]])
mean(outRem_allLink[[7]])
mean(outRem_noLink[[6]])
mean(outRem_noLink[[7]])


#Ensure that Disney data contains no true IRT tweets:
all_OT = tweets[-grep("^@", tweets$Content),]
all_IRT = tweets[grep("^@", tweets$Content),]
#Ensure that no official tweets have been wrongfully marked IRT:
wrong_place = all_IRT[which(all_IRT$Author == all_IRT$`In Reply To`),] #locate wrongfully placed official tweets
all_OT = rbind(all_OT, wrong_place) #add the tweet(s) to official category
all_IRT = all_IRT[-which(all_IRT$Author == all_IRT$`In Reply To`),]
#14 of 16 in all_IRT are in reply to @Disney (not true IRT)
#The contents of the remaining two tweets don't appear to be true IRT tweets either

#Ensure that Disney data contains no true IRT tweets using split process created in 'OT_IRTsplit.R':
#Perform initial separation based off "^@":
initIRT = grepl("^@", tweets$Content) #Find all tweets beginning with @username mention, initially mark them all IRT
#initOT = grepl("^(?!@\w+)", tweets$Content)
initOT = grepl("FALSE", initIRT) #Initial official tweets will be the opposite set of tweets from IRT
sum(initIRT)
sum(initOT)

#Create OT and IRT variables in the data:
tweets["OT"] = initOT
tweets["IRT"] = initIRT
tweets["OT"] = tweets["OT"] * 1
tweets["IRT"] = tweets["IRT"] * 1
sum(tweets[[20]])#initial official
sum(tweets[[21]])#initial IRT

#Determine length of data
nr = NROW(tweets)


#################################################################################################################################################

#Replace NA's in the 'In Reply To' field with 'OT'
#library(tidyr)
tweets["In Reply To"][is.na(tweets["In Reply To"])] = "OT"


#Clean up initial separation:
for (i in 1:nrow(tweets)){
  if (tweets$IRT[i] == 1){#if the tweet was initially marked as IRT
    if (tweets$`In Reply To`[i] == tweets$Author[i]){#and the tweet is in reply to @theCompany, not another user
      j = i #then index our current position so that we may examine the chain of 'next' (relative to our data, technically previous) tweets
      while (tweets$`In Reply To`[j] == tweets$Author[j]){
        j = j + 1 #follow the chain until you find a tweet 'in reply to' @anotherUser, or the "OT" string input above this for loop
      }
      if (tweets$`In Reply To`[j] == "OT"){#if following the thread led us up to an official tweet
        tweets$OT[i] = 1 #then this is technically an official tweet
        tweets$IRT[i] = 0 #and not a true IRT tweet, even though it began with @userName mention
      }
    }
  }
}

#Create official and IRT tweet datasets:
all_OT = tweets[which(tweets$OT == 1),]
all_IRT = tweets[which(tweets$IRT == 1),]

############################################################################################################################################
#Basic hashtag analysis:

#Find the row containing the max number of hashtags out of all company tweets
most_hash_row = which.max(tweets$num_hashtags)
#store the max number of hashtags used throughout all company tweets
max_hashes = tweets$num_hashtags[most_hash_row]

for (i in 0:max_hashes) {
  print(i)
  rel_data = tweets[tweets$num_hashtags == i,]
  num_tweets = NROW(rel_data$Content)
  print(paste0("Number of tweets with above amt. of hashtags: ", num_tweets))
  avg_likes = mean(rel_data[[6]])
  avg_RT = mean(rel_data[[7]])
  print(paste0("Average likes: ", avg_likes))
  print(paste0("Average retweets: ", avg_RT))
}

#There's a Disney tweet with 14 hashtags?
hash14 = tweets[which(tweets$num_hashtags == 14),]
#there sure is

#Perform same analysis on Disney data with 4 extreme outliers removed
most_hash_row = which.max(tweets2$num_hashtags)
#store the max number of hashtags used throughout all company tweets
max_hashes = tweets2$num_hashtags[most_hash_row]

for (i in 0:max_hashes) {
  print(i)
  rel_data = tweets2[tweets2$num_hashtags == i,]
  num_tweets = NROW(rel_data$Content)
  print(paste0("Number of tweets with above amt. of hashtags: ", num_tweets))
  avg_likes = mean(rel_data[[6]])
  avg_RT = mean(rel_data[[7]])
  print(paste0("Average likes: ", avg_likes))
  print(paste0("Average retweets: ", avg_RT))
}

















































