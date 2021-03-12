#This file is being created to go through the 'Generic_Exploration' process for Mercedes data
#Overall, this is program: 19

library(readxl)

#read in the data: Below line of code will need to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/MercedesBenz_Dec_1_2020.xlsx")

#Filter out retweets
tweets = data[-grep("^RT @", data$Content),]

#Optional: Filter to only keep tweets marked 'en' (English)
#!!! Don't run the below line unless you mean to! (below line of code has now been commented out)
#tweets = tweets[which(tweets$Language == 'en'),]

#Plot Retweets vs. likes:
RT_likes = data.frame(tweets["Number of Retweets"], tweets["Number of Likes"])
plot(RT_likes)

#Mercedes data seems to have one extreme outlier
#Creating a dataset without the extreme outlier:
which.max(tweets$`Number of Retweets`)#row 2656 contains most extreme outlier for Mercedes data
#!!!Change the below amount to be specific to your current data: (15301 should be correct for Mercedes data)
tweets2 = tweets[-which(tweets$`Number of Likes` == 15301),]

RT_likes2 = data.frame(tweets2["Number of Retweets"], tweets2["Number of Likes"])
plot(RT_likes2)

#Examine basic tweet statistics when outlier is included:
mean(tweets[[6]])#avg likes across all tweets
mean(tweets[[7]])#avg RT across all tweets

#Examine effect of link with extreme outlier included:
all_link = tweets[grep("https://", tweets$Content),]
all_noLink = tweets[-grep("https://", tweets$Content),]

mean(all_link[[6]]) #avg likes across all tweets containing link
mean(all_link[[7]]) #avg RT across all tweets containing link

mean(all_noLink[[6]]) #avg likes across all tweets NOT containing any links
mean(all_noLink[[7]]) #avg RT across all tweets NOT containing any links



#Examine same statistics when extreme outlier is removed:
mean(tweets2[[6]])
mean(tweets2[[7]])
outRem_allLink = tweets2[grep("https://", tweets2$Content),]
outRem_noLink = tweets2[-grep("https://", tweets2$Content),]
mean(outRem_allLink[[6]])
mean(outRem_allLink[[7]])
mean(outRem_noLink[[6]])
mean(outRem_noLink[[7]])



#Perform same analysis when split into OT vs. IRT (extreme outlier included):
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


mean(all_OT[[6]])#avg likes across all official tweets
mean(all_OT[[7]])#avg RT across all official tweets

mean(all_IRT[[6]])#avg likes across all IRT tweets
mean(all_IRT[[7]])#avg RT across all IRT tweets


#Examine effect of link within OT and IRT categories separately:
OT_links = all_OT[grep("https://", all_OT$Content),]
OT_noLinks = all_OT[-grep("https://", all_OT$Content),]
IRT_links = all_IRT[grep("https://", all_IRT$Content),]
IRT_noLinks = all_IRT[-grep("https://", all_IRT$Content),]

mean(OT_links[[6]])#avg likes of official tweets containing links
mean(OT_links[[7]])#avg RT of official tweets containing links

mean(OT_noLinks[[6]])#avg likes of official tweets NOT containing links
mean(OT_noLinks[[7]])#avg RT of official tweets NOT containing links

mean(IRT_links[[6]])#avg likes of IRT tweets containing links
mean(IRT_links[[7]])#avg RT of IRT tweets containing links

mean(IRT_noLinks[[6]])#avg likes of IRT tweets NOT containing links
mean(IRT_noLinks[[7]])#avg RT of IRT tweets NOT containing links




#Perform OT vs. IRT analysis when extreme outlier is removed:
#Perform initial separation based off "^@":
initIRT2 = grepl("^@", tweets2$Content) #Find all tweets beginning with @username mention, initially mark them all IRT
initOT2 = grepl("FALSE", initIRT2) #Initial official tweets will be the opposite set of tweets from IRT
sum(initIRT2)
sum(initOT2)

#Create OT and IRT variables in the data:
tweets2["OT"] = initOT2
tweets2["IRT"] = initIRT2
tweets2["OT"] = tweets2["OT"] * 1
tweets2["IRT"] = tweets2["IRT"] * 1
sum(tweets2[[20]])#initial official
sum(tweets2[[21]])#initial IRT


#Replace NA's in the 'In Reply To' field with 'OT'
#library(tidyr)
tweets2["In Reply To"][is.na(tweets2["In Reply To"])] = "OT"


#Clean up initial separation:
for (i in 1:nrow(tweets2)){
  if (tweets2$IRT[i] == 1){#if the tweet was initially marked as IRT
    if (tweets2$`In Reply To`[i] == tweets2$Author[i]){#and the tweet is in reply to @theCompany, not another user
      j = i #then index our current position so that we may examine the chain of 'next' (relative to our data, technically previous) tweets
      while (tweets2$`In Reply To`[j] == tweets2$Author[j]){
        j = j + 1 #follow the chain until you find a tweet 'in reply to' @anotherUser, or the "OT" string input above this for loop
      }
      if (tweets2$`In Reply To`[j] == "OT"){#if following the thread led us up to an official tweet
        tweets2$OT[i] = 1 #then this is technically an official tweet
        tweets2$IRT[i] = 0 #and not a true IRT tweet, even though it began with @userName mention
      }
    }
  }
}

#Create official and IRT tweet datasets:
all_OT2 = tweets2[which(tweets2$OT == 1),]
all_IRT2 = tweets2[which(tweets2$IRT == 1),]



mean(all_OT2[[6]])#avg likes across all official tweets
mean(all_OT2[[7]])#avg RT across all official tweets

mean(all_IRT2[[6]])#avg likes across all IRT tweets
mean(all_IRT2[[7]])#avg RT across all IRT tweets


#Examine effect of link within OT and IRT categories separately:
OT_links2 = all_OT2[grep("https://", all_OT2$Content),]
OT_noLinks2 = all_OT2[-grep("https://", all_OT2$Content),]
IRT_links2 = all_IRT2[grep("https://", all_IRT2$Content),]
IRT_noLinks2 = all_IRT2[-grep("https://", all_IRT2$Content),]

mean(OT_links2[[6]])#avg likes of official tweets containing links
mean(OT_links2[[7]])#avg RT of official tweets containing links

mean(OT_noLinks2[[6]])#avg likes of official tweets NOT containing links
mean(OT_noLinks2[[7]])#avg RT of official tweets NOT containing links

mean(IRT_links2[[6]])#avg likes of IRT tweets containing links
mean(IRT_links2[[7]])#avg RT of IRT tweets containing links

mean(IRT_noLinks2[[6]])#avg likes of IRT tweets NOT containing links
mean(IRT_noLinks2[[7]])#avg RT of IRT tweets NOT containing links


################################################################################################################################################
#Basic hashtag analysis performed below:

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




#Perform same analysis over official tweets only:
most_hash_row = which.max(all_OT$num_hashtags)
max_hashes = all_OT$num_hashtags[most_hash_row]

for (i in 0:max_hashes) {
  print(i)
  rel_data = all_OT[all_OT$num_hashtags == i,]
  num_tweets = NROW(rel_data$Content)
  print(paste0("Number of official tweets with above amt. of hashtags: ", num_tweets))
  avg_likes = mean(rel_data[[6]])
  avg_RT = mean(rel_data[[7]])
  print(paste0("Average likes: ", avg_likes))
  print(paste0("Average retweets: ", avg_RT))
}



#Perform same analysis over IRT tweets only:
most_hash_row = which.max(all_IRT$num_hashtags)
max_hashes = all_IRT$num_hashtags[most_hash_row]

for (i in 0:max_hashes) {
  print(i)
  rel_data = all_IRT[all_IRT$num_hashtags == i,]
  num_tweets = NROW(rel_data$Content)
  print(paste0("Number of IRT tweets with above amt. of hashtags: ", num_tweets))
  avg_likes = mean(rel_data[[6]])
  avg_RT = mean(rel_data[[7]])
  print(paste0("Average likes: ", avg_likes))
  print(paste0("Average retweets: ", avg_RT))
}


################################################################################################################################################
#Basic hashtag analysis when extreme outlier is removed performed below:

#Find the row containing the max number of hashtags out of all company tweets
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




#Perform same analysis over official tweets only:
most_hash_row = which.max(all_OT2$num_hashtags)
max_hashes = all_OT2$num_hashtags[most_hash_row]

for (i in 0:max_hashes) {
  print(i)
  rel_data = all_OT2[all_OT2$num_hashtags == i,]
  num_tweets = NROW(rel_data$Content)
  print(paste0("Number of official tweets with above amt. of hashtags: ", num_tweets))
  avg_likes = mean(rel_data[[6]])
  avg_RT = mean(rel_data[[7]])
  print(paste0("Average likes: ", avg_likes))
  print(paste0("Average retweets: ", avg_RT))
}



#Perform same analysis over IRT tweets only:
most_hash_row = which.max(all_IRT2$num_hashtags)
max_hashes = all_IRT2$num_hashtags[most_hash_row]

for (i in 0:max_hashes) {
  print(i)
  rel_data = all_IRT2[all_IRT2$num_hashtags == i,]
  num_tweets = NROW(rel_data$Content)
  print(paste0("Number of IRT tweets with above amt. of hashtags: ", num_tweets))
  avg_likes = mean(rel_data[[6]])
  avg_RT = mean(rel_data[[7]])
  print(paste0("Average likes: ", avg_likes))
  print(paste0("Average retweets: ", avg_RT))
}

################################################################################################################################################

#Might as well replot retweets vs. likes after splitting OT vs. IRT:
iRT_likes = data.frame(all_IRT["Number of Retweets"], all_IRT["Number of Likes"])
plot(iRT_likes)

OTrt_likes = data.frame(all_OT["Number of Retweets"], all_OT["Number of Likes"])
plot(OTrt_likes)























































































