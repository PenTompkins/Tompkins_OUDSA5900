#This file is being created to go throug the Generic_Exploration process for Microsoft data
#Overall, this is program: 20

library(readxl)

#read in the data: Below line of code will need to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/Microsoft_Dec_1_2020.xlsx")

#Filter out retweets
tweets = data[-grep("^RT @", data$Content),]

#Optional: Filter to only keep tweets marked 'en' (English)
#!!! Don't run the below line unless you mean to! (below line has now been commented out)
#tweets = tweets[which(tweets$Language == 'en'),]

#Plot Retweets vs. likes:
RT_likes = data.frame(tweets["Number of Retweets"], tweets["Number of Likes"])
plot(RT_likes)

#Removing the two most extreme outliers from Microsoft data:
which.max(tweets$`Number of Retweets`)#row 1295 contains most extreme outlier for Microsoft data
#!!!Change the below amount to be specific to your current data: (60229 should be correct for Microsoft)
tweets2 = tweets[-which(tweets$`Number of Likes` == 60229),]
#Identify second most extreme outlier (in terms of retweets):
which.max(tweets2$`Number of Retweets`)#row 710 of tweets2 contains 2nd most extreme outlier
tweets2 = tweets2[-which(tweets2$`Number of Retweets` == 7805),]

#Replot retweets vs. likes:
RT_likes2 = data.frame(tweets2["Number of Retweets"], tweets2["Number of Likes"])
plot(RT_likes2)


#######################
#Split based on OT vs. IRT and replot (with outliers included):
#Perform initial separation based off "^@":
initIRT = grepl("^@", tweets$Content) #Find all tweets beginning with @username mention, initially mark them all IRT
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

#Replot retweets vs. likes:
iRT_likes = data.frame(all_IRT["Number of Retweets"], all_IRT["Number of Likes"])
plot(iRT_likes)

OTrt_likes = data.frame(all_OT["Number of Retweets"], all_OT["Number of Likes"])
plot(OTrt_likes)


############################
#Basic tweet statistics (with outliers included):
mean(tweets[[6]])#avg likes across all tweets
mean(tweets[[7]])#avg RT across all tweets

#Examine effect of link with extreme outliers included:
all_link = tweets[grep("https://", tweets$Content),]
all_noLink = tweets[-grep("https://", tweets$Content),]

mean(all_link[[6]]) #avg likes across all tweets containing link
mean(all_link[[7]]) #avg RT across all tweets containing link

mean(all_noLink[[6]]) #avg likes across all tweets NOT containing any links
mean(all_noLink[[7]]) #avg RT across all tweets NOT containing any links


###########################
#Do the same without 3 extreme outliers
#However, to do that we must first identify the most extreme IRT outlier
#Which can most easily be done after splitting the outliers removed data set into OT vs. IRT
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

#Create official and IRT tweet datasets (with two most extreme outliers removed):
all_OT2 = tweets2[which(tweets2$OT == 1),]
all_IRT2 = tweets2[which(tweets2$IRT == 1),]

#Identify the most extreme IRT outlier (3rd outlier to be removed):
which.max(all_IRT2$`Number of Retweets`)#row 1342 of all_IRT2 contains extreme outlier for IRT tweets
#remove the observation from all_IRT2
all_IRT2 = all_IRT2[-which(all_IRT2$`Number of Retweets` == 523),]

#Remove the observation from tweets2:
tweets2 = tweets2[-which(tweets2$`Number of Retweets` == 523 & tweets2$`Number of Likes` == 12460),]
#tweets2 now has the 3 most extreme outliers removed (2 official and 1 IRT)

#Basic tweet statistics when 3 outliers are removed:
mean(tweets2[[6]])
mean(tweets2[[7]])

outRem_allLink = tweets2[grep("https://", tweets2$Content),]
outRem_noLink = tweets2[-grep("https://", tweets2$Content),]
mean(outRem_allLink[[6]])
mean(outRem_allLink[[7]])
mean(outRem_noLink[[6]])
mean(outRem_noLink[[7]])



#####################
#Examine basic tweet statistics when split into OT vs. IRT (with outliers included):
mean(all_OT[[6]])
mean(all_OT[[7]])

mean(all_IRT[[6]])
mean(all_IRT[[7]])

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


#####################
#Examine basic tweet statistics when split into OT vs. IRT (with outliers removed):
mean(all_OT2[[6]])
mean(all_OT2[[7]])

mean(all_IRT2[[6]])
mean(all_IRT2[[7]])

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
#Basic hashtag analysis performed below: (outliers included)

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
#Basic hashtag analysis when 3 extreme outliers are removed performed below:

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




































































