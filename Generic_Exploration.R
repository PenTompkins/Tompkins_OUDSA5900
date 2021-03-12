#This file will just be a template for running data through some preliminary examination
#Overall, this is program: 13

library(readxl)

#read in the data: Below line of code will have to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/Amazon_Dec_1_2020.xlsx")

#Filter out retweets: (Note: If the dataset contains no retweets, below line of code will remove all data)
tweets = data[-grep("^RT @", data$Content),]

#!!!Not part of template. If above line of code removes all data, ensure there really were no retweets:
idx = grepl("^RT @", data$Content)
sum(idx) #If there are 0 tweets starting with 'RT @'
#Then tweets = data, run the below line of code to restore the dataset
tweets = data

#Optional: Filter to only keep tweets marked 'en' (English)
#!!! Don't run the below line unless you mean to! (Below line of code has now been commented out, as it wasn't used during this analysis)
#tweets = tweets[which(tweets$Language == 'en'),]

#Plot Retweets vs. likes:
RT_likes = data.frame(tweets["Number of Retweets"], tweets["Number of Likes"])
plot(RT_likes)

#!!If your data has exactly one extreme outlier
#Creating a dataset without the extreme outlier:
which.max(tweets$`Number of Retweets`)#row which contains the outlier
#!!!Change the below amount to be specific to your current data, if you wish to remove the most extreme outlier:
#(Amazon == 10156), (BMW shouldn't have any obs. removed), (CocaCola == 15301)
tweets2 = tweets[-which(tweets$`Number of Likes` == 10156),]

RT_likes2 = data.frame(tweets2["Number of Retweets"], tweets2["Number of Likes"])
plot(RT_likes2)
#!!!NOT PART OF TEMPLATE: Run below line to remove outlier and perform analysis
tweets = tweets2

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


#Not part of template really, ensure that no links are in 'all_noLink'
hidden_links = all_noLink[grep("https", all_noLink$Content),]
#we want hidden_links to have 0 observations

#Split into OT vs. IRT tweet categories:
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

#########################################################################################
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





























































































