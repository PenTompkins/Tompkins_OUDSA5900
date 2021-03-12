#This file is being created to go through the Generic_Exploration process for Samsung data
#Overall, this is program: 21


library(readxl)

#read in the data: Below line of code will need to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/Samsung_Dec_1_2020.xlsx")

#Filter out retweets
tweets = data[-grep("^RT @", data$Content),]

#Optional: Filter to only keep tweets marked 'en' (English)
#!!! Don't run the below line unless you mean to! (below line of code has now been commented out)
#tweets = tweets[which(tweets$Language == 'en'),]

#Plot Retweets vs. likes:
RT_likes = data.frame(tweets["Number of Retweets"], tweets["Number of Likes"])
plot(RT_likes)


#Remove 4 right-most observations on previous plot:
which.max(tweets$`Number of Retweets`)#row 199 of tweets contains right-most observation
tweets2 = tweets[-which(tweets$`Number of Retweets` == 7011),]
which.max(tweets2$`Number of Retweets`)#row 38 of tweets2 contains 2nd most extreme outlier
tweets2 = tweets2[-which(tweets2$`Number of Retweets` == 6354),]
which.max(tweets2$`Number of Retweets`)#row 88 of tweets2 contains 3rd most extreme outlier
tweets2 = tweets2[-which(tweets2$`Number of Retweets` == 5560),]
which.max(tweets2$`Number of Retweets`)#row 162 of tweets2 contains 4th most extreme outlier
tweets2 = tweets2[-which(tweets2$`Number of Retweets` == 3104),]

#Replot:
RT_likes2 = data.frame(tweets2["Number of Retweets"], tweets2["Number of Likes"])
plot(RT_likes2)


##########################
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


##############################
#Basic tweet statistics (outliers removed):
mean(tweets2[[6]])
mean(tweets2[[7]])

outRem_allLink = tweets2[grep("https://", tweets2$Content),]
outRem_noLink = tweets2[-grep("https://", tweets2$Content),]
mean(outRem_allLink[[6]])
mean(outRem_allLink[[7]])
mean(outRem_noLink[[6]])
mean(outRem_noLink[[7]])

###
#Ensure that Samsung data truly contains no IRT tweets:
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


################################################################################################################################################
#Basic hashtag analysis performed below: (outliers excluded)

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

































































































