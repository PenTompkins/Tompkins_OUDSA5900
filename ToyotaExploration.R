#This file is being created to go through the Generic_Exploration process for Toyota data
#Overall, this is program: 21.2 (technically should've been 22, but I'm realizing that way too late)

library(readxl)

#read in the data: Below line of code will need to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/Toyota_Dec_1_2020.xlsx")

#Filter out retweets
tweets = data[-grep("^RT @", data$Content),]

#Optional: Filter to only keep tweets marked 'en' (English)
#!!! Don't run the below line unless you mean to! (below line has now been commented out)
#tweets = tweets[which(tweets$Language == 'en'),]

#Plot Retweets vs. likes:
RT_likes = data.frame(tweets["Number of Retweets"], tweets["Number of Likes"])
plot(RT_likes)

#Remove 3 most extreme outliers on previous plot:
which.max(tweets$`Number of Retweets`)#row 1544 of tweets contains most extreme outlier
tweets2 = tweets[-which(tweets$`Number of Retweets` == 4825),]
which.max(tweets2$`Number of Retweets`)#row 1513 of tweets2 contains 2nd most extreme outlier
tweets2 = tweets2[-which(tweets2$`Number of Retweets` == 4681),]
which.max(tweets2$`Number of Retweets`)#row 1475 of tweets2 contains 3rd most extreme outlier
tweets2 = tweets2[-which(tweets2$`Number of Retweets` == 3086),]

#Replot:
RT_likes2 = data.frame(tweets2["Number of Retweets"], tweets2["Number of Likes"])
plot(RT_likes2)

#Ensure Toyota data has no true IRT tweets:
potential_IRT = tweets[grep("^@", tweets$Content),]
#Grabs 6 tweets, but I checked using Twitter's advanced search and they're all actually official tweets
#weird that data collection indicated that these were 'in reply to' the @username mention when it didn't normally do
#that for official tweets beginning with an "@username" mention. Usually began ".@username"

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









































































