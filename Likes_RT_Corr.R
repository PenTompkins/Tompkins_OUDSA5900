#This file is being created to quantify relationship between likes and retweets
#
#Overall, this is program: 48

library(readxl)
library(ggpubr)

#read in the data: Below line of code will have to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/Toyota_Dec_1_2020.xlsx")

#Filter out retweets: (Note: If the dataset contains no retweets, below line of code will remove all data)
tweets = data[-grep("^RT @", data$Content),]

#!!!Not part of template. If above line of code removes all data, ensure there really were no retweets:
idx = grepl("^RT @", data$Content)
sum(idx) #If there are 0 tweets starting with 'RT @'
#Then tweets = data, run the below line of code to restore the dataset
tweets = data

##########################################################################################################
#All Tweets:
#Examine distribution throughout all tweet likes:
#all_likes = tweets$`Number of Likes`

ggdensity(tweets$`Number of Likes`,
          main = "All Tweets: Density Plot of Likes",
          xlab = "Number of Likes")

ggqqplot(tweets$`Number of Likes`)

#Ensure that, throughout all tweets, likes are not normally distributed:
shapiro.test(tweets$`Number of Likes`)

#Examine distribution throughout all tweet retweets:
ggdensity(tweets$`Number of Retweets`,
          main = "All Tweets: Density Plot of Retweets",
          xlab = "Number of Retweets")

ggqqplot(tweets$`Number of Retweets`)

#Ensure that, throughout all tweets, retweets are not normally distributed:
shapiro.test(tweets$`Number of Retweets`)

#If neither are normally distributed, calculate Spearman's correlation:
res = cor.test(tweets$`Number of Retweets`, tweets$`Number of Likes`, method = "spearman", exact = FALSE)
res
#Apparently, cor.test gives tie-corrected coefficient, so supressing warning was unnecessary
#Proof: https://stackoverflow.com/questions/10711395/spearman-correlation-and-ties


##################################################################################################################
#Split Analysis:
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

#Filter to only keep tweets marked 'en' (English)
#tweets = tweets[which(tweets$Language == 'en'),]

#Create official and IRT tweet datasets:
all_OT = tweets[which(tweets$OT == 1),]
all_IRT = tweets[which(tweets$IRT == 1),]


################################################################################################################
#Official Tweets:
#Examine density and qqplots for official tweet likes:
ggdensity(all_OT$`Number of Likes`,
          main = "Official Tweets: Density Plot of Likes",
          xlab = "Number of Likes")

ggqqplot(all_OT$`Number of Likes`)

#Ensure that, throughout official tweets, likes are not normally distributed:
shapiro.test(all_OT$`Number of Likes`)

#Examine distribution throughout official tweet retweets:
ggdensity(all_OT$`Number of Retweets`,
          main = "Official Tweets: Density Plot of Retweets",
          xlab = "Number of Retweets")

ggqqplot(all_OT$`Number of Retweets`)

#Ensure that, throughout official tweets, retweets are not normally distributed:
shapiro.test(all_OT$`Number of Retweets`)

#If neither are normally distributed, calculate Spearman's correlation:
res2 = cor.test(all_OT$`Number of Retweets`, all_OT$`Number of Likes`, method = "spearman", exact = FALSE)
res2



################################################################################################################################
#IRT Tweets:
#Examine density and qqplots for IRT tweet likes:
ggdensity(all_IRT$`Number of Likes`,
          main = "IRT Tweets: Density Plot of Likes",
          xlab = "Number of Likes")

ggqqplot(all_IRT$`Number of Likes`)

#Ensure that, throughout IRT tweets, likes are not normally distributed:
shapiro.test(all_IRT$`Number of Likes`)

#Examine distribution throughout IRT tweet retweets:
ggdensity(all_IRT$`Number of Retweets`,
          main = "IRT Tweets: Density Plot of Retweets",
          xlab = "Number of Retweets")

ggqqplot(all_IRT$`Number of Retweets`)

#Ensure that, throughout IRT tweets, retweets are not normally distributed:
shapiro.test(all_IRT$`Number of Retweets`)

#If neither are normally distributed, calculate Spearman's correlation:
res3 = cor.test(all_IRT$`Number of Retweets`, all_IRT$`Number of Likes`, method = "spearman", exact = FALSE)
res3





















