#This file is being created to run BTM topic effect val. process over Amazon OT tweets
#
#Overall, this is program: 80

library(readxl)
library(ggpubr)
library(fpp2)
library(geoR)
library(ggplot2)
library(rogme)
library(dplyr)
library(FSA)
library(tibble)
library(gridExtra)

set.seed(1)

#read in the data: Below line of code will have to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Pictures/BTMresults_OT/Amazon_BTMresults_OT.xlsx")

#No need to separate OT/IRT, already done in data
tweets = data
#Official Tweet Likes Analysis:
##################################################################################################################
#Regular:
#Create density plot of company OT likes:
ggdensity(tweets$`Number of Likes`,
          main = "Amazon OT: Likes",
          xlab = "Number of Likes")

#Log(+0.0001)
#Create density plot of company OT log(likes):
tweets$logLikes = log(tweets$`Number of Likes` + 0.0001)

ggdensity(tweets$logLikes,
          main = "Amazon OT: log(Likes)",
          xlab = "log(Likes)")

#Examine whether this log distribution may be considered normal
shapiro.test(tweets$logLikes)

#Ensure that topic is considered a factor, not a value:
tweets$topic = factor(tweets$topic)

#Create visualization for Topic effect on likes across all tweets:
ggboxplot(tweets, x = "topic", y = "Number of Likes",
          color = "topic", ylab = "Number of Likes", xlab = "Topics")


#Create visualization for Topic effect on log(likes+.0001) across all tweets:
ggboxplot(tweets, x = "topic", y = "logLikes",
          color = "topic", ylab = "Log(Likes)", xlab = "Topics")

#Perform Kruskal-Wallis test, for likes, grouped by topic:
kruskal.test(`Number of Likes` ~ topic, data = tweets)


#If above results were significant, perform Dunn's test for further examination:
dunnTest(`Number of Likes` ~ topic, data = tweets, method = "bonferroni")

#For Amazon, likes analysis ends here. No differences found in Dunn's test


#RETWEET ANALYSIS BELOW
############################################################################################################
#Regular:
#Create density plot of company OT Retweets:
ggdensity(tweets$`Number of Retweets`,
          main = "Amazon OT: Retweets",
          xlab = "Number of Retweets")

#Log(+0.0001)
#Create density plot of company OT log(Retweets):
tweets$logRetweets = log(tweets$`Number of Retweets` + 0.0001)

ggdensity(tweets$logRetweets,
          main = "Amazon OT: log(Retweets)",
          xlab = "log(Retweets)")

#Examine whether this log distribution may be considered normal
shapiro.test(tweets$logRetweets)

#Ensure that topic is considered a factor, not a value:
#tweets$topic = factor(tweets$topic)

#Create visualization for Topic effect on Retweets across all tweets:
ggboxplot(tweets, x = "topic", y = "Number of Retweets",
          color = "topic", ylab = "Number of Retweets", xlab = "Topics")


#Create visualization for Sentiment effect on log(Retweets+.0001) across all tweets:
ggboxplot(tweets, x = "topic", y = "logRetweets",
          color = "topic", ylab = "Log(Retweets)", xlab = "Topics")

#Perform Kruskal-Wallis test, for Retweets, grouped by topic:
kruskal.test(`Number of Retweets` ~ topic, data = tweets)


#If above results were significant, perform Dunn's test for further examination:
dunnTest(`Number of Retweets` ~ topic, data = tweets, method = "bonferroni")










































