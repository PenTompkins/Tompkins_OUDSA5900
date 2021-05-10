#This file is being created to run BTM topic effect val. process over McDonalds IRT tweets
#
#Overall, this is program: 81

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
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Pictures/BTMresults_IRT/McDonalds_BTMresults_IRT.xlsx")

#No need to separate OT/IRT, already done in data
tweets = data

#IRT Tweet Likes Analysis:
##################################################################################################################
#Regular:
#Create density plot of company IRT likes:
ggdensity(tweets$`Number of Likes`,
          main = "McDonalds IRT: Likes",
          xlab = "Number of Likes")

#Log(+0.0001)
#Create density plot of company IRT log(likes):
tweets$logLikes = log(tweets$`Number of Likes` + 0.0001)

ggdensity(tweets$logLikes,
          main = "McDonalds IRT: log(Likes)",
          xlab = "log(Likes)")

#Examine whether this log distribution may be considered normal
#shapiro.test(tweets$logLikes)

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


#Perform shift function:
top0 = tweets[which(tweets$topic == 0),]
top1 = tweets[which(tweets$topic == 1),]
top2 = tweets[which(tweets$topic == 2),]
top3 = tweets[which(tweets$topic == 3),]
top4 = tweets[which(tweets$topic == 4),]
top5 = tweets[which(tweets$topic == 5),]
nt0 = nrow(top0)
nt1 = nrow(top1)
nt2 = nrow(top2)
nt3 = nrow(top3)
nt4 = nrow(top4)
nt5 = nrow(top5)

df = tibble(gr = factor(c(rep("Zero", nt0), rep("One", nt1), rep("Two", nt2), rep("Three", nt3),
                          rep("Four", nt4), rep("Five", nt5))),
            obs = c(top0$`Number of Likes`, top1$`Number of Likes`, top2$`Number of Likes`,
                    top3$`Number of Likes`, top4$`Number of Likes`, top5$`Number of Likes`))

sf = shifthd_pbci(data = df, formula = obs ~gr, doall = TRUE)
sf

###################
#Again, on log-transformed data
df = tibble(gr = factor(c(rep("Zero", nt0), rep("One", nt1), rep("Two", nt2), rep("Three", nt3),
                          rep("Four", nt4), rep("Five", nt5))),
            obs = c(top0$logLikes, top1$logLikes, top2$logLikes,
                    top3$logLikes, top4$logLikes, top5$logLikes))

sf = shifthd_pbci(data = df, formula = obs ~ gr, doall = TRUE)
#Add more colors here, rather than below:
palette1 = c("violetred2", "steelblue2", "springgreen1", "slategray3")
psf = plot_sf(sf)
#Add labels for deciles 1 and 9:
psf = add_sf_lab(psf, sf, y_lab_nudge = .1, text_size = 2)
#do.call("grid.arrange", c(psf, ncol=2))
psf[[1]]
psf[[2]]
psf[[3]]

#Create 1D scatterplots with color coded differences:
p = plot_scat2(df, alpha = 0.3, shape = 21)
p
p = p + coord_flip()
p


#RETWEET ANALYSIS BELOW:
##########################################################################################################
#Regular:
#Create density plot of company IRT Retweets:
ggdensity(tweets$`Number of Retweets`,
          main = "McDonalds IRT: Retweets",
          xlab = "Number of Retweets")

#Log(+0.0001)
#Create density plot of company IRT log(Retweets):
tweets$logRetweets = log(tweets$`Number of Retweets` + 0.0001)

ggdensity(tweets$logRetweets,
          main = "McDonalds IRT: log(Retweets)",
          xlab = "log(Retweets)")

#Examine whether this log distribution may be considered normal
#shapiro.test(tweets$logRetweets)

#Ensure that topic is considered a factor, not a value:
#tweets$topic = factor(tweets$topic)

#Create visualization for Topic effect on Retweets across all tweets:
ggboxplot(tweets, x = "topic", y = "Number of Retweets",
          color = "topic", ylab = "Number of Retweets", xlab = "Topics")


#Create visualization for Topic effect on log(Retweets+.0001) across all tweets:
ggboxplot(tweets, x = "topic", y = "logRetweets",
          color = "topic", ylab = "Log(Retweets)", xlab = "Topics")

#Perform Kruskal-Wallis test, for Retweets, grouped by topic:
kruskal.test(`Number of Retweets` ~ topic, data = tweets)


#If above results were significant, perform Dunn's test for further examination:
dunnTest(`Number of Retweets` ~ topic, data = tweets, method = "bonferroni")

#Perform shift function:
#Reset these, so that they contain 'logRetweets' variable
top0 = tweets[which(tweets$topic == 0),]
top1 = tweets[which(tweets$topic == 1),]
top2 = tweets[which(tweets$topic == 2),]
top3 = tweets[which(tweets$topic == 3),]
top4 = tweets[which(tweets$topic == 4),]
top5 = tweets[which(tweets$topic == 5),]


df = tibble(gr = factor(c(rep("Zero", nt0), rep("One", nt1), rep("Two", nt2), rep("Three", nt3),
                          rep("Four", nt4), rep("Five", nt5))),
            obs = c(top0$`Number of Retweets`, top1$`Number of Retweets`, top2$`Number of Retweets`,
                    top3$`Number of Retweets`, top4$`Number of Retweets`, top5$`Number of Retweets`))

sf = shifthd_pbci(data = df, formula = obs ~gr, doall = TRUE)
sf

###################
#Again, on log-transformed data
df = tibble(gr = factor(c(rep("Zero", nt0), rep("One", nt1), rep("Two", nt2), rep("Three", nt3),
                          rep("Four", nt4), rep("Five", nt5))),
            obs = c(top0$logRetweets, top1$logRetweets, top2$logRetweets,
                    top3$logRetweets, top4$logRetweets, top5$logRetweets))

sf = shifthd_pbci(data = df, formula = obs ~ gr, doall = TRUE)
#Add more colors here, rather than below:
palette1 = c("violetred2", "steelblue2", "springgreen1", "slategray3")
psf = plot_sf(sf)
#Add labels for deciles 1 and 9:
psf = add_sf_lab(psf, sf, y_lab_nudge = .1, text_size = 2)
#do.call("grid.arrange", c(psf, ncol=2))
psf[[1]]
psf[[2]]
psf[[3]]

#Create 1D scatterplots with color coded differences:
p = plot_scat2(df, alpha = 0.3, shape = 21)
p
p = p + coord_flip()
p












