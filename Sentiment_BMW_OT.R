#This file is being created to run sentiment effect val. process over BMW OT tweets
#
#Overall, this is program: 78

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
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/OutlierFree_data/withSentiment/BMW_outsRem_ws.xlsx")

#Extract official tweets only:
tweets = data[which(data$OT == TRUE),]


#Official Tweet Likes Analysis:
##################################################################################################################
#Regular:
#Create density plot of company OT likes:
ggdensity(tweets$`Number of Likes`,
          main = "BMW OT: Likes",
          xlab = "Number of Likes")

#Log(+0.0001)
#Create density plot of company OT log(likes):
tweets$logLikes = log(tweets$`Number of Likes` + 0.0001)

ggdensity(tweets$logLikes,
          main = "BMW OT: log(Likes)",
          xlab = "log(Likes)")

#Examine whether this log distribution may be considered normal
shapiro.test(tweets$logLikes)

#Create sentiment categories in data:
sent_cat = ifelse(tweets$Sentiment > 0, "Positive",
                  ifelse(tweets$Sentiment == 0, "Neutral","Negative"))
tweets["sent_cat"] = sent_cat



#Create visualization for Sentiment effect on likes across all tweets:
ggboxplot(tweets, x = "sent_cat", y = "Number of Likes",
          color = "sent_cat", ylab = "Number of Likes", xlab = "Sentiment Classification")


#Create visualization for Sentiment effect on log(likes+.0001) across all tweets:
ggboxplot(tweets, x = "sent_cat", y = "logLikes",
          color = "sent_cat", ylab = "Log(Likes)", xlab = "Sentiment Classification")

#Separate tweets based on sentiment classification:
pos_tweets = tweets[which(tweets$sent_cat == "Positive"),]
neut_tweets = tweets[which(tweets$sent_cat == "Neutral"),]
neg_tweets = tweets[which(tweets$sent_cat == "Negative"),]


#Visualize dist. of likes for Positive tweets:
ggdensity(pos_tweets$`Number of Likes`,
          main = "BMW Positive: Likes",
          xlab = "Number of Likes") #+ xlim(0, max(neg_tweets$`Number of Likes`))

#Visualize dist. of likes for Neutral tweets:
ggdensity(neut_tweets$`Number of Likes`,
          main = "BMW Neutral: Likes",
          xlab = "Number of Likes") #+ xlim(0, max(neg_tweets$`Number of Likes`))

#Visualize dist. of likes for Negative tweets:
ggdensity(neg_tweets$`Number of Likes`,
          main = "BMW Negative: Likes",
          xlab = "Number of Likes") + xlim(0, max(pos_tweets$`Number of Likes`))



#Visualize dist. of log(likes) for Positive tweets:
ggdensity(pos_tweets$logLikes,
          main = "BMW Positive: log(Likes)",
          xlab = "log(Likes)") #+ xlim(min(pos_tweets$logLikes), max(neg_tweets$logLikes))

#Visualize dist. of log(likes) for Neutral tweets:
ggdensity(neut_tweets$logLikes,
          main = "BMW Neutral: log(Likes)",
          xlab = "log(Likes)") #+ xlim(min(neut_tweets$logLikes), max(neg_tweets$logLikes))

#Visualize dist. of log(likes) for Negative tweets:
ggdensity(neg_tweets$logLikes,
          main = "BMW Negative: log(Likes)",
          xlab = "log(Likes)") + xlim(min(pos_tweets$logLikes), max(pos_tweets$logLikes))

#See whether any of the above log distributions may be considered normal:
shapiro.test(pos_tweets$logLikes)
shapiro.test(neut_tweets$logLikes)
shapiro.test(neg_tweets$logLikes)


#Perform Kruskal-Wallis test, for likes, grouped by sentiment:
kruskal.test(`Number of Likes` ~ sent_cat, data = tweets)


#If above results were significant, perform Dunn's test for further examination:
#dunnTest(`Number of Likes` ~ factor(sent_cat), data = tweets, method = "bonferroni")

#Perform shift function:
#npos = nrow(pos_tweets)
#nneut = nrow(neut_tweets)
#nneg = nrow(neg_tweets)

#df = tibble(gr = factor(c(rep("Positive", npos), rep("Neutral", nneut), rep("Negative", nneg))),
#            obs = c(pos_tweets$`Number of Likes`, neut_tweets$`Number of Likes`, neg_tweets$`Number of Likes`))

#sf = shifthd_pbci(data = df, formula = obs ~gr, doall = TRUE)
#sf

###################
#Again, on log-transformed data
#df = tibble(gr = factor(c(rep("Positive", npos), rep("Neutral", nneut), rep("Negative", nneg))),
#            obs = c(pos_tweets$logLikes, neut_tweets$logLikes, neg_tweets$logLikes))

#sf = shifthd_pbci(data = df, formula = obs ~gr, doall = TRUE)
#Add more colors here, rather than below:
#palette1 = c("violetred2", "steelblue2", "springgreen1", "slategray3")
#psf = plot_sf(sf)
#Add labels for deciles 1 and 9:
#psf = add_sf_lab(psf, sf, y_lab_nudge = .1, text_size = 2)
#do.call("grid.arrange", c(psf, ncol=2))
#psf[[1]]
#psf[[2]]
#psf[[3]]

#Create 1D scatterplots with color coded differences:
#p = plot_scat2(df, alpha = 0.3, shape = 21)
#p
#p = p + coord_flip()
#p


#RETWEET ANALYSIS PERFORMED BELOW
##############################################################################################################
#Regular:
#Create density plot of company OT Retweets:
ggdensity(tweets$`Number of Retweets`,
          main = "BMW OT: Retweets",
          xlab = "Number of Retweets")

#Log(+0.0001)
#Create density plot of company OT log(Retweets):
tweets$logRetweets = log(tweets$`Number of Retweets` + 0.0001)

ggdensity(tweets$logRetweets,
          main = "BMW OT: log(Retweets)",
          xlab = "log(Retweets)")

#Examine whether this log distribution may be considered normal
shapiro.test(tweets$logRetweets)




#Create visualization for Sentiment effect on Retweets across all tweets:
ggboxplot(tweets, x = "sent_cat", y = "Number of Retweets",
          color = "sent_cat", ylab = "Number of Retweets", xlab = "Sentiment Classification")


#Create visualization for Sentiment effect on log(Retweets+.0001) across all tweets:
ggboxplot(tweets, x = "sent_cat", y = "logRetweets",
          color = "sent_cat", ylab = "Log(Retweets)", xlab = "Sentiment Classification")

#Reset these, such that they contain log(Retweets) variable:
pos_tweets = tweets[which(tweets$sent_cat == "Positive"),]
neut_tweets = tweets[which(tweets$sent_cat == "Neutral"),]
neg_tweets = tweets[which(tweets$sent_cat == "Negative"),]


#Visualize dist. of Retweets for Positive tweets:
ggdensity(pos_tweets$`Number of Retweets`,
          main = "BMW Positive: Retweets",
          xlab = "Number of Retweets") + xlim(0, max(neut_tweets$`Number of Retweets`))

#Visualize dist. of Retweets for Neutral tweets:
ggdensity(neut_tweets$`Number of Retweets`,
          main = "BMW Neutral: Retweets",
          xlab = "Number of Retweets") #+ xlim(0, max(neg_tweets$`Number of Retweets`))

#Visualize dist. of Retweets for Negative tweets:
ggdensity(neg_tweets$`Number of Retweets`,
          main = "BMW Negative: Retweets",
          xlab = "Number of Retweets") + xlim(0, max(neut_tweets$`Number of Retweets`))



#Visualize dist. of log(Retweets) for Positive tweets:
ggdensity(pos_tweets$logRetweets,
          main = "BMW Positive: log(Retweets)",
          xlab = "log(Retweets)") #+ xlim(min(pos_tweets$logRetweets), max(neg_tweets$logRetweets))

#Visualize dist. of log(Retweets) for Neutral tweets:
ggdensity(neut_tweets$logRetweets,
          main = "BMW Neutral: log(Retweets)",
          xlab = "log(Retweets)") + xlim(min(pos_tweets$logRetweets), max(neut_tweets$logRetweets))

#Visualize dist. of log(Retweets) for Negative tweets:
ggdensity(neg_tweets$logRetweets,
          main = "BMW Negative: log(Retweets)",
          xlab = "log(Retweets)") + xlim(min(pos_tweets$logRetweets), max(neg_tweets$logRetweets))

#See whether any of the above log distributions may be considered normal:
shapiro.test(pos_tweets$logRetweets)
shapiro.test(neut_tweets$logRetweets)
shapiro.test(neg_tweets$logRetweets)


#Perform Kruskal-Wallis test, for Retweets, grouped by sentiment:
kruskal.test(`Number of Retweets` ~ sent_cat, data = tweets)


#If above results were significant, perform Dunn's test for further examination:
#dunnTest(`Number of Retweets` ~ factor(sent_cat), data = tweets, method = "bonferroni")

#Perform shift function:
#npos = nrow(pos_tweets)
#nneut = nrow(neut_tweets)
#nneg = nrow(neg_tweets)

#df = tibble(gr = factor(c(rep("Positive", npos), rep("Neutral", nneut), rep("Negative", nneg))),
#            obs = c(pos_tweets$`Number of Retweets`, neut_tweets$`Number of Retweets`, neg_tweets$`Number of Retweets`))

#sf = shifthd_pbci(data = df, formula = obs ~gr, doall = TRUE)
#sf

###################
#Again, on log-transformed data
#df = tibble(gr = factor(c(rep("Positive", npos), rep("Neutral", nneut), rep("Negative", nneg))),
#            obs = c(pos_tweets$logRetweets, neut_tweets$logRetweets, neg_tweets$logRetweets))

#sf = shifthd_pbci(data = df, formula = obs ~gr, doall = TRUE)
#Add more colors here, rather than below:
#palette1 = c("violetred2", "steelblue2", "springgreen1", "slategray3")
#psf = plot_sf(sf, plot_theme = 2, symb_fill = palette1)
#psf = plot_sf(sf)
#Add labels for deciles 1 and 9:
#psf = add_sf_lab(psf, sf, y_lab_nudge = .1, text_size = 2)
#do.call("grid.arrange", c(psf, ncol=2))
#psf[[1]]
#psf[[2]]
#psf[[3]]

#Create 1D scatterplots with deciles and color coded differences:
#p = plot_scat2(df, alpha = 0.3, shape = 21)
#p
#p = p + coord_flip()
#p



