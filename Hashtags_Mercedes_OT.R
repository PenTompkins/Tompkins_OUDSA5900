#This file is being created to run hashtag effect val. process over Mercedes OT tweets
#
#Overall, this is program: 75

library(readxl)
library(ggpubr)
library(fpp2)
library(geoR)
library(ggplot2)
library(rogme)

set.seed(1)

#read in the data: Below line of code will have to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/OutlierFree_data/MercedesBenz_outsRemoved.xlsx")

#Extract official tweets only:
tweets = data[which(data$OT == TRUE),]


#Official Tweet Likes Analysis:
##################################################################################################################
#Regular:
#Create density plot of company OT likes:
ggdensity(tweets$`Number of Likes`,
          main = "Mercedes OT: Likes",
          xlab = "Number of Likes")

#Log(+0.0001)
#Create density plot of company OT log(likes):
tweets$logLikes = log(tweets$`Number of Likes` + 0.0001)

ggdensity(tweets$logLikes,
          main = "Mercedes OT: log(Likes)",
          xlab = "log(Likes)")

#Examine whether this log distribution may be considered normal
shapiro.test(tweets$logLikes)


#Create binary 'has_hash' (hashtag) variable in data:
has_hash = ifelse(tweets$num_hashtags > 0, TRUE, FALSE)
sum(has_hash)
tweets["has_hash"] = has_hash


#Create visualization for Hashtag effect on likes across all tweets:
ggboxplot(tweets, x = "has_hash", y = "Number of Likes",
          color = "has_hash", ylab = "Number of Likes", xlab = "Contains Hashtag")


#Create visualization for Hashtag effect on log(likes+.0001) across all tweets:
ggboxplot(tweets, x = "has_hash", y = "logLikes",
          color = "has_hash", ylab = "Log(Likes)", xlab = "Contains Hashtag")

#Extract tweets containing Hashtags:
Hashtag_tweets = tweets[which(tweets$has_hash == TRUE),]

#Extract tweets not containing Hashtags:
noHashtag_tweets = tweets[which(tweets$has_hash == FALSE),]

#Visualize dist. of likes for Hashtag_tweets:
ggdensity(Hashtag_tweets$`Number of Likes`,
          main = "Mercedes Hashtags: Likes",
          xlab = "Number of Likes") + xlim(0, max(noHashtag_tweets$`Number of Likes`))

#Visualize dist. of likes for noHashtag_tweets:
ggdensity(noHashtag_tweets$`Number of Likes`,
          main = "Mercedes No Hashtag: Likes",
          xlab = "Number of Likes") #+ xlim(0, max(Hashtag_tweets$`Number of Likes`))


#Visualize dist. of log(likes) for Hashtag_tweets:
ggdensity(Hashtag_tweets$logLikes,
          main = "Mercedes Hashtags: log(Likes)",
          xlab = "log(Likes)") #+ xlim(min(Hashtag_tweets$logLikes), max(noHashtag_tweets$logLikes))

#Visualize dist. of log(likes) for noHashtag_tweets:
ggdensity(noHashtag_tweets$logLikes,
          main = "Mercedes No Hashtag: log(Likes)",
          xlab = "log(Likes)")




#See whether dist. of log(likes) for Hashtag_tweets may be considered normal:
shapiro.test(Hashtag_tweets$logLikes)

#See whether dist. of log(likes) for noHashtag_tweets may be considered normal:
shapiro.test(noHashtag_tweets$logLikes)

#Perform Mann Whitney U test on set of all tweets w/ Hashtag vs. all tweets w/o Hashtag (for likes):
res = wilcox.test(Hashtag_tweets$`Number of Likes`, noHashtag_tweets$`Number of Likes`)
res


#Perform shift function analysis on non-transformed likes, to obtain confidence intervals:
Hashtag_likes = Hashtag_tweets$`Number of Likes`
noHashtag_likes = noHashtag_tweets$`Number of Likes`
df = mkt2(Hashtag_likes, noHashtag_likes)

#Compute the shift function
#sf = shifthd(data = df, formula = obs ~ gr, nboot = 200)
sf = shifthd_pbci(data = df, formula = obs ~ gr)
sf

#########################
#Again, on log transformed data:
df = mkt2(Hashtag_tweets$logLikes, noHashtag_tweets$logLikes)
#Make scatterplots for the two groups:
ps = plot_scat2(data = df, formula = obs ~ gr, alpha = 1, shape = 21)
ps = ps + coord_flip()
ps
#Re-compute the shift function
#sf = shifthd(data = df, formula = obs ~ gr, nboot = 200)
sf = shifthd_pbci(data = df, formula = obs ~ gr)
#Add more colors here, rather than below:
palette1 = c("violetred2", "steelblue2", "springgreen1")
psf = plot_sf(sf, plot_theme = 2, symb_fill = palette1)
#Add labels for deciles 1 and 9:
psf = add_sf_lab(psf, sf, y_lab_nudge = .1, text_size = 4)
#Change axis labels
psf[[1]] = psf[[1]] + labs(x = "Hashtags Tweets Quantiles of Likes",
                           y = "Hashtag Likes - NoHashtag Likes \nquantile differences")

psf[[1]]
#Create 1D scatterplots with deciles and color coded differences:
p = plot_scat2(df, alpha = 0.3, shape = 21)
p
p = plot_hd_links(p, sf[[1]],
                  q_size = 1,
                  md_size = 1.5,
                  add_rect = TRUE,
                  rect_alpha = .1,
                  rect_col = "grey50",
                  add_lab = TRUE,
                  text_size = 5)
p
p = p + coord_flip()
p


#RETWEET ANALYSIS BELOW:
################################################################################################################################
#Regular:
#Create density plot of company OT retweets:
ggdensity(tweets$`Number of Retweets`,
          main = "Mercedes OT: Retweets",
          xlab = "Number of Retweets")

#Log(+0.0001)
#Create density plot of company OT log(retweets):
tweets$logRetweets = log(tweets$`Number of Retweets` + 0.0001)

ggdensity(tweets$logRetweets,
          main = "Mercedes OT: log(Retweets)",
          xlab = "log(Retweets)")

#Examine whether this log distribution may be considered normal
shapiro.test(tweets$logRetweets)

#Reset Hashtag_tweets and noHashtag_tweets such that they contain 'logRetweets'
Hashtag_tweets = tweets[which(tweets$has_hash == TRUE),]
noHashtag_tweets = tweets[which(tweets$has_hash == FALSE),]


#Create visualization for Hashtag effect on retweets across all tweets:
ggboxplot(tweets, x = "has_hash", y = "Number of Retweets",
          color = "has_hash", ylab = "Number of Retweets", xlab = "Contains Hashtag")


#Create visualization for Hashtag effect on log(retweets+.0001) across all tweets:
ggboxplot(tweets, x = "has_hash", y = "logRetweets",
          color = "has_hash", ylab = "Log(Retweets)", xlab = "Contains Hashtag")

#Visualize dist. of retweets for Hashtag_tweets:
ggdensity(Hashtag_tweets$`Number of Retweets`,
          main = "Mercedes Hashtags: Retweets",
          xlab = "Number of Retweets") + xlim(0, max(noHashtag_tweets$`Number of Retweets`))

#Visualize dist. of retweets for noHashtag_tweets:
ggdensity(noHashtag_tweets$`Number of Retweets`,
          main = "Mercedes No Hashtag: Retweets",
          xlab = "Number of Retweets") #+ xlim(0, max(Hashtag_tweets$`Number of Retweets`))


#Visualize dist. of log(retweets) for Hashtag_tweets:
ggdensity(Hashtag_tweets$logRetweets,
          main = "Mercedes Hashtags: log(RTs)",
          xlab = "log(Retweets)") + xlim(min(Hashtag_tweets$logRetweets), max(noHashtag_tweets$logRetweets))

#Visualize dist. of log(retweets) for noHashtag_tweets:
ggdensity(noHashtag_tweets$logRetweets,
          main = "Mercedes No Hashtag: log(RTs)",
          xlab = "log(Retweets)")




#See whether dist. of log(retweets) for Hashtag_tweets may be considered normal:
shapiro.test(Hashtag_tweets$logRetweets)

#See whether dist. of log(retweets) for noHashtag_tweets may be considered normal:
shapiro.test(noHashtag_tweets$logRetweets)

#Perform Mann Whitney U test on set of all tweets w/ Hashtag vs. all tweets w/o Hashtag (for retweets):
res = wilcox.test(Hashtag_tweets$`Number of Retweets`, noHashtag_tweets$`Number of Retweets`)
res


#Perform shift function analysis on non-transformed retweets, to obtain confidence intervals:
Hashtag_retweets = Hashtag_tweets$`Number of Retweets`
noHashtag_retweets = noHashtag_tweets$`Number of Retweets`
df = mkt2(Hashtag_retweets, noHashtag_retweets)

#Compute the shift function
#sf = shifthd(data = df, formula = obs ~ gr, nboot = 200)
sf = shifthd_pbci(data = df, formula = obs ~ gr)
sf

#########################
#Again, on log transformed data:
df = mkt2(Hashtag_tweets$logRetweets, noHashtag_tweets$logRetweets)
#Make scatterplots for the two groups:
ps = plot_scat2(data = df, formula = obs ~ gr, alpha = 1, shape = 21)
ps = ps + coord_flip()
ps
#Re-compute the shift function
#sf = shifthd(data = df, formula = obs ~ gr, nboot = 200)
sf = shifthd_pbci(data = df, formula = obs ~ gr)
#Add more colors here, rather than below:
palette1 = c("violetred2", "steelblue2", "springgreen1")
psf = plot_sf(sf, plot_theme = 2, symb_fill = palette1)
#Add labels for deciles 1 and 9:
psf = add_sf_lab(psf, sf, y_lab_nudge = .1, text_size = 4)
#Change axis labels
psf[[1]] = psf[[1]] + labs(x = "Hashtags Tweets Quantiles of Likes",
                           y = "Hashtag Likes - NoHashtag Likes \nquantile differences")

psf[[1]]
#Create 1D scatterplots with deciles and color coded differences:
p = plot_scat2(df, alpha = 0.3, shape = 21)
p
p = plot_hd_links(p, sf[[1]],
                  q_size = 1,
                  md_size = 1.5,
                  add_rect = TRUE,
                  rect_alpha = .1,
                  rect_col = "grey50",
                  add_lab = TRUE,
                  text_size = 5)
p
p = p + coord_flip()
p





