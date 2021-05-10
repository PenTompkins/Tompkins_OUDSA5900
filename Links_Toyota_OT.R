#This file is being created to run link effect val. process over Toyota OT tweets
#
#Overall, this is program: 73...

library(readxl)
library(ggpubr)
library(fpp2)
library(geoR)
library(ggplot2)
library(rogme)

set.seed(1)

#read in the data: Below line of code will have to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/OutlierFree_data/Toyota_outsRemoved.xlsx")

#Extract official tweets only:
tweets = data[which(data$OT == TRUE),]


#Official Tweet Likes Analysis:
##################################################################################################################
#Regular:
#Create density plot of company OT likes:
ggdensity(tweets$`Number of Likes`,
          main = "Toyota OT: Likes",
          xlab = "Number of Likes")

#Log(+0.0001)
#Create density plot of company OT log(likes):
tweets$logLikes = log(tweets$`Number of Likes` + 0.0001)

ggdensity(tweets$logLikes,
          main = "Toyota OT: log(Likes)",
          xlab = "log(Likes)")

#Examine whether this log distribution may be considered normal
shapiro.test(tweets$logLikes)


#Create binary 'has_link' variable in the data:
has_link = grepl("https://", tweets$Content)
sum(has_link)
tweets["has_link"] = has_link


#Create visualization for link effect on likes across all tweets:
ggboxplot(tweets, x = "has_link", y = "Number of Likes",
          color = "has_link", ylab = "Number of Likes", xlab = "Contains Link")


#Create visualization for link effect on log(likes+.0001) across all tweets:
ggboxplot(tweets, x = "has_link", y = "logLikes",
          color = "has_link", ylab = "Log(Likes)", xlab = "Contains Link")

#Extract tweets containing links:
link_tweets = tweets[which(tweets$has_link == TRUE),]

#Extract tweets not containing links:
noLink_tweets = tweets[which(tweets$has_link == FALSE),]

#Visualize dist. of likes for link_tweets:
ggdensity(link_tweets$`Number of Likes`,
          main = "Toyota Linked: Likes",
          xlab = "Number of Likes")

#Visualize dist. of likes for noLink_tweets:
ggdensity(noLink_tweets$`Number of Likes`,
          main = "Toyota No Link: Likes",
          xlab = "Number of Likes") + xlim(0, max(link_tweets$`Number of Likes`))


#Visualize dist. of log(likes) for link_tweets:
ggdensity(link_tweets$logLikes,
          main = "Toyota Linked: log(Likes)",
          xlab = "log(Likes)") #+ xlim(min(link_tweets$logLikes), max(noLink_tweets$logLikes))

#Visualize dist. of log(likes) for noLink_tweets:
ggdensity(noLink_tweets$logLikes,
          main = "Toyota No Link: log(Likes)",
          xlab = "log(Likes)")

#Would above look better on same x-axis as link_tweets?
ggdensity(noLink_tweets$logLikes,
          main = "Toyota No Link: log(Likes)",
          xlab = "log(Likes)") + xlim(min(link_tweets$logLikes), max(noLink_tweets$logLikes))
#I believe so


#See whether dist. of log(likes) for link_tweets may be considered normal:
shapiro.test(link_tweets$logLikes)

#See whether dist. of log(likes) for noLink_tweets may be considered normal:
shapiro.test(noLink_tweets$logLikes)

#Perform Mann Whitney U test on set of all tweets w/ link vs. all tweets w/o link (for likes):
res = wilcox.test(link_tweets$`Number of Likes`, noLink_tweets$`Number of Likes`)
res


#Perform shift function analysis on non-transformed likes, to obtain confidence intervals:
link_likes = link_tweets$`Number of Likes`
noLink_likes = noLink_tweets$`Number of Likes`
df = mkt2(link_likes, noLink_likes)

#Compute the shift function
#sf = shifthd(data = df, formula = obs ~ gr, nboot = 200)
sf = shifthd_pbci(data = df, formula = obs ~ gr)
sf

#########################
#Again, on log transformed data:
df = mkt2(link_tweets$logLikes, noLink_tweets$logLikes)
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
psf[[1]] = psf[[1]] + labs(x = "Linked Tweets Quantiles of Likes",
                           y = "Link Likes - NoLink Likes \nquantile differences")

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
          main = "Toyota OT: Retweets",
          xlab = "Number of Retweets")

#Log(+0.0001)
#Create density plot of company OT log(retweets):
tweets$logRetweets = log(tweets$`Number of Retweets` + 0.0001)

ggdensity(tweets$logRetweets,
          main = "Toyota OT: log(Retweets)",
          xlab = "log(Retweets)")

#Examine whether this log distribution may be considered normal
shapiro.test(tweets$logRetweets)

#Reset link_tweets and noLink_tweets such that they contain 'logRetweets'
link_tweets = tweets[which(tweets$has_link == TRUE),]
noLink_tweets = tweets[which(tweets$has_link == FALSE),]


#Create visualization for link effect on retweets across all tweets:
ggboxplot(tweets, x = "has_link", y = "Number of Retweets",
          color = "has_link", ylab = "Number of Retweets", xlab = "Contains Link")


#Create visualization for link effect on log(retweets+.0001) across all tweets:
ggboxplot(tweets, x = "has_link", y = "logRetweets",
          color = "has_link", ylab = "Log(Retweets)", xlab = "Contains Link")

#Visualize dist. of retweets for link_tweets:
ggdensity(link_tweets$`Number of Retweets`,
          main = "Toyota Linked: Retweets",
          xlab = "Number of Retweets") + xlim(0, max(noLink_tweets$`Number of Retweets`))

#Visualize dist. of retweets for noLink_tweets:
ggdensity(noLink_tweets$`Number of Retweets`,
          main = "Toyota No Link: Retweets",
          xlab = "Number of Retweets") #+ xlim(0, max(link_tweets$`Number of Retweets`))


#Visualize dist. of log(retweets) for link_tweets:
ggdensity(link_tweets$logRetweets,
          main = "Toyota Linked: log(RTs)",
          xlab = "log(Retweets)")

#Visualize dist. of log(retweets) for noLink_tweets:
ggdensity(noLink_tweets$logRetweets,
          main = "Toyota No Link: log(RTs)",
          xlab = "log(Retweets)")

#Would above look better on same x-axis as link_tweets?
ggdensity(noLink_tweets$logRetweets,
          main = "Toyota No Link: log(RTs)",
          xlab = "log(Retweets)") + xlim(min(link_tweets$logRetweets), max(noLink_tweets$logRetweets))
#I believe so


#See whether dist. of log(retweets) for link_tweets may be considered normal:
shapiro.test(link_tweets$logRetweets)

#See whether dist. of log(retweets) for noLink_tweets may be considered normal:
shapiro.test(noLink_tweets$logRetweets)

#Perform Mann Whitney U test on set of all tweets w/ link vs. all tweets w/o link (for retweets):
res = wilcox.test(link_tweets$`Number of Retweets`, noLink_tweets$`Number of Retweets`)
res


#Perform shift function analysis on non-transformed retweets, to obtain confidence intervals:
link_retweets = link_tweets$`Number of Retweets`
noLink_retweets = noLink_tweets$`Number of Retweets`
df = mkt2(link_retweets, noLink_retweets)

#Compute the shift function
#sf = shifthd(data = df, formula = obs ~ gr, nboot = 200)
sf = shifthd_pbci(data = df, formula = obs ~ gr)
sf

#########################
#Again, on log transformed data:
df = mkt2(link_tweets$logRetweets, noLink_tweets$logRetweets)
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
psf[[1]] = psf[[1]] + labs(x = "Linked Tweets Quantiles of Likes",
                           y = "Link Likes - NoLink Likes \nquantile differences")

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





