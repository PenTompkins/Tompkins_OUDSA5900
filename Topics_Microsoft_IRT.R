#This file is being created to run BTM topic effect val. process over Microsoft IRT tweets
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
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/Pictures/BTMresults_IRT/Microsoft_BTMresults_IRT.xlsx")

#No need to separate IRT/IRT, already done in data
tweets = data
#IRT Tweet Likes Analysis:
##################################################################################################################
#Regular:
#Create density plot of company IRT likes:
ggdensity(tweets$`Number of Likes`,
          main = "Microsoft IRT: Likes",
          xlab = "Number of Likes")

#Log(+0.0001)
#Create density plot of company IRT log(likes):
tweets$logLikes = log(tweets$`Number of Likes` + 0.0001)

ggdensity(tweets$logLikes,
          main = "Microsoft IRT: log(Likes)",
          xlab = "log(Likes)")

#Examine whether this log distribution may be considered normal
#shapiro.test(tweets$logLikes)

#Ensure that topic is considered a factor, nIRT a value:
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
top6 = tweets[which(tweets$topic == 6),]
top7 = tweets[which(tweets$topic == 7),]
top8 = tweets[which(tweets$topic == 8),]
top9 = tweets[which(tweets$topic == 9),]
top10 = tweets[which(tweets$topic ==10),]
top11 = tweets[which(tweets$topic ==11),]
top12 = tweets[which(tweets$topic ==12),]
top13 = tweets[which(tweets$topic ==13),]
top14 = tweets[which(tweets$topic ==14),]
top15 = tweets[which(tweets$topic ==15),]
top16 = tweets[which(tweets$topic ==16),]
top17 = tweets[which(tweets$topic ==17),]
top18 = tweets[which(tweets$topic ==18),]
top19 = tweets[which(tweets$topic ==19),]
nt0 = nrow(top0)
nt1 = nrow(top1)
nt2 = nrow(top2)
nt3 = nrow(top3)
nt4 = nrow(top4)
nt5 = nrow(top5)
nt6 = nrow(top6)
nt7 = nrow(top7)
nt8 = nrow(top8)
nt9 = nrow(top9)
nt10 = nrow(top10)
nt11 = nrow(top11)
nt12 = nrow(top12)
nt13 = nrow(top13)
nt14 = nrow(top14)
nt15 = nrow(top15)
nt16 = nrow(top16)
nt17 = nrow(top17)
nt18 = nrow(top18)
nt19 = nrow(top19)

df = tibble(gr = factor(c(rep("0", nt0), rep("1", nt1), rep("2", nt2), rep("3", nt3),
                          rep("4", nt4), rep("5", nt5), rep("6", nt6), rep("7", nt7),
                          rep("8", nt8), rep("9", nt9), rep("10", nt10), rep("11", nt11),
                        rep("12", nt12), rep("13", nt13), rep("14", nt14), rep("15", nt15), rep("16", nt16),
                        rep("17", nt17), rep("18", nt18), rep("19", nt19))),
            obs = c(top0$`Number of Likes`, top1$`Number of Likes`, top2$`Number of Likes`, top3$`Number of Likes`,
                    top4$`Number of Likes`, top5$`Number of Likes`, top6$`Number of Likes`, top7$`Number of Likes`,
                    top8$`Number of Likes`, top9$`Number of Likes`, top10$`Number of Likes`, top11$`Number of Likes`,
                    top12$`Number of Likes`, top13$`Number of Likes`, top14$`Number of Likes`, top15$`Number of Likes`,
                    top16$`Number of Likes`, top17$`Number of Likes`, top18$`Number of Likes`, top19$`Number of Likes`))

sf = shifthd_pbci(data = df, formula = obs ~gr, doall = TRUE)
sf

###################
#Again, on log-transformed data
df = tibble(gr = factor(c(rep("0", nt0), rep("1", nt1), rep("2", nt2), rep("3", nt3),
                          rep("4", nt4), rep("5", nt5), rep("6", nt6), rep("7", nt7),
                          rep("8", nt8), rep("9", nt9), rep("10", nt10), rep("11", nt11),
                          rep("12", nt12), rep("13", nt13), rep("14", nt14), rep("15", nt15), rep("16", nt16),
                          rep("17", nt17), rep("18", nt18), rep("19", nt19))),
            obs = c(top0$logLikes, top1$logLikes, top2$logLikes, top3$logLikes,
                    top4$logLikes, top5$logLikes, top6$logLikes, top7$logLikes,
                    top8$logLikes, top9$logLikes, top10$logLikes, top11$logLikes,
                    top12$logLikes, top13$logLikes, top14$logLikes, top15$logLikes,
                    top16$logLikes, top17$logLikes, top18$logLikes, top19$logLikes))



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
          main = "Microsoft IRT: Retweets",
          xlab = "Number of Retweets")

#Log(+0.0001)
#Create density plot of company IRT log(Retweets):
tweets$logRetweets = log(tweets$`Number of Retweets` + 0.0001)

ggdensity(tweets$logRetweets,
          main = "Microsoft IRT: log(Retweets)",
          xlab = "log(Retweets)")

#Examine whether this log distribution may be considered normal
#shapiro.test(tweets$logRetweets)

#Ensure that topic is considered a factor, nIRT a value:
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

#Dont' believe any results above are significant

#Perform shift function:
#Reset to contain 'logRetweets'
top0 = tweets[which(tweets$topic == 0),]
top1 = tweets[which(tweets$topic == 1),]
top2 = tweets[which(tweets$topic == 2),]
top3 = tweets[which(tweets$topic == 3),]
top4 = tweets[which(tweets$topic == 4),]
top5 = tweets[which(tweets$topic == 5),]
top6 = tweets[which(tweets$topic == 6),]
top7 = tweets[which(tweets$topic == 7),]
top8 = tweets[which(tweets$topic == 8),]
top9 = tweets[which(tweets$topic == 9),]
top10 = tweets[which(tweets$topic ==10),]
top11 = tweets[which(tweets$topic ==11),]
top12 = tweets[which(tweets$topic ==12),]
top13 = tweets[which(tweets$topic ==13),]
top14 = tweets[which(tweets$topic ==14),]
top15 = tweets[which(tweets$topic ==15),]
top16 = tweets[which(tweets$topic ==16),]
top17 = tweets[which(tweets$topic ==17),]
top18 = tweets[which(tweets$topic ==18),]
top19 = tweets[which(tweets$topic ==19),]


df = tibble(gr = factor(c(rep("0", nt0), rep("1", nt1), rep("2", nt2), rep("3", nt3),
                          rep("4", nt4), rep("5", nt5), rep("6", nt6), rep("7", nt7),
                          rep("8", nt8), rep("9", nt9), rep("10", nt10), rep("11", nt11),
                          rep("12", nt12), rep("13", nt13), rep("14", nt14), rep("15", nt15), rep("16", nt16),
                          rep("17", nt17), rep("18", nt18), rep("19", nt19))),
            obs = c(top0$`Number of Retweets`, top1$`Number of Retweets`, top2$`Number of Retweets`, top3$`Number of Retweets`,
                    top4$`Number of Retweets`, top5$`Number of Retweets`, top6$`Number of Retweets`, top7$`Number of Retweets`,
                    top8$`Number of Retweets`, top9$`Number of Retweets`, top10$`Number of Retweets`, top11$`Number of Retweets`,
                    top12$`Number of Retweets`, top13$`Number of Retweets`, top14$`Number of Retweets`, top15$`Number of Retweets`,
                    top16$`Number of Retweets`, top17$`Number of Retweets`, top18$`Number of Retweets`, top19$`Number of Retweets`))

sf = shifthd_pbci(data = df, formula = obs ~gr, doall = TRUE)
sf

###################
#Again, on log-transformed data
df = tibble(gr = factor(c(rep("0", nt0), rep("1", nt1), rep("2", nt2), rep("3", nt3),
                          rep("4", nt4), rep("5", nt5), rep("6", nt6), rep("7", nt7),
                          rep("8", nt8), rep("9", nt9), rep("10", nt10), rep("11", nt11),
                          rep("12", nt12), rep("13", nt13), rep("14", nt14), rep("15", nt15), rep("16", nt16),
                          rep("17", nt17), rep("18", nt18), rep("19", nt19))),
            obs = c(top0$logRetweets, top1$logRetweets, top2$logRetweets, top3$logRetweets,
                    top4$logRetweets, top5$logRetweets, top6$logRetweets, top7$logRetweets,
                    top8$logRetweets, top9$logRetweets, top10$logRetweets, top11$logRetweets,
                    top12$logRetweets, top13$logRetweets, top14$logRetweets, top15$logRetweets,
                    top16$logRetweets, top17$logRetweets, top18$logRetweets, top19$logRetweets))



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
