#Twitter Project: Exploring Amazon Data
#Pen Tompkins
#Overall, this is program: 1
library(readxl)

#Read in Amazon data: Below line of code will need to be re-configured for your personal file path
amazon_data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/Amazon_Dec_1_2020.xlsx")
num_likes = amazon_data[,6]
num_RT = amazon_data[,7]
RT_likes = data.frame(num_RT, num_likes)
#Plot relationship between retweets and likes
plot(RT_likes)
#Doesn't appear to be a strong relationship

likes_RT = data.frame(num_likes, num_RT)
plot(likes_RT)
#Although, when looking at it the other way around, there may be a weak linear relationship
#However, there are definitely some massive outliers to this

which.max(likes_RT[,2])
#Row 2252 is a RT of Jeff Bezos' account

#likes_RT2 = likes_RT[-2252,]
#plot(likes_RT2)
#which.max(likes_RT2[,2])
#There are still Amazon retweets in the data

#Trying to identify the Amazon retweets so I can clean them out of the data:
#tweets = amazon_data[,2]
#RT_locations = grep("^RT @", tweets)
#length(RT_locations)
#Above wasn't working

amazon_tweets = amazon_data[-grep("^RT @", amazon_data$Content),]
#Seems to have worked, time to check

RT_locations = grep("^RT @", amazon_data$Content)
length(RT_locations) #61
#And 3235-3174 = 61 as well. Think it worked

#Might as well manually check too, just to be safe
RT_locations
#Manually checked, they're all retweets from the Amazon account (not of)
#Doesn't necessarily mean we removed all the RTs (think we did though),
#but everything we removed was definitely a RT

num_likes2 = amazon_tweets[,6]
num_RT2 = amazon_tweets[,7]

#Do the above again, without retweets from Amazon
RT_likes2 = data.frame(num_RT2, num_likes2)
plot(RT_likes2)
#Now, there appears to be a much more clear, stronger linear relationship between the two

likes_RT2 = data.frame(num_likes2, num_RT2)
plot(likes_RT2)
#However, there's definitely not a perfectly linear relationship

#Might as well examine effect of including link:
without_links = amazon_tweets[-grep("https://", amazon_tweets$Content),]
with_links = amazon_tweets[grep("https://", amazon_tweets$Content),]

wo_links_likes = without_links[[6]]
mean(wo_links_likes) #7.006, but it's including 'customer service' tweets

w_links_likes = with_links[[6]]
mean(w_links_likes) #77.6143

wo_links_RT = without_links[[7]]
mean(wo_links_RT) #1.161535

w_links_RT = with_links[[7]]
mean(w_links_RT) #13.03198

#What happens if we remove the 'customer service' tweets from w/o links?
without_links_OT = without_links[-grep("^@", without_links$Content),]
mean(without_links_OT[[6]])#average of 296.6538 likes
mean(without_links_OT[[7]])#average of 46.76923 retweets

without_links_CS = without_links[grep("^@", without_links$Content),]
mean(without_links_CS[[6]]) #average of 3.39 likes
mean(without_links_CS[[7]]) #average of 0.59 RT


#Seems like my 'with_links' data has customer service tweets in it too
with_links_notCS = with_links[-grep("^@", with_links$Content),]
with_links_andCS = with_links[grep("^@", with_links$Content),]
mean(with_links_notCS[[6]]) #avg of 361.2136 likes
mean(with_links_notCS[[7]]) #avg of 60.97727 RT

#With links and customer service:
mean(with_links_andCS[[6]]) #avg of 3.60261 likes
mean(with_links_andCS[[7]]) #avg of 0.519573 RT




























