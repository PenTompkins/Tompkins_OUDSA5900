#Creating this file just to help figure out best method for separating OT vs. IRT across all datasets
#Overall, this is program: 18

library(readxl)

#read in the data: Below line of code will need to be reconfigured for your filepath
data = read_excel("C:/Users/Pendl/OneDrive/Documents/TwitterProject/DoyleData/Toyota_Dec_1_2020.xlsx")

#Filter out retweets
tweets = data[-grep("^RT @", data$Content),]

#Keep in mind that for datasets which contain no retweets,
tweets = data

#Examine results original split would have produced:
all_OT = tweets[-grep("^@", tweets$Content),]
all_IRT = tweets[grep("^@", tweets$Content),]

#Ensure that no official tweets have been wrongfully marked IRT:
wrong_place = all_IRT[which(all_IRT$Author == all_IRT$`In Reply To`),] #locate wrongfully placed official tweets
all_OT = rbind(all_OT, wrong_place) #add the tweet(s) to official category
all_IRT = all_IRT[-which(all_IRT$Author == all_IRT$`In Reply To`),]


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
#Examine results
all_OTnew = tweets[which(tweets$OT == 1),]
all_IRTnew = tweets[which(tweets$IRT == 1),]
#Results seem pretty solid for Disney. They technically have 2 IRT tweets, they're just not customer service oriented
#Results seem pretty solid for McDonalds. Got rid of the two 'fake' official tweets (McDonalds double IRTs)
#Results for Amazon match results in 'DataExplorationNotes'
#Results for BMW match results in 'DataExplorationNotes'
#Results for Coca Cola match results in 'DataExplorationNotes'
#Results for Google match results in 'DataExplorationNotes'












