#This file is used to create the 'num_hashtags' variable for each dataset
#Overall, this is program: 14

import pandas as pd
import numpy as np
import nltk, os, sys, email, re

#Read in the data: Below line of code will need to be reconfigured for your filepath
data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Toyota_Dec_1_2020.xlsx')
#print(amazon_data.shape)

#old way: Works well except gets tricked when things like '#1' occur
#amazon_data["num_hashtags"] = amazon_data["Content"].str.count("#")

#New way:
#PATTERN = re.compile(r'#(?!\d+\.?,? )(\w+)') #hashtags that aren't followed by any number of digits then (optionally a period or comma) a space (aren't like '#1' or '#123709. ')
#directly above works best so far..only doesn't work when '#1' or something is last part of tweet and has no spaces afterwards
#PATTERN = re.compile(r'#(?!\d+\.?,?( |))(\w+)') #should be same as above, but followed by space OR nothing..we'll see if it worked
#Above pattern seems to now work even when the tweet ends with something like '#01290832.'
#Actually, above pattern failed to count '#24hNBR.' as a true hashtag. Seeing if getting rid of '( |)'s parantheses works
#PATTERN = re.compile(r'#(?!\d+\.?,? |)(\w+)') #removing the parantheses caused no hashtags to be counted at all

#What if I try a pattern that says 'match digits only if they're followed by chars'?
#PATTERN = re.compile(r'#\d+(?=\D+)(\w+)')
#I THINK the above pattern is saying 'find hashtags, match digits only if they're followed by letters, then match anything? Might need to change the \w+ at the end
#Above pattern failed in that it ONLY counted when hashtags were like '#24hNBR'

#What if I try to change my negative look ahead to look for digits that aren't followed by letters?
#PATTERN = re.compile(r'#(?!\d+\.?,?( |))(\w+)') #copied for comparison
#PATTERN = re.compile(r'#(?!\d+(?=\D+))(\w+)') #this performed exactly the same as best so far (missed #24hNBR and something on row 1894 of BMW)
#trying exact same pattern with '\w+' replaced by '\D+'
#PATTERN = re.compile(r'#(?!\d+(?=\D+))(\D+)') #this caused only 1090 BMW hashtags to be found

#What if I only search for hashtags then perform a negative lookahead for digits that aren't followed by letters?
#It shouldn't theoretically matter to match letters, so long as the hashtag is present and it's not followed by solely digits
#PATTERN = re.compile(r'#(?!\d+(?=\D+))') #seems to have perfomed the same as best so far
#PATTERN = re.compile(r'#(?!\d+(?=!\D+))') #same as above, but only if they're NOT followed by letters...1315 for BMW, same as initial
#Changing above pattern slightly, negative lookahead for digits NOT followed by upper or lower case letters
#PATTERN = re.compile(r'#(?!\d+(?=![A-Za-z]))') #counted 1315 for BMW
#Seeing if adding a dollar sign to above regex helps
#PATTERN = re.compile(r'#(?!\d+(?=![A-Za-z]$))') #counted 1315 for BMW 
#What if I get rid of the negative lookahead and only count digits when followed by letters?
#PATTERN = re.compile(r'#\d+(?=[A-Za-z])') #this one only found the 2 hashtags that start w/ numbers and end w/ letters
#adding to the above pattern to hopefully also match when no numbers are present at all
#PATTERN = re.compile(r'#\d+(?=[A-Za-z])[A-Za-z]') #only 2 hashtags match still for BMW, as above
#Keeping the same general idea as before, but putting OR condition in the regex
PATTERN = re.compile(r'#(\d+(?=[A-Za-z])|[A-Za-z])') #This counted 1314 for BMW!!!
#Match hashtags followed by one or more digits, but only if those are followed by a letter. Or, match hashtags followed directly by a letter



data["num_hashtags"] = data["Content"].str.count(PATTERN)

#print(amazon_data["num_hashtags"])
data.to_csv("CMPNY_hashtags.csv")