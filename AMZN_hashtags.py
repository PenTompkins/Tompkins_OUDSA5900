#Amazon Initial Hashtag Exploration
#Overall, this is program: 6

import pandas as pd
import numpy as np
import nltk, os, sys, email, re

#Read in the Amazon data
amazon_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\DoyleData\\Amazon_Dec_1_2020.xlsx')
#print(amazon_data.shape)

#old way: Works well except gets tricked when things like '#1' occur
#amazon_data["num_hashtags"] = amazon_data["Content"].str.count("#")

#New way:
PATTERN = re.compile(r'#(?!\d )(\w+)') #hashtags that aren't followed by a single digit then a space, I believe
amazon_data["num_hashtags2"] = amazon_data["Content"].str.count(PATTERN)

#print(amazon_data["num_hashtags"])
amazon_data.to_csv("AMZN_hashtags.csv")

#!!! This seems to have worked. However, saving to csv messes up emojis and a bunch of other textual contents
#Thus, I'm just going to copy the num_hashtags column of AMZN_hashtags.csv into the real Amazon data from Dr. Yoon



