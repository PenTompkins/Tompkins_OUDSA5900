#This file is being created to shuffle analysis data and split it into reproducible training and testing datasets, for official tweets

import pandas as pd
import numpy as np
import nltk, os, sys, email, re
import random

from sklearn.model_selection import train_test_split


#Never hurts:
random.seed(1)
np.random.seed(2)

#Read in all data contained in 'Analysis_Data_OT' folder: (below line will need to be reconfigured for your filepath)
company_data = pd.read_excel('C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Analysis_Data_OT\\Toyota_relTweets_OT.xlsx')

company_name = company_data.iloc[0, company_data.columns.get_loc("Author Name")]

#Split data into test/train sets:
train_data, test_data = train_test_split(company_data, test_size = 0.2, random_state = 1)

#Save train_data to 'train' folder
train_path = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Analysis_Data_OT\\Train\\'
trainEnd = '_train_OT.xlsx'
fpath = train_path + str(company_name) + trainEnd
train_data.to_excel(fpath)

#Save test_data to 'test' folder:
test_path = 'C:\\Users\\Pendl\\OneDrive\\Documents\\TwitterProject\\Analysis_Data_OT\\Test\\'
testEnd = '_test_OT.xlsx'
fpath2 = test_path + str(company_name) + testEnd
test_data.to_excel(fpath2)
print("All done")