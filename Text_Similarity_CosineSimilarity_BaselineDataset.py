# coding: utf-8
# Author: Bao Wan-Yun

import pandas as pd
from sklearn.model_selection import train_test_split
import Text_Similarity_CosineSimilarity_Factory as Factory
f = Factory.Data_preprocessing()

df = pd.read_csv(r'C:\Users\baowanyun\SMS_Spam_Collection_Dataset.csv',usecols=[0,1], encoding='latin-1') ; #print(df[0:5])
df = df.rename(columns = {"v1":"Label", "v2": "EmailContent"}) #print(df.groupby('Label').count())

X = pd.DataFrame(df.EmailContent) # load the dataset as a pandas data frame
Y = pd.DataFrame(df.Label) # define the target variable (dependent variable) as y

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
TrainingDataset = pd.concat([y_train, X_train], axis = 1).reset_index(drop=True) ; print(TrainingDataset.shape)
TrainingDataset['Spam'] = [1 if i == 'spam' else 0 for i in TrainingDataset.Label]
TestingDataset = pd.concat([y_test, X_test], axis = 1).reset_index(drop=True) ; print(TestingDataset.shape)
TestingDataset['Spam'] = [1 if i == 'spam' else 0 for i in TestingDataset.Label]
#print(TrainingDataset.groupby('Label').count()) ; print(TestingDataset.groupby('Label').count

EmailContent_clean = [] ; EmailContent_len = []
for num, content in enumerate(TrainingDataset.EmailContent, start=1):
    #Spam = 0
    print("Email {}: {}".format(num, content))  
    
    clean = f.text_preprocessing(content)
    content_clean = clean[0]
    msg_len = clean[1]
    EmailContent_clean.append(content_clean)
    EmailContent_len.append(msg_len)

## Store Baseline Dataset (.csv)
TrainingDataset['EmailContent_clean'] = EmailContent_clean
TrainingDataset['Length'] = EmailContent_len
TrainingDataset = TrainingDataset[['Label','Spam','EmailContent','EmailContent_clean','Length']]

TrainingDataset.to_csv('C:\\Users\\baowanyun\\Text-Similarity-Algorithms_BaselineDataset.csv')
TestingDataset.to_csv('C:\\Users\\baowanyun\\Text-Similarity-Algorithms_TestingDataset.csv')

