
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
from sklearn.metrics.pairwise import cosine_similarity

import Text_Similarity_CosineSimilarity_Factory as Factory
f = Factory.Data_preprocessing()

ModelBase = pd.read_csv(r'C:\\Users\\baowanyun\\Text-Similarity-Algorithms_ModelBase.csv', encoding='latin-1') ; #print(df[0:5])
ModelBase = ModelBase[ModelBase['Spam'] == 1].reset_index(drop = True)
#print(ModelBase[0:2]) ; print(ModelBase.shape)  ## (603, 6)


for num, content in enumerate(TestingDataset.EmailContent, start=1):
    print(num)
    #print('Original spam : %s' %TestingDataset.Spam[num-1])
    #print("Email {}: {}".format(num, content)) 
    clean = f.text_preprocessing(content)
    content_clean = clean[0] ; #print(content_clean)
    msg_len = clean[1]
    
    External = f.External_URL(content)
    Result = External[0]
    ExternalURL = External[1]
    if Result == True :
        Spam = 1
    else :                   
        ref = list(ModelBase.EmailContent_clean)
        tar = content_clean
        combin = [tar] + ref ;       
        tfidf_matrix = tfidf_vectorizer.fit_transform(combin)
        Email_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])                        
        word_cnt_array = np.sqrt(abs(np.asarray(ModelBase.Length)-msg_len)+1) ## Normalize
        sim_norm = np.divide(Email_sim[0],word_cnt_array)
        sim_norm_list = sim_norm.tolist()
        index, value = max(enumerate(sim_norm_list), key=operator.itemgetter(1))
        sim_email = ModelBase[index:index+1] ; #print(str(sim_email.EmailContent_clean))
        sim = value ; #print('Similarity') ; #print(sim)
        Sim_Spam = int(sim_email['Spam'].values) ; #print('Sim_Spam') ;print(Sim_Spam)
        if Sim_Spam == 1 and sim > 0.3:
            Spam = 1
        elif Sim_Spam == 0 and sim > 0.3:                              
            Spam = 1
        else:
            Spam = 0
    #print('Spam or not: %s' %spam)
    if TestingDataset.Spam[num-1] != Spam:
        print('Original spam : %s' %TestingDataset.Spam[num-1])
        print("Email {}: {}".format(num, content)) 
        print(content_clean)
       
        print(str(sim_email.EmailContent_clean))
        print(sim)
        print('Sim_Spam') ;print(Sim_Spam)
        print('Spam or not: %s' %Spam)

