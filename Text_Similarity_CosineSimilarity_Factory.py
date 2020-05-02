# coding: utf-8
# Author: Bao Wan-Yun

import unicodedata
import re
import urllib.request
import urllib.error
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/49.0.2')]
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[\d.]+\b\.*|[\w\']+')
tokenizer2 = RegexpTokenizer(r'\w+')

#from nltk.stem import WordNetLemmatizer
#wordnet_lemmatizer = WordNetLemmatizer()
from nltk.stem import SnowballStemmer
Snow_stemmer = SnowballStemmer('english')

from stop_words import get_stop_words
en_stop = get_stop_words('en') 

class Data_preprocessing():
    def __init__(self):
        self.author = 'WY Bao'
        
    def text_preprocessing(self , content):
        content = content.encode('ascii', 'ignore').decode('utf-8') ## ignore 'Ì©'  ; <class 'bytes'> decode=> <class 'str'>
        NotFullWidth = unicodedata.normalize('NFKC',content) ## didn't (didnt) 全型字
        NotEmojiPattern = NotFullWidth.encode('ASCII','ignore').decode('utf-8') ## Emoji Pattern
        content = NotEmojiPattern.lower()
        a = re.sub("'t",'t',content)
        b = re.sub("'m",'m',a) 
        c = re.sub("'ve",'ve',b) 
        d = re.sub("'ll",'ll',c)
        k = re.sub("'s",'s',d)
        f = re.sub("'re",'re',k)
        content=re.sub('[!"#$%&()*+-/:,;<=>?@\\^_`{|}~\t\n]',' ', f)  
        content = tokenizer2.tokenize(content)
        content_1 =''
        for i in range(0,len(content)): 
            if content[i] not in en_stop:
                content_1 = content_1 + ' ' + Snow_stemmer.stem(content[i])            
        msgLength = len(tokenizer2.tokenize(content_1))
        return content_1.lstrip(), msgLength 

    def External_URL(self , content):
        URL = '' ; Result = '' ; 
        S_string = 'www.' ; E_string = ' ' ; #E_string = '.html' ; E_string_ = '.com' 
        content = content.lower() ; #print(content.find(S_string))
        if content.find(S_string) > 0:
            S = content.find(S_string)
            if content[S:].find(E_string) > 0 :
                URL = content[S:S+content[S:].find(E_string)+len(E_string)] ; #print(URL)
        if len(URL) > 0 :
            URL = 'http://' + URL ; #print(URL)
            try:
                f = opener.open(URL);   
                Result = True
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print(e.reason)
                Result = True #Result = False
        else:
            Result = False
        return Result, URL

