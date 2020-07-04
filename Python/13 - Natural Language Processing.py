# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 19:11:34 2020

@author: bayha
"""
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')

raw_data= pd.read_csv("twitter-nlp.csv",encoding="latin-1")

data = pd.concat([raw_data.gender,raw_data.description],axis=1)

data.gender = [0 if each=="male" else 1 if each=="female" else 2 for each in data.gender] # male 0 , female 1 , brand 2

data.dropna(axis=0,inplace=True)

lemma = nltk.WordNetLemmatizer()

description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    description.lower()
    description = nltk.word_tokenize(description)
    #description = [word for word in description if not word in set(stopwords.words("english"))]
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)






