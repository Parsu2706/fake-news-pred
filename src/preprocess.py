import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


from src.data_loader import load_data
import re


def preprocessing(data : pd.DataFrame) ->pd.DataFrame:


    df = data.copy()
    if df.duplicated().sum() > 0:
        df.drop_duplicates(keep='first' , inplace=True)
    else:
        print("Zero Duplicates")
    
    return df
    

class text_cleaner:
    def __init__(self , df : pd.DataFrame):
        self.df = df

    def cleaner(self) -> pd.DataFrame:

        try:
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            def cleaned_text(text):
                text = text.lower()
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'[^a-zA-Z]', ' ', text)           
                text = re.sub(r'\s+', ' ', text).strip()         
                words = text.split()                             
                cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
                return ' '.join(cleaned_words)                   
            self.df["clean_text"] = self.df["text"].astype(str).apply(cleaned_text)
            return self.df
        

        except Exception as e:
            
            print(f"[ERROR] Cleaning failed: {e}")
        return self.df
