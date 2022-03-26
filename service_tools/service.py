import re
import numpy as np
from gensim.models import FastText
import joblib

class Preprocessor:
    def __init__(self):
        self.text_cleaning_re = "[^A-Za-z0-9 ]+"
        self.ft_model = FastText.load('models/ft_model.model')
        

    #clean and tokenize text, helper function for get_doc_vec
    def __clean_and_tokenize(self, text):

        text = re.sub(self.text_cleaning_re, '', text)
        #lowercase and simple whitespace tokenizer
        text = text.lower().split()
        return text

    def get_doc_vec(self, text):
        """
        text : raw text entered in input textbox
        """
        text = self.__clean_and_tokenize(text)
        output=np.zeros(64)
        numw=0
        for token in text:
            try:
                output += self.ft_model.wv[token]
                numw+=1
            except:
                pass
        if output[0]==0:
            return output
        else:
            return output/numw

class Predictor:
    def __init__(self):
        self.lgr_model = joblib.load("models/lgr.pkl")
        self.svm_model = joblib.load("models/svm.pkl")

    def predict_sentiment(self, mlmodel, vector):
        """
        mlmodel : string. used to specify prediction model type; available options:'lgr','svm'
        vector : document vector. result from get_doc_vec()
        """
        if mlmodel == 'lgr':
            prediction = self.lgr_model.predict([vector])
        elif mlmodel == 'svm':
            prediction = self.svm_model.predict([vector])
        
        if prediction == 1:
            return "This review seems to be positive😇 !!"
        else:
            return "This review seems to be negative😨 !!"
        