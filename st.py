import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from vncorenlp import VnCoreNLP
annotator = VnCoreNLP('VnCoreNLP-1.1.1.jar', annotators="wseg", max_heap_size='-Xmx500m')
with open('vietnamese_stopwords.txt',encoding='utf8') as f:
    stopwords = f.read().splitlines() 
pat = r'\b(?:{})\b'.format('|'.join(stopwords))
class WordPreprocessing(BaseEstimator, TransformerMixin):
    def fit(self, X_df, y=None):
        return self
    
    def transform(self, X_df, y=None):
        
        out_df = X_df.copy()
        out_df['text'] = out_df['text'].str.lower()
        out_df['text'] = out_df['text'].str.replace(pat, '',regex=True)
        out_df['text'] = out_df['text'].str.replace('\W', ' ',regex=True)
        out_df['text'] = out_df['text'].str.replace('[^\w\s]',' ',regex=True)
        out_df['text'] = out_df['text'].apply(annotator.tokenize)
        out_df['text'] = out_df['text'].apply(lambda x: x[0])
        out_df['domain'] = out_df['domain'].str.replace('\W','',regex=True)  
        
        out_df = out_df.sort_index(axis=1)
        return out_df
def dummy_fun(a):
    return a
col_pipeline = ColumnTransformer(
    transformers=[
        ('tfidf',TfidfVectorizer(analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun, 
    token_pattern=None), 'text'),
    ('ohe',OneHotEncoder(handle_unknown='ignore'),['domain'])
    ]
)
@st.cache
def getData():
    return pd.read_csv('vn_news_226_tdlfr.csv')


def getModel():
    mlp_model = joblib.load('mlp_model.sav')
    lr_model = joblib.load('lr_model.sav')
    return mlp_model,lr_model

def trainModel(model):
    data_df = getData()
    y_sr = data_df["label"]
    X_df = data_df.drop("label", axis=1)
    model.fit(X_df,y_sr)
    return model
st.set_page_config(page_title="Project03")

st.title("Vietnamese Fake News Detector")

model_select = st.selectbox('Choose model', options=['MLP Classifier','Logistic Regression'],index=0)
new = st.text_area('Write the new here')
domain = st.text_input('Domain of the new above (optional)')

mlp_model,lr_model = getModel()

left,right = st.columns(2)
if left.button('Predict'):
    train_model = None
    if model_select == 'MLP Classifier':
        trainModel = trainModel(mlp_model)
    elif model_select == 'Logistic Regression':
        trainModel = trainModel(lr_model)
    if len(new) > 0 :
        obj = {
            "text":[new],
            "domain":[domain]
        }
        result = trainModel.predict(pd.DataFrame.from_dict(obj))
        if result[0] ==0:
            right.write("Real")
        elif result[0]==1:
            right.write("Fake")
        else:
            right.write("error")
    else:
        right.write("Please write some new")
