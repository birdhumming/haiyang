import pandas as pd
import numpy as np
import tensorflow
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

df = pd.read_csv("Data/texts_and_fin.csv")
df.head()

df['release_date'] = df['release_date'].map(lambda x: pd.to_datetime(x))
df['items'] = df['items'].map(lambda x: ast.literal_eval(x))

from nltk.corpus import stopwords
stop_words = stopwords.words("english")
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
wordnet_lemmatizer = WordNetLemmatizer()
import string
punctuations = string.punctuation
import matplotlib.pyplot as plt
import seaborn as sns

import dask.dataframe as dd
from dask.multiprocessing import get
from dask.diagnostics import ProgressBar

def cleanup_text(doc, logging=False):
    doc = re.sub( '\s+', ' ', doc ).strip()
    doc = nlp(doc, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc]
    tokens = [tok for tok in tokens if tok.isalpha()]
    tokens = [tok for tok in tokens if tok not in stop_words and tok not in punctuations]
    tokens_len = len(tokens)
    tokens = ' '.join(tokens)
    return tokens,tokens_len

def nltk_tokenizer(text):
    try:
        tokens = [word for word in word_tokenize(text) if word.isalpha()]
        tokens = list(filter(lambda t: t not in punctuations, tokens))
        tokens = list(filter(lambda t: t.lower() not in stop_words, tokens))
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        filtered_tokens = list(
            map(lambda token: wordnet_lemmatizer.lemmatize(token.lower()), filtered_tokens))
        filtered_tokens = list(filter(lambda t: t not in punctuations, filtered_tokens))
        return filtered_tokens
    except Exception as e:
        raise e

def dask_tokenizer(df):
    df['processed_text'] = df['text'].map(nltk_tokenizer)
    df['text_len'] = df['processed_text'].map(lambda x: len(x))
    return df

pbar = ProgressBar()
pbar.register()
ddata = dd.from_pandas(df, npartitions=20)
#df = ddata.map_partitions(dask_tokenizer).compute(get=get)
df = ddata.map_partitions(dask_tokenizer).compute(scheduler='single-threaded')

df.head()

df.to_csv("Data/lemmatized_text.csv",chunksize=1000)

df = pd.read_csv("Data/lemmatized_text.csv")
df.head()
#df.drop(['Unnamed: 0','Unnamed: 0.1',"doc_name",'txt_link','text'],axis=1,inplace=True)
df.drop(['Unnamed: 0',"doc_name",'txt_link','text'],axis=1,inplace=True)
df['items'] = df['items'].map(lambda x: ast.literal_eval(x))
df['items'] = df['items'].map(lambda items: [' '.join(x.split()) for x in items])

plt.style.use("ggplot")
#df['text_len'].plot.hist(bins=50,normed=True)
df['text_len'].plot.hist(bins=50)

plt.xlabel("Document Length")
plt.show()


int(df['text_len'].quantile(.9))

df['text_len'].describe()

df['ticker'].nunique()

sns.countplot(y=df['GICS Sector'])
plt.savefig("Graphs/sectors.png",format="png")
plt.show()

#Count plot of signals
#sns.countplot(df['signal'])
#plt.show()

df['release_date'] = df['release_date'].map(lambda x: pd.to_datetime(x))
sns.countplot(pd.DatetimeIndex(df['release_date']).year)
plt.savefig("Graphs/year_balances.png",format="png")
plt.show()
