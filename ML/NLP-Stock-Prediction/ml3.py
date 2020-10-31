import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import pickle
import json
tqdm.pandas()

df = pd.read_pickle("Pickles/lemmatized.pkl")

import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, GRU,CuDNNGRU,Input, LSTM, Embedding, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, TimeDistributed, BatchNormalization
from keras.layers import concatenate as lconcat
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K


#sess_config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from sklearn.metrics import roc_auc_score
from keras.utils import np_utils,plot_model, multi_gpu_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#### Define number of words, and embedding dimensions
max_words = 34603
embed_dim = 100

def load_embeddings(vec_file):
    print("Loading Glove Model")
    f = open(vec_file,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done. {} words loaded!".format(len(model)))
    return model

def tokenize_and_pad(docs,max_words=max_words):
    global t
    t = Tokenizer()
    t.fit_on_texts(docs)
    docs = pad_sequences(sequences = t.texts_to_sequences(docs),maxlen = max_words, padding = 'post')
    global vocab_size
    vocab_size = len(t.word_index) + 1
    
    return docs

def oversample(X,docs,y):
    # Get number of rows with imbalanced class
    target = y.sum().idxmax()
    n = y[target].sum()
    # identify imbalanced targets
    imbalanced = y.drop(target,axis=1)
    #For each target, create a dataframe of randomly sampled rows, append to list
    append_list =  [y.loc[y[col]==1].sample(n=n-y[col].sum(),replace=True,random_state=20) for col in imbalanced.columns]
    append_list.append(y)
    y = pd.concat(append_list,axis=0)
    # match y indexes on other inputs
    X = X.loc[y.index]
    docs = pd.DataFrame(docs_train,index=y_train.index).loc[y.index]
    assert (y.index.all() == X.index.all() == docs.index.all())
    return X,docs.values,y


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('items')),columns=mlb.classes_,),sort=False,how="left")
