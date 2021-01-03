import pandas as pd
import os
import yaml
from tqdm import tqdm
import re
import string
import sys
import unidecode
from tensorflow import keras
import numpy as np


from keras.preprocessing.text import  Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import  Model , Sequential
from keras.layers import Dense,LSTM,GRU,Input , SpatialDropout1D ,Embedding , Dropout , Bidirectional , GlobalMaxPool1D
from keras.regularizers import l2

import fasttext
import csv
from sklearn.model_selection import train_test_split

class Solution():
    def __init__(self,config='conf.yaml'):
        if os.path.exists(config):
            with open(config , 'r') as c:
                config = yaml.load(c,Loader=yaml.FullLoader)

            self.config = Config(**config)
        else:
            self.config = Config()

        self.input_chars = set()

        if self.config.model_type == 'FASTTEXT':

            if self.config.train:
                self.load_data_for_fasttext()
                self.model = fasttext.train_supervised(input=self.config.fasttext  , autotuneValidationFile= self.config.fasttext_valid , verbose = 1 , autotuneDuration= self.config.duration )
                self.model.save_model(self.config.model_name)
            else :
                self.model = fasttext.load_model(self.config.model_name)

        else:
            self.load_data()
            self.create_tokens()
            self.create_model()

        if self.config.train and self.config.model_type != 'FASTTEXT' :
            self.train()

    def load_data(self):
        if not os.path.exists(self.config.data_path):
            raise Exception("data not found !")
        self.data = pd.read_csv(self.config.data_path,encoding='utf8',index_col=0 ,sep=',')
        
        if self.config.clean_data :
            tqdm.pandas(desc='Cleaning dataset from noise ')
            self.x = self.data['text'].progress_apply(lambda x: self.clean_text(x))
        else:
            self.x = self.data['text']


        self.y = pd.get_dummies(self.data['label'])

        

        if self.config.debug :

            tqdm.pandas(desc='Create input chars')

            self.x.progress_apply(lambda x : self.create_input_chars(x))

            print('-------------------------------------------------')
            print('-------------------------------------------------')
            print(self.x.head(10))
            print('-------------------------------------------------')
            print('-------------------------------------------------')
            print(self.y.head(10))
            print('-------------------------------------------------')
            print('-------------------------------------------------')
            print(self.input_chars)
    
    def load_data_for_fasttext(self):
        if not os.path.exists(self.config.data_path):
            raise Exception("data not found !")
        self.data = pd.read_csv(self.config.data_path,encoding='utf8',index_col=0 ,sep=',')

        self.data = self.data[['label','text']]

        tqdm.pandas(desc='Change label format for fasttext')

        self.data['label'] = self.data['label'].progress_apply(lambda x : self.change_label(x))

        

        if self.config.clean_data :
            tqdm.pandas(desc='Cleaning dataset from noise ')
            self.data['text'] = self.data['text'].progress_apply(lambda x: self.clean_text(x))
 
        self.data.to_csv('fast',encoding='utf8',index=None,header=None , sep=' ', quoting=csv.QUOTE_NONE , escapechar=' ' , quotechar='') 

        self.data = pd.read_csv('fast',header=None,encoding='utf8')

        train , valid = train_test_split(self.data, test_size = self.config.validation_split)

        train = pd.DataFrame(train)

        valid = pd.DataFrame(valid)

        if self.config.debug:
            print('-------------------------------------------------')
            print('-------------------------------------------------')
            print(train.head(10))
            print('-------------------------------------------------')
            print('-------------------------------------------------')
            print(valid.head(10))

        train.to_csv(self.config.fasttext,encoding='utf8',index=None,header=None )
        valid.to_csv(self.config.fasttext_valid,encoding='utf8',index=None,header=None )
    
    def change_label(self,data , reverse = False):

        if reverse:

            if data == '__label__neg':
                return -1
            if data == '__label__neu' :
                return 0
            if data == '__label__pos':
                return 1
        else:

            if data == -1:
                return '__label__neg'
            if data == 0:
                return '__label__neu'
            else:
                return '__label__pos'

    def word_count(self,line):
        return len(line.split())

    def clean_text(self,text):

        arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
        english_punctuations = string.punctuation
        other_noise = '''@£'''
        punctuations_list = arabic_punctuations + english_punctuations + other_noise

        translator = str.maketrans('', '', punctuations_list)
        text = " ".join(text.split())
        text = text.translate(translator)
        text = unidecode.unidecode(text)
        text = text.replace('@','')
        return text.lower()

    def create_input_chars(self,in_text):
        for char in in_text:
            if char not in self.input_chars:
                self.input_chars.add(char)
            
    def create_tokens(self):
        self.tokenizer = Tokenizer(num_words=self.config.max_words)
        self.tokenizer.fit_on_texts(self.x.values)
        self.seq_dict = self.tokenizer.word_index
        self.word_dict = dict((num,val) for (val,num) in self.seq_dict.items() )
        X = self.tokenizer.texts_to_sequences(self.x.values)
        self.X = pad_sequences(X,maxlen=self.config.max_len)

        if self.config.debug:
            print('-------------------------------------------------')
            print('-------------------------------------------------')
            print('Tokens created .')
    
    def create_model(self):
        if self.config.model_type == 'LSTM':
            

            # model = Sequential()
            # model.add(Embedding(len(self.word_dict), self.config.max_words ,input_length = self.X.shape[1]))
            # model.add(LSTM(self.config.dim, return_sequences=True , recurrent_dropout=self.config.dropout))
            # model.add(Dropout(self.config.dropout))
            # model.add(LSTM(self.config.dim, return_sequences=True , recurrent_dropout=self.config.dropout ))
            # model.add(Dropout(self.config.dropout))
            # model.add(LSTM(self.config.dim , recurrent_dropout=self.config.dropout))
            # model.add(Dense(self.config.dim,activation='relu'))
            # model.add(Dense(3,activation='softmax'))

            # model = Sequential()
            # model.add(Embedding(len(self.word_dict), self.config.max_words ,input_length = self.X.shape[1]))
            # model.add(LSTM(self.config.dim, dropout=self.config.dropout , recurrent_dropout=self.config.dropout))
            # model.add(Dropout(self.config.dropout))
            # model.add(Dense(3,activation='softmax'))

            inp = Input(shape=(self.config.max_len,))
            x = Embedding(self.config.max_words, self.config.max_len ,input_length = self.X.shape[1])(inp)
            x = Bidirectional(LSTM(self.config.dim, return_sequences=True, dropout=self.config.dropout, recurrent_dropout=self.config.dropout , kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))(x)
            x = GlobalMaxPool1D()(x)
            x = Dense(self.config.dim, activation="sigmoid")(x)
            x = Dropout(self.config.dropout)(x)
            x = Dense(3, activation='softmax')(x)
            model = Model(inputs=inp, outputs=x)


            self.model = model

        if self.config.model_type == 'GRU':

            model = Sequential()
            model.add(Embedding(len(self.word_dict), self.config.max_words ,input_length = self.X.shape[1]))
            model.add(GRU(self.config.dim, return_sequences=True , recurrent_dropout=self.config.dropout))
            model.add(Dropout(self.config.dropout))
            model.add(GRU(self.config.dim, return_sequences=True , recurrent_dropout=self.config.dropout ))
            model.add(Dropout(self.config.dropout))
            model.add(GRU(self.config.dim , recurrent_dropout=self.config.dropout))
            model.add(Dense(self.config.dim,activation='relu'))
            model.add(Dense(3,activation='softmax'))

            self.model = model

        if self.config.model_type == 'MLP':
            model = Sequential()
            model.add(Dense(len(self.word_dict), input_shape=(self.X.shape[1],) , activation="relu"))
            model.add(Dropout(self.config.dropout))
            model.add(Dense(self.config.dim,activation='relu'))
            model.add(Dropout(self.config.dropout))
            model.add(Dense(self.config.dim,activation="relu"))
            model.add(Dropout(self.config.dropout))
            model.add(Dense(self.config.dim , activation="relu"))
            model.add(Dense(3,activation='softmax'))

            self.model = model

        if self.config.debug:
            print(self.model.summary())

    def train(self):

        import wandb
        from wandb.keras import WandbCallback

        os.environ['WANDB_ANONYMOUS'] = 'allow'
        wandb.init(project=self.config.wandb_project)

        early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss',patience=self.config.patience)

        self.model.compile(loss=self.config.loss,optimizer=self.config.optimizer , metrics=self.config.metrics)

        self.model.fit(
            self.X,
            self.y,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            callbacks=[WandbCallback(),early_stop_callback],shuffle=True
        )

        self.model.save_weights(self.config.model_name)
        self.model.save(os.path.join(wandb.run.dir, self.config.model_name))
        if self.config.debug:
            print('------------------------------------')
            print('weights saved {} ...'.format(self.config.model_name))
    
    def create_submission(self,test_path):
        if not os.path.exists(test_path):
            raise Exception("data test not found !")
        test = pd.read_csv(test_path,encoding='utf8',index_col=0 ,sep=',')

        if self.config.clean_data :
            tqdm.pandas(desc='Cleaning dataset from noise ')
            test['text'] = test['text'].progress_apply(lambda x: self.clean_text(x))
            test['label'] = 0

        tqdm.pandas(desc='Predict submission ... ')
        
        test['label'] = test['text'].progress_apply(lambda x: self.predict(x))

        test = test.drop(['text'] , axis=1)
        test.to_csv(self.config.submit_path)

    def predict(self,line):
        r = self.model.predict(line)
        return self.change_label(r[0][0],reverse = True)



class Config():
    def __init__(self,data_path='Train.csv',submit_path='Submit.csv',batch_size=128,
                 dim = 128,debug = False, optimizer = 'NAdam' , loss='categorical_crossentropy' , 
                 metrics = ['accuracy'] , wandb_project = 'ai4d' , validation_split = 0.2,
                 epochs=1000,train = False , model_type = 'LSTM',model_name = 'model.h5',
                 max_len = 100 , max_words = 20000 , create_submit = False ,dropout = 0.2,
                 patience = 10 , clean_data = False , fasttext = 'fast.train' , fasttext_valid = 'fast.valid' , duration = 600
                ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.dim = dim
        self.epochs = epochs
        self.debug = debug
        self.model_type = model_type
        self.model_name = model_name
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.wandb_project = wandb_project
        self.submit_path = submit_path
        self.validation_split = validation_split
        self.train = train
        self.max_len = max_len
        self.max_words = max_words
        self.create_submit = create_submit
        self.dropout = dropout
        self.patience = patience
        self.clean_data = clean_data
        self.fasttext = fasttext
        self.fasttext_valid = fasttext_valid
        self.duration = duration