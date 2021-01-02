import pandas as pd
import os
import yaml
from tqdm import tqdm
import re
import string
import sys
import unidecode
from tensorflow import keras


from keras.preprocessing.text import  Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import  Model , Sequential
from keras.layers import Dense,LSTM,GRU,Input , SpatialDropout1D ,Embedding , Dropout


class Solution():
    def __init__(self,config='conf.yaml'):
        if os.path.exists(config):
            with open(config , 'r') as c:
                config = yaml.load(c,Loader=yaml.FullLoader)

            self.config = Config(**config)
        else:
            self.config = Config()

        self.input_chars = set()
        self.load_data()
        self.create_tokens()
        self.create_model()

        if self.config.train:
            self.train()

    def load_data(self):
        if not os.path.exists(self.config.data_path):
            raise Exception("data not found !")
        self.data = pd.read_csv(self.config.data_path,encoding='utf8',index_col=0 ,sep=',')
        tqdm.pandas(desc='Cleaning dataset from noise ')

        self.x = self.data['text'].progress_apply(lambda x: self.clean_text(x))
        self.y = pd.get_dummies(self.data['label'])

        

        if self.config.debug :

            self.config.max_len = self.x.apply(lambda x: self.word_count(x)).max()

            tqdm.pandas(desc='Create input chars')

            self.x.progress_apply(lambda x : self.create_input_chars(x))
            
            print('-------------------------------------------------')
            print('-------------------------------------------------')
            print('max len : {}'.format(self.config.max_len))
            print('-------------------------------------------------')
            print('-------------------------------------------------')
            print(self.x.head(10))
            print('-------------------------------------------------')
            print('-------------------------------------------------')
            print(self.y.head(10))
            print('-------------------------------------------------')
            print('-------------------------------------------------')
            print(self.input_chars)
    
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
        tokenizer = Tokenizer(num_words=self.config.max_words, split=' ')
        tokenizer.fit_on_texts(self.x.values)
        X = tokenizer.texts_to_sequences(self.x.values)
        X = pad_sequences(X)
        self.X = X

        if self.config.debug:
            print('-------------------------------------------------')
            print('-------------------------------------------------')
            print('Tokens created .')
    
    def create_model(self):
        if self.config.model_type == 'LSTM':
            

            model = Sequential()
            model.add(Embedding(self.config.max_words, self.config.dim,input_length = self.X.shape[1]))
            model.add(SpatialDropout1D( self.config.dropout ))
            model.add(LSTM(self.config.dim, return_sequences=True))
            model.add(Dropout(self.config.dropout))
            model.add(LSTM(self.config.dim, return_sequences=True))
            model.add(Dropout(self.config.dropout))
            model.add(LSTM(self.config.dim))
            model.add(Dense(3,activation='softmax'))

            self.model = model

        if self.config.model_type == 'GRU':
            model = Sequential()
            model.add(Embedding(self.config.max_words, self.config.dim,input_length = self.X.shape[1]))
            model.add(SpatialDropout1D(self.config.dropout))
            model.add(GRU(self.config.dim, return_sequences=True))
            model.add(Dropout(self.config.dropout))
            model.add(GRU(self.config.dim, return_sequences=True))
            model.add(GRU(self.config.dim, dropout=self.config.dropout, recurrent_dropout=self.config.dropout ))
            model.add(Dense(3,activation='softmax'))

            self.model = model

        if self.config.debug:
            print(self.model.summary())

    def train(self):

        import wandb
        from wandb.keras import WandbCallback

        os.environ['WANDB_ANONYMOUS'] = 'allow'
        wandb.init(project=self.config.wandb_project)

        early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

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
    

class Config():
    def __init__(self,data_path='Train.csv',submit_path='Submit.csv',batch_size=128,
                 dim = 128,debug = False, optimizer = 'NAdam' , loss='categorical_crossentropy' , 
                 metrics = ['accuracy'] , wandb_project = 'ai4d' , validation_split = 0.2,
                 epochs=1000,train = False , model_type = 'LSTM',model_name = 'model.h5',
                 max_len = 300 , max_words = 3000 , create_submit = False ,dropout = 0.2
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