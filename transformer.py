"""
Title: Text classification with Transformer
"""

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import  Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers


import pandas as pd
import os
import yaml
from tqdm import tqdm
import re
import string
import sys
import unidecode
import numpy as np



"""
## Implement multi head self attention as a Keras layer
"""


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
"""
## Implement a Transformer block as a layer
"""


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

"""
## Implement embedding layer

Two seperate embedding layers, one for tokens, one for token index (positions).
"""

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

"""
## Implement TRansformerClassifier
"""


class TransformerClassifier():

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
        else:
            self.load_model()
        

        if self.config.create_submit:
            self.create_submission()
    
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
        self.tokenizer = Tokenizer(num_words=self.config.vocab_size)
        self.tokenizer.fit_on_texts(self.x.values)
        X = self.tokenizer.texts_to_sequences(self.x.values)
        self.X = pad_sequences(X,maxlen=self.config.max_len)

        if self.config.debug:
            print('-------------------------------------------------')
            print('-------------------------------------------------')
            print('Tokens created .')
        
    def create_model(self):

        inputs = layers.Input(shape=(self.config.max_len,))
        embedding_layer = TokenAndPositionEmbedding(self.config.max_len, self.config.vocab_size, self.config.embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(self.config.embed_dim, self.config.num_heads, self.config.ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(self.config.dropout)(x)
        x = layers.Dense(self.config.dim_last_dense, activation="relu")(x)
        x = layers.Dropout(self.config.dropout)(x)
        outputs = layers.Dense(self.config.labels, activation="softmax")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        if self.config.debug:
            print(self.model.summary())
    
    def train(self):

        import wandb
        from wandb.keras import WandbCallback

        os.environ['WANDB_ANONYMOUS'] = 'allow'
        wandb.init(project=self.config.wandb_project)


        self.model.compile(loss=self.config.loss,optimizer=self.config.optimizer , metrics=self.config.metrics)

        self.model.fit(
            self.X,
            self.y,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            callbacks=[WandbCallback()],shuffle=True
        )

        self.model.save_weights(self.config.model_name)

        if self.config.debug:
            print('------------------------------------')
            print('weights saved {} ...'.format(self.config.model_name))

    def load_model(self):
        self.model.load_weights(self.config.model_name)
    
    def create_submission(self):
        if not os.path.exists(self.config.test_path):
            raise Exception("data test not found !")
        test = pd.read_csv(self.config.test_path,encoding='utf8',index_col=0 ,sep=',')

        if self.config.clean_data :
            tqdm.pandas(desc='Cleaning dataset from noise ')
            test['text'] = test['text'].progress_apply(lambda x: self.clean_text(x))
            test['label'] = 0

        tqdm.pandas(desc='Predict submission ... ')
        
        test['label'] = test['text'].progress_apply(lambda x: self.predict(x))
        test['label'].astype('int32')

        test = test.drop(['text'] , axis=1)
        test.to_csv(self.config.submit_path)

    def predict(self,line):
        X = self.tokenizer.texts_to_sequences([line])

        X = pad_sequences(X,maxlen=self.config.max_len)
        prediction = self.model.predict(X)

        if self.config.debug:

            print('------------------------------------')

            print('------------------------------------')
            print(line)
            print(prediction , self.config.result_labels[np.argmax(prediction)] )
            print('------------------------------------')
        return self.config.result_labels[np.argmax(prediction)] 



class Utils():
    pass


class Config():
    def __init__(self,data_path='Train.csv',submit_path='Submit.csv',batch_size=32,
                 dim = 128,debug = False, optimizer = 'adam' , loss='categorical_crossentropy' , 
                 metrics = ['accuracy'] , wandb_project = 'ai4d' , validation_split = 0.2,
                 epochs=1000,train = False , model_type = 'LSTM',model_name = 'model.h5',
                 max_len = 250 , vocab_size = 20000 , create_submit = False ,dropout = 0.1,
                 patience = 10 , clean_data = False , fasttext = 'fast.train' , fasttext_valid = 'fast.valid' , duration = 600,

                 embed_dim = 32, num_heads= 2 , ff_dim = 32 , labels = 3 ,dim_last_dense = 20 , test_path = 'Test.csv' , result_labels = [-1,0,1]
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
        self.vocab_size = vocab_size
        self.create_submit = create_submit
        self.dropout = dropout
        self.patience = patience
        self.clean_data = clean_data
        self.fasttext = fasttext
        self.fasttext_valid = fasttext_valid
        self.duration = duration

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.labels = labels
        self.dim_last_dense = dim_last_dense
        self.test_path = test_path
        self.result_labels = result_labels