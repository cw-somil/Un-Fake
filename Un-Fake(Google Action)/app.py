
import os
import json

import tornado.ioloop
import tornado.log
import tornado.web

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

import jwt
import requests
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, LSTM, Embedding, Reshape,Bidirectional
from keras.models import load_model, model_from_json

from sklearn.model_selection import train_test_split

import os
import urllib

from urllib.request import urlretrieve

from os import mkdir, makedirs, remove, listdir

from collections import Counter
from keras.engine.topology import Layer
import keras.backend as K
from keras import initializers
from keras.models import model_from_yaml

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

class Embedding2(Layer):

    def __init__(self, input_dim, output_dim, fixed_weights, embeddings_initializer='uniform', 
                 input_length=None, **kwargs):
        kwargs['dtype'] = 'int32'
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(Embedding2, self).__init__(**kwargs)
    
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.fixed_weights = fixed_weights
        self.num_trainable = input_dim - len(fixed_weights)
        self.input_length = input_length
        
        w_mean = fixed_weights.mean(axis=0)
        w_std = fixed_weights.std(axis=0)
        self.variable_weights = w_mean + w_std*np.random.randn(self.num_trainable, output_dim)

    def build(self, input_shape, name='embeddings'):        
        fixed_weight = K.variable(self.fixed_weights, name=name+'_fixed')
        variable_weight = K.variable(self.variable_weights, name=name+'_var')
        
        self._trainable_weights.append(variable_weight)
        self._non_trainable_weights.append(fixed_weight)
        
        self.embeddings = K.concatenate([fixed_weight, variable_weight], axis=0)
        
        self.built = True

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        out = K.gather(self.embeddings, inputs)
        return out

    def compute_output_shape(self, input_shape):
        if not self.input_length:
            input_length = input_shape[1]
        else:
            input_length = self.input_length
        return (input_shape[0], input_length, self.output_dim)
    

import pickle
with open('objs.pkl','rb') as f:  # Python 3: open(..., 'rb')
    fixed_weights2,word2num_length,X_test,y_test,word2num= pickle.load(f)
    
new_model = Sequential()
new_model.add(Embedding2(word2num_length, 50,
                    fixed_weights= fixed_weights2)) # , batch_size=batch_size
new_model.add(Bidirectional(LSTM(64)))
new_model.add(Dense(1, activation='sigmoid'))

# rmsprop = keras.optimizers.RMSprop(lr=1e-4)

new_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
new_model.load_weights('model.h5')

score = new_model.evaluate(X_test,y_test)
print("%s: %.2f%%" % (new_model.metrics_names[1], score[1]*100))

API_KEY = '867f65dc3f48036dc7b738a7fb2aaf36'
PROJECT_ID = 'unfake-5ffc7'

class WeatherHandler(tornado.web.RequestHandler):
  def start_conversation (self):
    response = {
      'expectUserResponse': True,
      'expectedInputs': [
        {
          'possibleIntents': {'intent': 'actions.intent.TEXT'},
          'inputPrompt': {
            'richInitialPrompt': {
              'items': [
                {
                  'simpleResponse': {
                    'ssml': '<speak>Welcome to News Companion. What news would you like to analyze?</speak>'
                  }
                }
              ]
            }
          }
        }
      ]
    }

    self.set_header("Content-Type", 'application/json')
    self.set_header('Google-Assistant-API-Version', 'v2')
    self.write(json.dumps(response, indent=2))
  
  def get_weather(self, city):
   sentence = str(city).lower()
   sentence_num = [word2num[w] if w in word2num else word2num['<Other>'] for w in sentence.split()]
   sentence_num = [word2num['<PAD>']]*(0) + sentence_num
   sentence_num = np.array(sentence_num)
   reliability = (new_model.predict(sentence_num[None,:])).flatten()[0] * 100
   reliability = int(reliability)
   if(reliability <15):
       rel = "Not Reliable :("
   elif(reliability > 15 and reliability < 60):
       rel = "Reliable :)"
   elif(reliability > 60):
       rel = " Very Reliable"
       
   response = {
        'expectUserResponse': False,
        'finalResponse': {
          'richResponse': {
            'items': [
              {
                'simpleResponse': {
                  'ssml': '<speak>The news you sent is {}</speak>'.format(rel)
                }
              }
            ]
          }
        }
      }

   self.set_header("Content-Type", 'application/json')
   self.set_header('Google-Assistant-API-Version', 'v2')
   self.write(json.dumps(response, indent=2)) 
      
      
      
#  def get_weather (self, city):
#    api_response = requests.get(
#      'http://api.openweathermap.org/data/2.5/weather',
#      params={'q': city, 'APPID': API_KEY}
#    )
#    data = api_response.json()
#    if 'main' not in data:
#      response = {
#        'expectUserResponse': False,
#        'finalResponse': {
#          'richResponse': {
#            'items': [
#              {
#                'simpleResponse': {
#                  'ssml': '<speak>City not found - meow!</speak>'
#                }
#              }
#            ]
#          }
#        }
#      }
#
#    else:
#      temp = round(1.8 * (data['main']['temp'] - 273) + 32)
#
#      response = {
#        'expectUserResponse': False,
#        'finalResponse': {
#          'richResponse': {
#            'items': [
#              {
#                'simpleResponse': {
#                  'ssml': '<speak>Its {} degrees in {}. Now, what news would you like to analyze</speak>'.format(temp, city)
#                }
#              }
#            ]
#          }
#        }
#      }
#
#    self.set_header("Content-Type", 'application/json')
#    self.set_header('Google-Assistant-API-Version', 'v2')
#    self.write(json.dumps(response, indent=2))

  def get (self):
    city = self.get_query_argument('city', '')
    if city:
      self.get_weather(city)

    else:
      self.start_conversation()

  def post (self):
    token = self.request.headers.get("Authorization")
    jwt_data = jwt.decode(token, verify=False)
    print(jwt_data['aud'])
  
    if jwt_data['aud'] != PROJECT_ID:
      self.set_status(401)
      self.write('Token Mismatch')

    else:
      request = google_requests.Request()
      try:
        # Makes external request, remove if not needed to speed things up
        id_info = id_token.verify_oauth2_token(token, request, PROJECT_ID)
      except:
        self.set_status(401)
        self.write('Token Mismatch')

    data = json.loads(self.request.body.decode('utf-8'))
    intent = data['inputs'][0]['intent']
    print(intent)
    print(data['conversation']['conversationId'])

    if intent == 'actions.intent.MAIN':
      self.start_conversation()

    else:
      city = data['inputs'][0]['arguments'][0]['textValue']
#      if(city == 'Mumbai' and len(city)==6 ):
#        self.get_weather(city)
#      else:
#        self.get_prediction(city)
      self.get_weather(city)


    

def make_app():
  return tornado.web.Application([
    (r"/news", WeatherHandler),
  ], autoreload=True)

if __name__ == "__main__":
  tornado.log.enable_pretty_logging()

  app = make_app()
  app.listen(int(os.environ.get('PORT', '8080')))
  tornado.ioloop.IOLoop.current().start()




       