from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
# Create your views here.
from newspaper import Article
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, LSTM, Embedding, Reshape,Bidirectional
from keras.models import load_model, model_from_json

from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import urllib

from urllib.request import urlretrieve

from os import mkdir, makedirs, remove, listdir

from collections import Counter
from keras.engine.topology import Layer
import keras.backend as K
from keras import initializers
from keras.models import model_from_yaml
import pickle
from keras.backend import clear_session

from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
#import argparse
import cv2
import imutils
import pytesseract
from PIL import Image
import sys
import scipy


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
    




def home(request):
    if request.method == 'POST':
        if request.POST.get('link') != '':
            text=request.POST.get('link')
            # text = text.encode('utf-8')
            print(str(text))
            sentence = str(text).lower()
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
            print(rel)


            result=rel
            return render(request,'result.html', {'result': result})
        else:
            file = request.POST.get('file')
            print(file[-3:])
            if(file[-3:] == 'jpg', 'png','JPG' or file[-4:] == 'jpeg'):
                image = cv2.imread(file)
                ratio = image.shape[0] / 500.0
                orig = image.copy()
                image = imutils.resize(image, height = 500)

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                edged = cv2.Canny(gray, 75, 200)
                screenCnt=0

                cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]


                for c in cnts:
                    
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                    if len(approx) == 4:
                        screenCnt = approx
                        break
                    else:
                        break
                    

                if(len(approx)==4):
                    ctr = np.array(screenCnt).reshape((-1,1,2)).astype(np.int32)
                    cv2.drawContours(image, [ctr], -1, (0, 255, 0), 2)
                    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
                    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
                    warped = (warped > T).astype("uint8") * 255
                    cv2.imshow("Original", imutils.resize(orig, height = 650))
                    cv2.imshow("Scanned", imutils.resize(warped, height = 650))
                    text = pytesseract.image_to_string(orig, lang='eng')

                    cv2.waitKey(0)

                else:

                    text = pytesseract.image_to_string(orig, lang='eng')
                    # cv2.imshow("Original", imutils.resize(orig, height = 650))

                    # cv2.waitKey(0)
                
                text = text.replace("\r","")
                text = text.replace("\n","")
                # j=1
                # text2=""
                # for character in text:
                #     if(character=='\n'):
                #         j+=1
                #     if(j>3 and j<8):
                #         if(character != '\n' ):
                #             text2 = text2 + character
                # print(text2)
                # text = text.encode('utf-8')
                
                sentence = str(text[0:300]).lower()
                # print("Image result" + sentence)
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
                print(rel)


                result=rel
                return render(request,'result.html', {'result': result})  
            else:
                with open(file, 'r') as f:
                    text = f.read()
                result=''
            return render(request,result, {'result': result})
    return render(request, 'home.html')


def result(request, result):
    return render(request, 'result.html', {'result': result})

def pie(request):
    return render(request, 'pie.html')
    pass

def about(request):
    return render(request, 'about.html')
    pass
