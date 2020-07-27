# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 10:30:24 2020

@author: admin
"""

#importing libraries
import numpy as np
import pickle
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
from tensorflow.python.keras.backend import set_session
sess = tf.Session()
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

set_session(sess)

#loading model
model=load_model('proj_cnn.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/cnnmodel', methods =['GET','POST'])
def cnnmodel():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
            print("prediction",preds)
            
        index = ['cardboard','glass','metal','paper','plastic','trash']
        
        text = "The prediction is : " + str(index[preds[0]])
        
    return render_template('index.html',abc=text)
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
 
