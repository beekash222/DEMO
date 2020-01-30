import numpy as np
import pickle as pkl
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
import lightgbm as lgb
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from flask import Flask, request,render_template,redirect,url_for
from flask_cors import CORS
from sklearn.externals import joblib
import pickle
import flask
import urllib
import pandas as pd
import numpy as np
import gzip
import re
import os
from scipy.sparse import hstack
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
import lightgbm as lgb
from textblob import TextBlob
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from flask_restful import reqparse, abort, Api, Resource
from flask import Flask, flash, request, redirect, url_for
from flask import Flask, url_for, send_from_directory, request
import gunicorn
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup as bs
from youtube_transcript_api import YouTubeTranscriptApi
import re
                   
app = Flask(__name__)
CORS(app)

app=flask.Flask(__name__,template_folder='templates')


@app.route('/')
def main():
    return render_template('main.html')



@app.route('/predict',methods=['GET','POST'])
def predict():
    video_id = request.form['video_id']
    predictions = YouTubeTranscriptApi.get_transcript(video_id)
    return render_template('main.html', prediction_text= predictions)


    
if __name__=="__main__":
#    port=int(os.environ.get('PORT',5000))
     app.run(host='0.0.0.0', port=8080, debug=True)

    
