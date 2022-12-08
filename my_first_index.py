from flask import  Flask, jsonify, request
# from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
import json 

# import numpy as np
# import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from urllib.parse import quote
import json
import urllib.request
# import print
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
# import time
# import sys
# import os
# import sklearn.metrics.pairwise
from tqdm import tnrange
from sklearn.metrics import jaccard_score
import scipy
# import re
from sentence_transformers import SentenceTransformer

      

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/": {"origins": "*"}})




@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"



@app.route('/getMsg', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def getReply():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        jsonIn = request.json
        print(jsonIn)
        botRes = genBotResponse(jsonIn['query'], jsonIn['topics']) 
        temp1 = generateReply(botRes)
        json_object = json.dumps(temp1, indent = 4) 
        

        return json_object

    else:
        return 'Content-Type not supported!'

def generateReply(botRes):
    temp = {
        'content': botRes,
        'timestamp': '',
        'avatar': '',
        'isBot': True
    }

    return temp    


def genBotResponse(inQuery, inTopic):
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')

        
    # with open('count_vect', 'rb') as f:
    #     cv_pred_2 = pd.read_pickle(f)   #pickle.load(f)
    # with open('text_classifier', 'rb') as training_model:
    #     model = pd.read_pickle(training_model)

    # objects = []
    # with (open("myfile", "rb")) as openfile:
    #     while True:
    #         try:
    #             objects.append(pickle.load(openfile))
    #         except EOFError:
    #             break
    queries = inQuery
    topic = inTopic
    text= [queries]

    print('topic ::  ', topic)
    # ser = pd.Series(text)
    # text =  str(text[0]).lower()
    # m1 = cv_pred_2.transform(ser)
    # predictions = model.predict(m1)
    # #print(predictions)

    # if predictions == 'reddit':
    #     core_name = "reddit"
    # else:
    #     core_name = "chitchat2"

    # #print(core_name)

    localhost = "http://34.125.137.204:8983/solr/"
    select_q = "/select?q="

    #text = "Nick likes to play football, however he is not too fond of tennis."
    query_tokens = word_tokenize(queries)
    tokens_without_sw = [word for word in query_tokens if not word in stopwords.words()]
    filtered_query = (" ").join(tokens_without_sw)

    # inurl = f'http://34.125.137.204:8983/solr/{core_name}/select?q={quote(text)}&fl=id%2Cscore&wt=json&indent=true&rows=20'

    #inurl = f'http://34.125.137.204:8983/solr/chitchat2/select?q=message:{quote(query)}&fl=id%2C%20score%2Creply&wt=json&indent=true&rows=20&q.op=OR'

    #inurl_one = f'http://34.125.137.204:8983/solr/{core_name}/select?indent=true&q.op=OR&q=message%3A%20%22{quote(query)}%22'


    inurl_chitchat = f'http://34.125.137.204:8983/solr/chitchat3/select?indent=true&q.op=OR&q=message%3A%20%22{quote(queries)}%22'
    #inurl_reddit =  f'http://34.125.137.204:8983/solr/reddit/select?indent=true&q.op=OR&q=message%3A%20%22{quote(filtered_query)}%22'
    inurl_reddit =  f'http://34.125.137.204:8983/solr/reddit1/select?indent=true&q.op=AND&q=message%3A%20%22{quote(filtered_query)}%22%0Atopic%3A(%22{quote(topic)}%22)'
    print("REDDIT URL       ",inurl_reddit)
    data_chitchat = urllib.request.urlopen(inurl_chitchat)
    data_reddit = urllib.request.urlopen(inurl_reddit)
    ##print(data)
    docs_reddit = json.load(data_reddit)['response']['docs']
    docs_chitchat = json.load(data_chitchat)['response']['docs']
    dataset_chitchat = []
    dataset_reddit = []
    for doc in docs_reddit:
        #document = re.sub(r'\W', ' ', str(doc['reply']))
        # document='"' + document + '"' 
        dataset_reddit.append(str(doc['reply']))
        
    for doc in docs_chitchat:
        #document = re.sub(r'\W', ' ', str(doc['reply']))
        #'"{}"'.format(document)
        # "\"" + document + "\""
        dataset_chitchat.append(str(doc['reply']))

    #print("length of reddit ",len(dataset_reddit))
    #print("length of chitchat ",len(dataset_chitchat))
    # #print(dataset_chitchat)



    #text = ["who is donald trump"]
    text = [queries]
    result_chitchat = ""
    result_reddit = ""
    flag=0
    result=""
    score_chitchat=0
    score_reddit=0

    query_embeddings = embedder.encode(text)
    if  len(dataset_chitchat) != 0:
        corpus_embeddings=embedder.encode(dataset_chitchat)
        closest_n = 1
        for query, query_embedding in zip(text, query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]
            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])
            #print("\n\n======================\n\n")
            #print("Query:", query)
            #print("\nTop 5 most similar sentences in chitchat corpus:")
            for idx, distance in results[0:closest_n]:
                result_chitchat = dataset_chitchat[idx]
                score_chitchat = 1-distance
                # #print(idx)
                # #print(dataset_chitchat[idx], "(Score: %.4f)" % (1-distance))

    if  len(dataset_reddit) != 0:
        corpus_embeddings_reddit=embedder.encode(dataset_reddit)
        closest_n = 1
        for query, query_embedding in zip(text, query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings_reddit, "cosine")[0]
            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])
            #print("\n\n======================\n\n")
            #print("Query:", query)
            #print("\nTop 5 most similar sentences in reddit corpus:")
            for idx, distance in results[0:closest_n]:
                result_reddit = dataset_reddit[idx]
                score_reddit = 1-distance
                # #print(idx)
                # #print(dataset_reddit[idx], "(Score: %.4f)" % (1-distance))
    if len(dataset_chitchat) ==0 and  len(dataset_reddit) == 0:
        result = "Sorry, I did not get you"
        flag=1


    if flag==0:
        if score_chitchat>=score_reddit :
            result=result_chitchat
        
        else :
            result=result_reddit

        if len(topic)!=0:
                result=result_reddit




    #print("reddit score ",score_reddit)
    #print("chitchat score ",score_chitchat)

    #print("reddit string   ",result_reddit)
    #print("chitchat string   ",result_chitchat)
    #print(result)

    return result


if __name__ == '__main__':
   app.run(debug = True, ssl_context='adhoc')