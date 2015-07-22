========================
Relevancer Python Script
========================

All code can be reached from `GitHub <https://github.com/cengelif/Relevancer/blob/master/relevancer.py/>`_

Beginning
------------

.. rubric:: Imports;

We are using these::

	import configparser
	import sys
	import pymongo as pm
	import logging
	import argparse
	import json
	import datetime
	import time
	import math as m
	import pandas as pd
	import numpy as np
	import scipy as sp
	import re
	
	from collections import Counter, OrderedDict, namedtuple
	from html.parser import HTMLParser
	from pymongo import MongoClient
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.cluster import AgglomerativeClustering
	from sklearn.cluster import KMeans, MiniBatchKMeans
	from sklearn import metrics
	
.. rubric:: Logging;

We keep the background information of script in logging file with using following code::

	logging.basicConfig(filename=myapp.log,
                           	 format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                           	 datefmt='%d-%m-%Y, %H:%M',
                           	 level=logging.INFO)
                           	 
.. warning:: Your all logging informations are kept in myapp.log file. Also you can see the last loggings on terminal using with this command line::

	>> tail -F myapp.log

Database
-----------

.. rubric:: Config Parser;

Configuration file keep your private information that is using for authentication so you should call ConfigParser() firstly::

	config = configparser.ConfigParser()
	config.read(configfile)

.. rubric:: MongoLab OAuth;

Here you can access the configuration file::

	client_host = config.get('mongodb', 'client_host')
	client_port = int(config.get('mongodb', 'client_port'))
	db_name = config.get('mongodb', 'db_name')
	if (coll_name == None):
		coll_name = config.get('mongodb', 'coll_name')
	if config.has_option('mongodb', 'user_name'):
		user_name = config.get('mongodb', 'user_name')
	if config.has_option('mongodb', 'passwd'):
		passwd = config.get('mongodb', 'passwd')   
		
.. rubric:: Connecting to Database;

You can connect to database with the following code::

	try:
		connection = pm.MongoClient(client_host, client_port)
		rlvdb = connection[db_name]  #Database
		if ('user_name' in locals()) and ('passwd' in locals()):
			rlvdb.authenticate(user_name, passwd)
		rlvcl = rlvdb[coll_name] #Collection
		logging.info('Connected to Database')
	except Exception:
		logging.error("Unexpected error:"+ str(sys.exc_info()))
		sys.exit("Database connection failed!")
		pass
		
Reading Data
--------------

.. rubric:: Read data from text file;

Scripts gets the tweets from a text file that is given::

	ftwits = []
	with open(myjsontweetfile) as jfile:
	for i, ln in enumerate(jfile):
		t = json.loads(ln)	
		t["created_at"] = datetime.datetime.strptime(t["created_at"],"%a %b %d %H:%M:%S +0000 %Y")
		ftwits.append(t) 
	
.. rubric:: Read data from database;

Scripts gets the all tweets` information from database::

	ftwits = []
	for i, t in enumerate(rlvcl.find(mongo_query)):
   		t["created_at"] = datetime.datetime.strptime(t["created_at"],"%a %b %d %H:%M:%S +0000 %Y")
   		ftwits.append(t)
   		
.. note:: 'rlvcl' is keeping the collection name.

restrict line numbers for test::

	if i == tweet_count: 
		break
		
.. note:: 'tweetlist' is a list which contains all data that are getting from database. tweetlist = read_json_tweets_database() is the call method.
   		
Processing Data and Output
----------------------------

.. rubric:: Get fields from database;

To get fields which are tweets` id and text from database and also create a list that contains annotated ids::

	def read_json_tweet_fields_database(rlvcl, mongo_query, tweet_count=-1, annotated_ids=[]):
		ftwits = []
		for i, t in enumerate(rlvcl.find(mongo_query, { 'text': 1, 'id_str': 1, '_id':0 })):
   			if i != tweet_count and t['id_str'] not in annotated_ids:
   			ftwits.append(t) 
   			
.. rubric:: Get tweet ids from collection;

The following code, create a list and append all tweet ids that are on database::

	tw_id_list = []
	for clstr in rlvcl.find ({}, {"id_str" : 1, "_id":0}):
		tw_id_list.append(clstr["id_str"])

.. rubric:: Create dataframe;

Create a data frame as TweetsDF::	

	dataframe = pd.DataFrame(tweetlist)
	logging.info("columns:"+str(dataframe.columns))

	tweetsDF = create_dataframe(tweetlist) 
	
.. rubric:: Tokenizer;
::	

	if no_tok:
		twtknzr = Twtokenizer()
		tweetsDF = twtknzr.tokenize_df(tweetsDF, texcol='text', rescol=tok_result_col, addLowerTok=False)
		tweetsDF[tok_result_lower_col] = tweetsDF[tok_result_col].str.lower()
		
.. warning:: 'no_tok' is used by defult as no_tok=False.

Eliminate retweets::

	if elimrt:
		rttext = ~tweetsDF[tok_result_lower_col].str.contains(r"\brt @")
		rtfield = tweetsDF["is_retweet"]==False
		tweetsDF["is_notrt"] = rtfield.values & rttext.values # The default setting is to eliminate retweets
		tweetsDF = tweetsDF[tweetsDF.is_notrt]   
   
.. rubric:: Get vectorizer and multinominalNB classifier;

With the following code, we can get the vectorizer, classifier and label for a new tweet::

	word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, norm='l2', min_df=freqcutoff, token_pattern = my_token_pattern, sublinear_tf=True)
	X2_train = word_vectorizer.fit_transform(tweets_as_text_label_df.text.values)
   
   	y_train=tweets_as_text_label_df.label.values
   	
   	MNB = MultinomialNB(alpha=.1)
	MNB.fit(X2_train, y_train)
	
	vect_and_classifier={'vectorizer' : word_vectorizer, 'classifier' : MNB}
	
.. rubric:: Pickle file;

Write to a pickle file::

	with open(pickle_file, 'wb') as f:
        	pickle.dump(vect_and_classifier, f, pickle.HIGHEST_PROTOCOL)
        	print("Pickle file was written to", pickle_file)	
	
.. rubric:: Different clustering;

In the following code, if the data set bigger than the threshold, the scripts use MiniBatchKMeans to increase the performance::

	if len(tweetsDF) < 1000000:
		km = KMeans(n_clusters=n_clust, init='k-means++', max_iter=500, n_init=n_initt) #, n_jobs=16
		logging.info("The data set is small enough to use Kmeans")
	else: 
		km = MiniBatchKMeans(n_clusters=n_clust, init='k-means++', max_iter=500, n_init=n_initt, batch_size=1000)
		logging.info("The data set is BIG, MiniBatchKMeans is used.")



.. automodule:: LamaEventsScr
    :members:
    :undoc-members:
    :show-inheritance:

	
	
	
	
	
	
	
	
	
   
   
   
   
