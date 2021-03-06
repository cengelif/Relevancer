import configparser
import pickle
import sys
import pymongo as pm
import logging
import json
import datetime
import time
import math as m
from collections import Counter
from html.parser import HTMLParser
import pandas as pd
import numpy as np
import scipy as sp
import re
from operator import itemgetter
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.cluster import entropy
# from sklearn.decomposition import PCA

# Logging
root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler) # remove previous ones. So It will work at module level.
logging.basicConfig(filename='myapp2.log', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%d-%m-%Y, %H:%M', level=logging.INFO)
logging.info("\nScript started")

def connect_mongodb(configfile="myconfig.ini", coll_name=None):

	logging.info("Connecting to the db with parameters:\n configfile:"+str(configfile)+"\t coll_name:"+str(coll_name))

# Config Parser;
	config = configparser.ConfigParser()
	config.read(configfile)
# MongoLab OAuth;
	client_host = config.get('mongodb', 'client_host')
	client_port = int(config.get('mongodb', 'client_port'))
	db_name = config.get('mongodb', 'db_name')
	if coll_name is None:
		coll_name = config.get('mongodb', 'coll_name')
	if config.has_option('mongodb', 'user_name'):
		user_name = config.get('mongodb', 'user_name')
	if config.has_option('mongodb', 'passwd'):
		passwd = config.get('mongodb', 'passwd')
# Connect to database;
	try:
		connection = pm.MongoClient(client_host, client_port)
		rlvdb = connection[db_name]  # Database
		if ('user_name' in locals()) and ('passwd' in locals()):
			rlvdb.authenticate(user_name, passwd)
		rlvcl = rlvdb[coll_name]  # Collection
		logging.info("Connected to the DB successfully.")
	except Exception:
		sys.exit("Database connection failed!")

	return rlvdb, rlvcl

# Global variables;
#min_dist_thres = 0.65  # the smallest distance of a tweet to the cluster centroid should not be bigger than that.
#max_dist_thres = 0.85  # the biggest distance of a tweet to the cluster centroid should not be bigger than that.
#target_labeling_ratio = 0.5  # percentage of the tweets that should be labeled, until this ratio achieved iteration will repeat automatically.
# result_collection = "relevancer_result"
http_re = re.compile(r'https?[^\s]*')
usr_re = re.compile(r'@[^\s]*')
no_tok = False  # no_tok by default. 
my_token_pattern=r"[#@]?\w+\b|[\U00010000-\U0010ffff]"
#active_column = ""

if no_tok:  # create a function for that step!
	tok_result_col = "texttokCap"
else:
	tok_result_col = "text"

class MLStripper(HTMLParser):

	def __init__(self):
		self.reset()
		self.strict = False
		self.convert_charrefs = True
		self.fed = []

	def handle_data(self, d):
		self.fed.append(d)

	def get_data(self):
		return ''.join(self.fed)

class Twtokenizer():

	def __init__(self):
		self.abbreviations = ['i.v.m.', 'a.s.', 'knp.']
		print('init:', self.abbreviations)

	def tokenize(self, tw):
		# abbcheck = False
		newtw = ''
		lentw = len(tw)
		# print(tw)
		for i, c in enumerate(tw):
			if (c in "'`´’‘") and ((i+1 != lentw) and (i != 0)) and ((tw[i-1].isalpha()) or tw[i-1] in "0123456789") and (tw[i+1].isalpha()):
				newtw += ' '+c+' '
			elif (c in "'`´’()+*-") and ((lentw>i+1) and (i > 0)) and (tw[i-1] in ' 0123456789') and (tw[i+1].isalpha()):
				newtw += c+' '
			elif (c in '();>:') and ((lentw>i+1) and (i != 0)) and ((tw[i-1].isalpha()) or (tw[i-1] in '0123456789')) and ((tw[i+1] == ' ') or (i == lentw-1)):
				newtw += ' '+c
			elif (c in '.') and ((lentw>i + 1) and (i != 0)) and ((tw[i-1].isalpha()) or (tw[i-1] in '0123456789')) and ((tw[i+1] == ' ') or (i == lentw-1)) \
								and (newtw.split()[-1]+c not in self.abbreviations):
				newtw += " " + c
			elif (c in "'`´’‘()+*->") and (i == 0) and (lentw > 1) and ((tw[1].isalpha()) or tw[1] in "0123456789"):
				newtw += c+' '
			elif (c in "'`´’‘()+*->") and (i+1 == lentw) and (lentw > 1) and ((tw[i-1].isalpha()) or tw[i-1] in "0123456789"):
				newtw += ' '+c
			elif (c in ",") and ((i != 0) and (i+1 != lentw)) and tw[i-1].isdigit() and tw[i+1].isdigit():  # for 3,5 7,5
				newtw += c
			elif (c in ",&"):
				newtw += " "+c+" "
			elif (c in "â"):  # create a dictionary for character mappings. if c in dict: newtw += dict[c]
				newtw += "a"
			elif (c in "ê"):
				newtw += "e"
			elif (c in "î"):
				newtw += "i"
			elif (c in "ú"):
				newtw += "ü"
			# elif (c in ":")
			else:
				newtw += c
			# print(c in "'`´’()+*-", lentw>i+1, i>0, tw[i-1] == ' 0123456789', tw[i+1].isalpha())
			# if abbcheck:
			# 	print("abbcheck is true:",newtw.split())
			# print(i,c,(c in '.'), ((lentw>i+1) and (i!=0)), ((tw[i-1].isalpha()) or (tw[i-1] in '0123456789')), ((tw[i+1] == ' ') or (i == lentw-1)) \
			# 							and (newtw.split()[-1]+c not in self.abbreviations))
		# print('\n\n')
		return newtw

	def tokenize_df(self, tokdf, texcol="tweet", newtexcol='texttokCap', rescol="ttextlist", addLowerTok=True):
		# concert_table.drop_duplicates()
		# Note
		# tokdf[newtexcol] = tokdf[newtexcol].str.replace("""\xa0"""," ")
		# tokdf[newtexcol] = tokdf[newtexcol].str.replace("\n"," . ")
		tokdf[newtexcol] = tokdf[texcol].copy()
		# tokdf[newtexcol] = tokdf[newtexcol].replace(self.toReplaceDict, regex=True)
		tokdf[newtexcol][tokdf[newtexcol].str.endswith(".")] = tokdf[tokdf[newtexcol].str.endswith(".")][newtexcol].apply(lambda tw: tw[:-1] + ' .') 
		tokdf[newtexcol][tokdf[newtexcol].str.endswith(".'")] = tokdf[tokdf[newtexcol].str.endswith(".'")][newtexcol].apply(lambda tw: tw[:-2] + " . '") 
		tokdf[newtexcol][tokdf[newtexcol].str.startswith("'")] = tokdf[tokdf[newtexcol].str.startswith("'")][newtexcol].apply(lambda tw: "' " + tw[1:])
		tokdf[newtexcol] = tokdf[newtexcol].apply(self.tokenize)
		tokdf[newtexcol] = tokdf[newtexcol].str.strip()
		# tokdf[rescol] = tokdf[newtexcol].str.split()
			
		if addLowerTok:
			tokdf[newtexcol[:-3]] = tokdf[newtexcol].str.lower()

		return tokdf.copy()

def strip_tags(html):

	s = MLStripper()
	s.feed(html)
	return s.get_data()

def set_active_column(active_col="active_text"):
	global active_column
	active_column = active_col

def set_token_pattern(default_token_pattern=r"[#@]?\w+\b|[\U00010000-\U0010ffff]"):
	global my_token_pattern
	my_token_pattern = default_token_pattern

def read_json_tweets_file(myjsontweetfile, reqlang='en'):

	ftwits = []
	lang_cntr = Counter()
	with open(myjsontweetfile) as jfile:
		for i, ln in enumerate(jfile):
			if i == 10000:  # restrict line numbers for test
				break
			t = json.loads(ln)
			lang_cntr[t["lang"]] += 1
			if t["lang"] == reqlang:
				t["created_at"] = datetime.datetime.strptime(t["created_at"], "%a %b %d %H:%M:%S +0000 %Y")
				# if t["created_at"].strftime("%Y-%m-%d") in flood_AnomBreakoutDaysList:
				if "media" in t["entities"]:
					for tm in t["entities"]["media"]:
						if tm["type"] == 'photo':
							t["entity_type"] = 'photo'
							break
				t["entity_hashtags"] = [ehs["text"] for ehs in t["entities"]["hashtags"]]
				t["entity_mentions"] = [ems["screen_name"] for ems in t["entities"]["user_mentions"]]
				t["entity_urls"] = [ers["display_url"] for ers in t["entities"]["urls"]]
				try:
					if "place" in t:
						t["country"] = t["place"]["country"]
				except:
					pass
				if "retweeted_status" in t:
					t["is_retweet"] = True
				else:
					t["is_retweet"] = False

				t["device"] = strip_tags(t["source"])
				t["user_id"] = t["user"]["id_str"]
				t["user_followers"] = t["user"]["followers_count"]
				t["user_following"] = t["user"]["friends_count"]

				t2 = {k:v for k,v in t.items() if k in ["entity_type","entity_hashtags","entity_mentions","entity_urls",\
														"country","created_at","text","in_reply_to_user_id","id_str","user_id",\
														"user_followers","user_following", "coordinates", "is_retweet","device"]}
				# print(i, end = ',')
				ftwits.append(t2)  # .splitlines()
		print("Number of documents per languge:", lang_cntr)
		return ftwits

def read_json_tweets_database(rlvcl, mongo_query, tweet_count=-1, reqlang='en'):

	ftwits = []
	lang_cntr = Counter()

	for i, t in enumerate(rlvcl.find(mongo_query)):
		# time = datetime.datetime.now()
		# logging.info("reading_database_started_at: " + str(time))
		if i == tweet_count:  # restrict line numbers for test
			break
	# t = json.loads(ln)
		lang_cntr[t["lang"]] += 1
		if t["lang"] == reqlang:
			t["created_at"] = datetime.datetime.strptime(t["created_at"], "%a %b %d %H:%M:%S +0000 %Y")
			# if t["created_at"].strftime("%Y-%m-%d") in flood_AnomBreakoutDaysList:
			if "media" in t["entities"]:
				for tm in t["entities"]["media"]:
					if tm["type"] == 'photo':
						t["entity_type"] = 'photo'
						break
			t["entity_hashtags"] = [ehs["text"] for ehs in t["entities"]["hashtags"]]
			t["entity_mentions"] = [ems["screen_name"] for ems in t["entities"]["user_mentions"]]
			t["entity_urls"] = [ers["display_url"] for ers in t["entities"]["urls"]]
			try:
				if "place" in t:
					t["country"] = t["place"]["country"]
			except:
				pass
			if "retweeted_status" in t:
				t["is_retweet"] = True
			else:
				t["is_retweet"] = False
			t["device"] = strip_tags(t["source"])
			t["user_id"] = t["user"]["id_str"]
			t["user_followers"] = t["user"]["followers_count"]
			t["user_following"] = t["user"]["friends_count"]

			t2 = {k:v for k,v in t.items() if k in ["entity_type","entity_hashtags","entity_mentions","entity_urls",\
													"country","created_at","text","in_reply_to_user_id","id_str","user_id",\
													"user_followers","user_following", "coordinates", "is_retweet","device"]}
			# print(i, end=',')
			ftwits.append(t2)  # .splitlines()
		# time2 = datetime.datetime.now()
		# logging.info("reading_database_ended_at: " + str(time2))
	print("Number of documents per languge:", lang_cntr)
	return ftwits

def read_json_tweet_fields_database(rlvcl, mongo_query, read_fields={'text': 1, 'id_str': 1, '_id': 0, 'user': 1}, tweet_count=-1, annotated_ids=[], annotated_users=[]):
	"""
		"annotated_users" may contain user screen names or user_ids.

	"""
	logging.info("reading_fields_started with the following parameters:\nmongoquery="+str(mongo_query)+"\trlvcl(collection)="+str(rlvcl))

	
	ftwits = []
	for i, t in enumerate(rlvcl.find(mongo_query, read_fields)):
		if (i != tweet_count) and (t['id_str'] not in annotated_ids) and ((("user" in t) and (t["user"]["screen_name"] not in annotated_users)) or (("user_id" in t) and (t["user_id"] not in annotated_users))):  # restrict line numbers for test
			# break
			if "retweeted_status" in t:
				t["is_retweet"] = True
			else:
				t["is_retweet"] = False

			if "user" in t:
				t['screen_name'] = t["user"]['screen_name']

			t1 = {k: v for k, v in t.items() if k not in ["user"]} # exclude it after you get the required information. It contains many information in an original tweet.
			ftwits.append(t1)  # .splitlines()
		elif i == tweet_count:
			logging.info("Number of tweets read:"+str(i))
			break

	logging.info("end of database read, example tweet:" + str(ftwits[-1]))

	return ftwits

def get_ids_from_tw_collection(rlvcl):


	time1 = datetime.datetime.now()
	logging.info("get_tw_ids_started_at: " + str(time1))

	print('In get_tw_ids')
	tw_id_list = []		
	for clstr in rlvcl.find({}, {"id_str": 1, "_id": 0}):
		# print("cluster from rlvcl:\n",clstr)
		tw_id_list.append(clstr["id_str"])
		# break

	return tw_id_list

def get_cluster_sizes(kmeans_result, doclist):

	clust_len_cntr = Counter()
	for l in set(kmeans_result.labels_):
		clust_len_cntr[str(l)] = len(doclist[np.where(kmeans_result.labels_ == l)])

	return clust_len_cntr

def create_dataframe(tweetlist):

	dataframe = pd.DataFrame(tweetlist)
	logging.info("columns:" + str(dataframe.columns))
	print(len(dataframe))
	
	if "created_at" in dataframe.columns:
		dataframe.set_index("created_at", inplace=True)
		dataframe.sort_index(inplace=True)
	else:
		logging.info("There is not the field created_at, continue without datetime index.")

	logging.info("Number of the tweets:" + str(len(dataframe)))
	logging.info("Available attributes of the tweets:" + str(dataframe.columns))

	return dataframe

def normalize_text(mytextDF, create_intermediate_result=False):

	if create_intermediate_result:
		mytextDF["normalized_http"] = mytextDF[active_column].apply(lambda tw: re.sub(http_re, 'urlurlurl', tw))
		mytextDF["normalized_usr"] = mytextDF["normalized_http"].apply(lambda tw: re.sub(usr_re, 'usrusrusr', tw))
		mytextDF[active_column] = mytextDF["normalized_usr"]
	else:
		mytextDF[active_column] = mytextDF[active_column].apply(lambda tw: re.sub(http_re, 'urlurlurl', tw))
		mytextDF[active_column] = mytextDF[active_column].apply(lambda tw: re.sub(usr_re, 'usrusrusr', tw))

	return mytextDF   

def get_and_eliminate_near_duplicate_tweets(tweetsDF, distancemetric='cosine', debug=False, similarity_threshold=0.25, debug_threshold=1000, defaultfreqcut_off=2):

	start_time = datetime.datetime.now()

	# active_tweet_df = tweetsDF  	

	logging.info("We use 'np.random.choice' method for creating a data frame 'active_tweet_df' that contains random tweets and its parameter is 'tweetsDF'.")
	logging.info("The used distance metric is:"+distancemetric)
	
	if debug:
		if len(tweetsDF) > debug_threshold:
			active_tweet_df = tweetsDF[:debug_threshold]    # We choose random data for testing. 
		else:
			active_tweet_df = tweetsDF
	else:
		active_tweet_df = tweetsDF
	

	#logging.info("mytext:" + str(active_tweet_df["text"]))
	logging.info("\n size of mytext:" + str(len(active_tweet_df[active_column])))

	if len(active_tweet_df) > 1000:
		freqcutoff = int(m.log(len(active_tweet_df)))
	else:
		freqcutoff = defaultfreqcut_off # default 2 is applicable for short texts tweets.

	logging.info("Tweet count is:"+str(len(active_tweet_df))+"\tfreqcutoff:"+str(freqcutoff))

	logging.info("In 'word_vectorizer' method, we use 'TfidfVectorizer' to get feature names and parameter is 'active_tweet_df[active_column]' .")
	word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, norm='l2', min_df=freqcutoff, token_pattern=my_token_pattern, sublinear_tf=True)
	X2_train = word_vectorizer.fit_transform(active_tweet_df[active_column])
	#X2_train = X2_train.toarray() # we do not need this, we should work with sparse matrices for the efficiency.
	logging.info("features:" + str(word_vectorizer.get_feature_names()))
	logging.info("number of features:" + str(len(word_vectorizer.get_feature_names())))
	logging.info("End of the 'word_vectorizer' method.")

	allowed_metrics = ['cosine',  'euclidean', 'cityblock', 'jaccard']
	if distancemetric not in allowed_metrics:
		raise Exception("distance metric should be one of the allowed ones. Allowed metrics are: " + str(allowed_metrics))

	# dist = distance.pdist(X2_train, distancemetric2)  # Distances are defined as a parameter in the function.
	# dist_matrix = scipy.spatial.distance.squareform(dist)   # Valid values for metric are 'Cosine', 'Cityblock', 'Euclidean' and 'Jaccard'.
	# logging.info("distances:"+str(dist_matrix))   # These metrics do not support sparse matrix inputs.

	logging.info("In scikit-learn pairwise distance, we use different parameters for different distances that are defined in 'allowed_metrics'.")
	dist_matrix = pairwise_distances(X2_train, metric=distancemetric, n_jobs=1)   # Valid values for metric are 'Cosine', 'Cityblock', 'Euclidean' and 'Manhattan'.
	logging.info("distances:" + str(dist_matrix))  # These metrics support sparse matrix inputs.

	similarity_dict = {}
	for a, b in np.column_stack(np.where(dist_matrix < similarity_threshold)):  # zip(np.where(overthreshold)[0],np.where(overthreshold)[1]):
		if a != b:
			if active_tweet_df.index[a] not in similarity_dict: # work with the actual index no in the dataframe, not with the order based one!
				similarity_dict[active_tweet_df.index[a]] = [active_tweet_df.index[a]]  # a is the first member of the group.
			similarity_dict[active_tweet_df.index[a]].append(active_tweet_df.index[b])


	if len(similarity_dict) == 0:
		print("There is not any group of near-duplicate tweets.")
		return active_tweet_df

	# unique_tweets = 
	# return unique_tweets =

	# logging.info("Demote the duplicate tweets to 1 'list(set([tuple(sorted(km))'.")
	# Delete the duplicate clusters.
	cluster_tuples_list = list(set([tuple(sorted(km)) for km in similarity_dict.values()]))  # for each element have a group copy in the group, decrease 1.
	cluster_tuples_list = sorted(cluster_tuples_list, key=len, reverse=True)

	cluster_tuples2 = [cluster_tuples_list[0]]
	
	if debug:
		logging.info("The length based sorted cluster tuples are:"+str(cluster_tuples_list)) # being in this dictionary means a and b are similar.
		logging.info("The length of the length based sorted cluster tuples are:"+str(len((cluster_tuples_list)))) # being in this dictionary means a and b are similar.


	duplicate_tweet_indexes = list(cluster_tuples_list[0])
	for ct in cluster_tuples_list[1:]:
		if len(set(duplicate_tweet_indexes) & set(ct)) == 0:
			cluster_tuples2.append(ct)
			duplicate_tweet_indexes += list(ct)
	#print("Number of cluster2:", len(cluster_tuples2))

	duplicate_tweet_indexes = list(set(duplicate_tweet_indexes))

	one_index_per_duplicate_group = []
	for clst in cluster_tuples2:
		one_index_per_duplicate_group.append(clst[0])

	#print("list len vs. set len (one_index_per_duplicate_group):", len(one_index_per_duplicate_group), len(set(one_index_per_duplicate_group)))
	#print("list len vs. set len (duplicate_tweet_indexes):", len(duplicate_tweet_indexes), len(set(duplicate_tweet_indexes)))

	indexes_of_the_uniques = [i for i in active_tweet_df.index if i not in duplicate_tweet_indexes]
	#print("list len vs. set len (indexes_of_the_uniques):", len(indexes_of_the_uniques), len(set(indexes_of_the_uniques)))


	#print("Any null in the main dataframe:",active_tweet_df[active_tweet_df.text.isnull()])
	unique_active_tweet_df = active_tweet_df.ix[[i for i in active_tweet_df.index if i not in duplicate_tweet_indexes]+one_index_per_duplicate_group] # + 
	#print("Any null in the unique dataframe:",unique_active_tweet_df[unique_active_tweet_df.text.isnull()])

	tweet_sets = []
	for i, ct in enumerate(cluster_tuples2):
		tweets = []
		for t_indx in ct:
			tweets.append(active_tweet_df[active_column].ix[t_indx])
		tweet_sets.append(tweets)
		logging.info("\nNear duplicate tweet groups are below: ")
		logging.info("\nsize of group " + str(i) + ':' + str(len(tweets)))

	if debug:
		logging.info("COMPLETE: Near duplicate tweet sets:" + "\n\n\n".join(["\n".join(twset) + "\n" + "\n".join(twset) for twset in tweet_sets]))
	else:
		logging.info("SUMMARY: Near duplicate tweet sets:" + "\n\n\n".join(["\n".join(twset[:1]) + "\n" + "\n".join(twset[-1:]) for twset in tweet_sets]))

	# logging.info("Near duplicate tweet sets:" + "\n\n\n".join(["\n".join(twset) for twset in tweet_sets]))
	logging.info('End of the "scikit-learn pairwise distance".')
	# logging.info('End of the 'scipy pairwise distance'.')

	

	logging.info("unique tweet sets:" + '\n' + str(unique_active_tweet_df[active_column][:3]) + str(unique_active_tweet_df[active_column][:3][-3:]))
	logging.info("\n size of unique tweets:" + str(len(unique_active_tweet_df)))

	logging.info(" 'datetime.datetime.now()' method is used for calculating the process time. ")
	end_time = datetime.datetime.now()
	logging.info(str('Duration: {}'.format(end_time - start_time)))  # It calculates the processing time.
	logging.info("end of the datetime.datetime.now() method.")

	number_eliminated = len(active_tweet_df) - len(unique_active_tweet_df)
	logging.info("number of eliminated text:"+str(number_eliminated))

	per = number_eliminated/len(active_tweet_df)  # It calculates the number of eliminated tweets as percentage.
	logging.info("percentage of eliminated tweet is " + str(per))

	logging.info("final DataFrame info:" + str(unique_active_tweet_df.info()))
	# logging.info("head:"+str(active_tweet_df.head()))

	return unique_active_tweet_df

def tok_results(tweetsDF, elimrt=False): 

	results = []
	if no_tok:  # create a function for that step!
		tok_result_lower_col = "texttok"

		twtknzr = Twtokenizer()
		tweetsDF = twtknzr.tokenize_df(tweetsDF, texcol=active_column, rescol=tok_result_col, addLowerTok=False)
		tweetsDF[tok_result_lower_col] = tweetsDF[tok_result_col].str.lower()
		print("\nAvailable attributes of the tokenized tweets:", tweetsDF.columns)
		print("\ntweet set summary:", tweetsDF.info())
		print(tweetsDF[tok_result_col][:5])
		print("Tweets ARE tokenized.")
	else:  # do not change the text col
		tok_result_lower_col = "texttok"

		tweetsDF[tok_result_lower_col] = tweetsDF[tok_result_col].str.lower()
		print("\nAvailable attributes of the tweets:", tweetsDF.columns)
		print("\ntweet set summary:", tweetsDF.info())
		print(tweetsDF[tok_result_col][:5])
		print("\ntweets are NOT tokenized.")
	if elimrt:
		rttext = ~tweetsDF[tok_result_lower_col].str.contains(r"\brt @")
		rtfield = tweetsDF["is_retweet"] == False
		tweetsDF["is_notrt"] = rtfield.values & rttext.values  # The default setting is to eliminate retweets
		tweetsDF = tweetsDF[tweetsDF.is_notrt]
		print("Retweets were eliminated.")
	else:
		print("Retweets were NOT eliminated.")

	tweetsDF[active_column] = tweetsDF[tok_result_lower_col].copy()
	return tweetsDF

def get_uni_bigrams(text, token_pattern=my_token_pattern):

	token_list = re.findall(token_pattern, text)
	return [" ".join((u, v)) for (u, v) in zip(token_list[:-1], token_list[1:])] + token_list

def reverse_index_frequency(cluster_bigram_cntr):

	reverse_index_freq_dict = {}
	for k, freq in cluster_bigram_cntr.most_common():
		if str(freq) not in reverse_index_freq_dict:
			reverse_index_freq_dict[str(freq)] = []
		reverse_index_freq_dict[str(freq)].append(k)

	return reverse_index_freq_dict

def get_annotated_tweets(collection_name): 

	"""
	Dataframe of:
		text    label 
		
		txt1     l1
		txt2     l2
	"""

	return None

def get_vectorizer_and_mnb_classifier(tweets_as_text_label_df, my_token_pattern, pickle_file=None): 

	print('In get_mnb_classifier:')
	Counter()

	freqcutoff = int(m.log(len(tweets_as_text_label_df))/2)

	now = datetime.datetime.now()
	logging.info("feature_extraction_started_at: " + str(now))

	word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, norm='l2', min_df=freqcutoff, token_pattern=my_token_pattern, sublinear_tf=True)
	X2_train = word_vectorizer.fit_transform(tweets_as_text_label_df.text.values)

	logging.info("Number of features:"+str(len(word_vectorizer.get_feature_names())))
	logging.info("Features are:"+str(word_vectorizer.get_feature_names()))
	# logging("n_samples: %d, n_features: %d" % X2_train.shape)

	now2 = datetime.datetime.now()
	logging.info("feature_extraction_ended_at: " + str(now2))

	now3 = datetime.datetime.now()
	logging.info("Training started at: " + str(now3))

	y_train = tweets_as_text_label_df.label.values
	MNB = MultinomialNB(alpha=.1)
	MNB.fit(X2_train, y_train)

	now4 = datetime.datetime.now()
	logging.info("Training ended at: " + str(now4))

	vect_and_classifier = {'vectorizer': word_vectorizer, 'classifier': MNB}

	if (pickle_file is not None) and isinstance(pickle_file, str):
		if not pickle_file.endswith(".pickle"):
			pickle_file += '.pickle'	
		with open(pickle_file, 'wb') as f:
			pickle.dump(vect_and_classifier, f, pickle.HIGHEST_PROTOCOL)
			print("Pickle file was written to", pickle_file)
	else:
		print("The pickle file is not a string. It was not written to a pickle file.")

	return vect_and_classifier

def create_clusters(tweetsDF,  my_token_pattern, min_dist_thres=0.6, min_max_diff_thres=0.4, max_dist_thres=0.8, iteration_no=1, min_clusters=1, printsize=True, nameprefix='',  selection=True, strout=False, user_identifier='screen_name', cluster_list=None): 
	"""
	Have modes:
	mode1: get a certain number of clusters. Relax parameters for it. (This is the current Mode!)
	mode2: get clusters that comply with certain conditions.

	"min_max_diff_thres" should not be too small. Then You miss thresholds like: min 0.3 - min 0.7: The top is controlled by the maximum anyway. Do not fear from having it big: around 0.4

	"""

	if min_dist_thres > 0.85 and max_dist_thres>0.99:
		logging.info("The parameter values are too high to allow a good selection. We just finish searching for clusters at that stage.")
		logging.info("Threshold Parameters are: \nmin_dist_thres="+str(min_dist_thres)+"\tmin_max_diff_thres:="+str(min_max_diff_thres)+ "\tmax_dist_thres="+str(max_dist_thres))
		return cluster_list


	len_clust_list = 0
	if cluster_list is None:
		cluster_list = []

	elif not selection and len(cluster_list)>0:
		return cluster_list
	else:
		len_clust_list = len(cluster_list)
		logging.info("Starting the iteration with:"+str(len_clust_list)+" clusters.")

		clustered_tweet_ids = []

		for clust_dict in cluster_list:
			clustered_tweet_ids += clust_dict["twids"]

		logging.info("Number of already clustered tweets are:"+str(len(clustered_tweet_ids)))

		logging.info("Tweet set size to be clustered:"+str(len(tweetsDF)))
		tweetsDF = tweetsDF[~tweetsDF.id_str.isin(clustered_tweet_ids)]
		logging.info("Tweet set size to be clustered(after elimination of the already clustered tweets):"+str(len(tweetsDF)))

		if len(tweetsDF)==0:
			logging.info("Please check that the id_str has a unique value for each item.")
			print("Please check that the id_str has a unique value for each item.")
			return cluster_list

	logging.info('Creating clusters was started!!')
	logging.info("Threshold Parameters are: \nmin_dist_thres="+str(min_dist_thres)+"\tmin_max_diff_thres:="+str(min_max_diff_thres)+ "\tmax_dist_thres="+str(max_dist_thres))
	cluster_bigram_cntr = Counter()

	freqcutoff = int(m.log(len(tweetsDF))/2)
	if freqcutoff == 0:
		freqcutoff = 1 # make it at least 1.

	#freqcutoff = int(m.log(len(tweetsDF))/2) # the bigger freq threshold is the quicker to find similar groups of tweets, although precision will decrease.
	logging.info("Feature extraction parameters are:\tfrequencyCutoff:"+str(freqcutoff))

	word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, norm='l2', min_df=freqcutoff, token_pattern=my_token_pattern)
	text_vectors = word_vectorizer.fit_transform(tweetsDF[active_column])
	# logging.info("Number of features:"+str(len(word_vectorizer.get_feature_names())))
	logging.info("Features are:"+str(word_vectorizer.get_feature_names()))

	n_clust = int(m.sqrt(len(tweetsDF))/2)+iteration_no*(min_clusters-len_clust_list) # The more clusters we need, the more clusters we will create.
	n_initt = int(m.log10(len(tweetsDF)))+iteration_no  # up to 1 million, in KMeans setting, having many iterations is not a problem. # more iterations higher chance of having candidate clusters.

	logging.info("Clustering parameters are:\nnclusters="+str(n_clust)+"\tn_initt="+str(n_initt))
	

	if len(tweetsDF) < 1000000:
		km = KMeans(n_clusters=n_clust, init='k-means++', max_iter=500, n_init=n_initt)  # , n_jobs=16
		logging.info("The data set is small enough to use Kmeans")
	else: 
		km = MiniBatchKMeans(n_clusters=n_clust, init='k-means++', max_iter=500, n_init=n_initt, batch_size=1000)
		logging.info("The data set is BIG, MiniBatchKMeans is used.")

	km.fit(text_vectors)
	
	# Cluster = namedtuple('Cluster', ['cno', 'cstr','tw_ids'])
	clustersizes = get_cluster_sizes(km, tweetsDF[active_column].values)

	logging.info("Cluster sizes are:"+str(clustersizes))

	for cn, csize in clustersizes.most_common():  # range(args.ksize):
		cn = int(cn)
		similar_indices = (km.labels_ == cn).nonzero()[0]
		similar = []
		similar_tuple_list = []
		for i in similar_indices: 
			dist = sp.linalg.norm((km.cluster_centers_[cn] - text_vectors[i]))
			similar_tuple_list.append((dist, tweetsDF['id_str'].values[i], tweetsDF[active_column].values[i], tweetsDF[user_identifier].values[i])) 
			if strout:
				similar.append(str(dist) + "\t" + tweetsDF['id_str'].values[i] + "\t" + tweetsDF[active_column].values[i] + "\t" + tweetsDF[user_identifier].values[i])
		
		similar_tuple_list = sorted(similar_tuple_list, key=itemgetter(0)) # sort based on the 0th, which is the distance from the center, element.
		# test sortedness!

		if strout:	
			similar = sorted(similar, reverse=False)
		cluster_info_str = ''
		user_list = [t[3] for t in similar_tuple_list]  # t[3] means the third element in the similar_tuple_list.
		if selection:
			if (len(similar_tuple_list)>2) and (similar_tuple_list[0][0] < min_dist_thres) and (similar_tuple_list[-1][0] < max_dist_thres) and ((similar_tuple_list[0][0] + min_max_diff_thres) > similar_tuple_list[-1][0]):  # the smallest and biggest distance to the centroid should not be very different, we allow 0.4 for now!
				cluster_info_str += "cluster number and size are: " + str(cn) + '    ' + str(clustersizes[str(cn)]) + "\n"
				for txt in tweetsDF[active_column].values[similar_indices]:
					cluster_bigram_cntr.update(get_uni_bigrams(txt))  # regex.findall(r"\b\w+[-]?\w+\s\w+", txt, overlapped=True))
					# cluster_bigram_cntr.update(txt.split()) # unigrams
				frequency = reverse_index_frequency(cluster_bigram_cntr)
				if strout:
					topterms = [k+":" + str(v) for k, v in cluster_bigram_cntr.most_common() if k in word_vectorizer.get_feature_names()]  
					cluster_info_str += "Top terms are:" + ", ".join(topterms) + "\n"
				if strout:
					cluster_info_str += "distance_to_centroid" + "\t" + "tweet_id" + "\t" + "tweet_text\n"
					if len(similar) > 20:
						cluster_info_str += 'First 10 documents:\n'
						cluster_info_str += "\n".join(similar[:10]) + "\n"
						# print(*similar[:10], sep='\n', end='\n')

						cluster_info_str += 'Last 10 documents:\n'
						cluster_info_str += "\n".join(similar[-10:]) + "\n"
					else:
						cluster_info_str += "Tweets for this cluster are:\n"
						cluster_info_str += "\n".join(similar) + "\n"
			else:
				logging.info("Cluster is not good. Smallest and largest distance to the cluster center are:"+str(similar_tuple_list[0][0])+"\t"+str(similar_tuple_list[-1][0]))
		else:
				cluster_info_str += "cluster number and size are: " + str(cn) + '    ' + str(clustersizes[str(cn)]) + "\n"
				cluster_bigram_cntr = Counter()
				for txt in tweetsDF[active_column].values[similar_indices]:
					cluster_bigram_cntr.update(get_uni_bigrams(txt))
				frequency = reverse_index_frequency(cluster_bigram_cntr)
				if strout:
					topterms = [k+":"+str(v) for k, v in cluster_bigram_cntr.most_common() if k in word_vectorizer.get_feature_names()]  
					cluster_info_str += "Top terms are:" + ", ".join(topterms) + "\n"
				if strout:
					cluster_info_str += "distance_to_centroid" + "\t" + "tweet_id" + "\t" + "tweet_text\n"
					if len(similar) > 20:
						cluster_info_str += 'First 10 documents:\n'
						cluster_info_str += "\n".join(similar[:10]) + "\n"
						# print(*similar[:10], sep='\n', end='\n')

						cluster_info_str += 'Last 10 documents:\n'
						cluster_info_str += "\n".join(similar[-10:]) + "\n"
					else:
						cluster_info_str += "Tweets for this cluster are:\n"
						cluster_info_str += "\n".join(similar) + "\n"

		if len(cluster_info_str) > 0:  # that means there is some information in the cluster.
			logging.info("\nCluster was appended. cluster_info_str:"+cluster_info_str+"\tmin_dist="+str(similar_tuple_list[0][0])+"\tmax_dist="+str(similar_tuple_list[-1][0]))
			cluster_list.append({'cno': cn, 'cnoprefix': nameprefix+str(cn), 'user_entropy': entropy(user_list), 'rif': frequency, 'cstr': cluster_info_str, 'ctweettuplelist': similar_tuple_list,  'twids': list(tweetsDF[np.in1d(km.labels_, [cn])]["id_str"].values)})  # 'user_ent':entropy(user_list),

	logging.info("length of cluster_list:"+str(len(cluster_list)))
	len_clust_list = len(cluster_list) # use to adjust the threshold steps for the next iteration. If you are closer to the target step smaller.
	if len_clust_list<min_clusters:
		logging.info("There is not enough clusters, call the create_clusters again with relaxed threshold parameters (recursively). Iteration no:"+str(iteration_no))

		factor = (min_clusters-len_clust_list)/1000 # if it needs more clusters, it will make a big step
		
		min_dist_thres2, max_dist_thres2, min_max_diff_thres2 = relax_parameters(min_dist_thres, max_dist_thres, min_max_diff_thres, factor)
		logging.info("Threshold step sizes are: \nmin_dist_thres="+str(min_dist_thres-min_dist_thres2)+"\tmax_dist_thres="+str(max_dist_thres-max_dist_thres2)+"\tmin_max_diff_thres="+str(min_max_diff_thres-min_max_diff_thres2))
		return create_clusters(tweetsDF,  my_token_pattern, min_dist_thres=min_dist_thres2, min_max_diff_thres=min_max_diff_thres2, max_dist_thres=max_dist_thres2,  \
			iteration_no=iteration_no+1, min_clusters=min_clusters, user_identifier=user_identifier, cluster_list=cluster_list)

	return cluster_list

def relax_parameters(min_dist_thres,max_dist_thres,min_max_diff_thres,factor):
	min_dist_thres = min_dist_thres + min_dist_thres*(factor/2)  # As you are closer to the center distance change fast (Euclidean distance). Step a bit big.
	max_dist_thres = max_dist_thres + max_dist_thres*(factor/3) # If you are far from the center, there will be much variation in small differences of distance (Euclidean). Step small.
	min_max_diff_thres = min_max_diff_thres + min_max_diff_thres*(factor/4) # There is not much sense in increasing it much. Otherwise it will loose its meaning easily. 

	return min_dist_thres, max_dist_thres, min_max_diff_thres

def eliminate_duplicates_bucketwise(df, step=10000):
    """
    The actual near-duplicate detection algorithm is not memory-efficient enough. Therefore,
    we mostly need to divide the data in the buckets, eliminate duplicates, merge the data, shuffle it, and repeat
    the same cycle, until no-duplicate detected in any bucket. That may take long for big data sets. Conditions can
    be relaxed to be quicker but leave a few duplicates.
    """
            
    logging.info("starting eliminate_duplicates_bucketwise, df length:"+str(len(df)))
    df = df.reindex(np.random.permutation(df.index))
    df.reset_index(inplace=True, drop=True)

    tmp_df2 = pd.DataFrame()
    for i in range(0, len(df), step):
        tmp_unique = get_and_eliminate_near_duplicate_tweets(df[i:i+step], similarity_threshold=0.10, debug=True, debug_threshold=10000)
        tmp_df2 = pd.concat([tmp_df2, tmp_unique], ignore_index=True)

    if len(df) > len(tmp_df2):
        logging.info(str(len(df) - len(tmp_df2))+" tweets were eliminated!")
        return eliminate_duplicates_bucketwise(tmp_df2)

    return df

if __name__ == "__main__":
	import output
	# parser = argparse.ArgumentParser(description='Detect information groups in a microtext collection')
	# parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
	# parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
	# Find a way to print the help with the default parameters when the command is: python relevancer --help
	# parser.add_argument('-f', '--infile', type=str, help='input file should contain the microtexts') # format of the file should be defined later.
	# parser.add_argument('-l', '--lang', type=str, default='en', action='store', help='language of the microtext that will be selected to be processed further.')
	# parser.add_argument('-t', '--tok', type=str, default=False, action='store', help='Should the tweets be tokenized? Default: False')
	# parser.add_argument('-d', '--mongodb', type=str, default='myconfig.ini', action='store', help='provide MongoDB credentials')
	# parser.add_argument('-g', '--logfile', type=str, default='myapp.log', action='store', help='provide log file name')
	# args = parser.parse_args()
	# if args.infile is not None:
	# 	logging.info("The tweet file is:"+args.infile)  # This give None in case there is not a file provided by -f parameter. Be aware! It is not a problem now.
	# else:
	# 	logging.info("There is not any tweet text file. The default MongoDB configuration file is being read!")
	# logging.info("The language to be processed is:" + args.lang)

	tweetlist = read_json_tweets_database(args.lang)
	# tweetlist = read_json_tweets_file(args.lang)  #You need to give a text file.
	logging.info("number of tweets", len(tweetlist))

	tweetsDF = create_dataframe(tweetlist)
	tok = tok_results(tweetsDF)
	tw_id = get_tw_id(rlvcl)

	start_tweet_size = len(tweetsDF)
	print("Number of the tweets after retweet elimination:", start_tweet_size)
	print("Choose mode of the annotation.")
	print("1. relevant vs. irrelevant (default)")

	information_groups = {"relevant": [], "irrelevant": []}  # contains IDs of tweets, updated after each clustering/labeling cycle
	print("2. provide a list of labels")
	print("3. define labels during exploration")
	mchoice = input("Your choice:")
	if mchoice == '2':
		mylabels = input("Enter a comma seperated label list:").split(",")
		information_groups = {k: [] for k in mylabels}
		print("Labels are:", [l for l in sorted(list(information_groups.keys()))])

	identified_tweet_ids = []
	while True:
		km, doc_feat_mtrx, word_vectorizer = output.create_clusters(tweetsDF[tok_result_col])
		print("\nThe silhouette score (between 0 and 1, the higher is the better):", metrics.silhouette_score(doc_feat_mtrx, km.labels_, metric='euclidean', sample_size=5000))
		clustersizes = get_cluster_sizes(km, tweetsDF[tok_result_col].values)
		print("\nCluster sizes:", clustersizes.most_common())
		print("cluster candidates:", end=' ')
		# local_information_groups = {'noise':[]} # contains cluster number for this iteration. It should pass the tweet ID information to the global_information_groups at end of each cycle
		output_cluster_list = output.get_candidate_clusters(km, doc_feat_mtrx, word_vectorizer, tweetsDF, tok_result_col, min_dist_thres, max_dist_thres)
		if len(output_cluster_list) == 0:
			print("\nThere is not any good group candidate in the last clustering. New clustering with the relaxed selection conditions.")
			min_dist_thres += 0.01
			max_dist_thres += 0.01
			print("Relaxed distance thresholds for the group selection are:", min_dist_thres, max_dist_thres)
			time.sleep(5)
			continue
		for cn, c_str, tw_ids in output_cluster_list:
			# for cn, csize in clustersizes.most_common():
			# 	cn = int(cn)
			print('\n'+c_str)
			# print("current ids:", tw_ids) # remove this after testing
			available_labels_no = [str(i)+"-"+l for i, l in enumerate(sorted(list(information_groups.keys())))]
			available_labels = [l for l in sorted(list(information_groups.keys()))]  # do not move it to somewhere else, because in explorative mode, label updates should be reflected.
			print("\n Available labels:", available_labels_no)

			cluster_label = input("Enter (number of) an available or a new label (q to quit the iteration, Enter to skip making a decision for this group):")
			if cluster_label == 'q':
				break
			elif cluster_label == '':
				print("This group of tweets were skipped for this iteration!")
				continue
			elif cluster_label.isdigit() and int(cluster_label) < len(information_groups):
				information_groups[available_labels[int(cluster_label)]] += tw_ids  # list(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
				print("Group assigned to the available label:", available_labels[int(cluster_label)])
				print("Cluster number:", cn, "its label:", available_labels[int(cluster_label)])
			elif mchoice == '3':
				if cluster_label not in information_groups:  # in order to avoid overwriting content of a label
					information_groups[cluster_label] = []
				information_groups[cluster_label] += tw_ids  # list(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
				print("Group assigned to a NEW label:", cluster_label)
				print("Cluster number:", cn, "its label:", cluster_label)
			# print(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
			else:
				print("\nEnter a label or label number that is available. If you want to add new labels as you are exploring you should activate the explorative mode:3 from the previous step. \n Annotation of this group will be skipped.")
			print("Next cluster is coming ...")
			time.sleep(1)  # wait while showing result of the assignment
		print('\nEnd of Iteration, available groups:')
		print(*information_groups, sep='\n', end='\n\n')

		for k, v in information_groups.items():
			identified_tweet_ids += v
		identified_tweet_ids = list(set(identified_tweet_ids))  # dublicates is added in successive iterations.
		print('number of classified tweets:', len(identified_tweet_ids))

		tweetsDF = tweetsDF[~tweetsDF.id_str.isin(identified_tweet_ids)]
		print('number of remaining tweets to be identified:', len(tweetsDF))
		print('current labeled ratio is:', len(identified_tweet_ids)/len(tweetsDF))
		print('target labeling ratio is:', target_labeling_ratio)
		print("current distance thresholds are:", min_dist_thres, max_dist_thres)
		time.sleep(5)
		
		if len(identified_tweet_ids)/len(tweetsDF) < target_labeling_ratio:
			print('\nNew clustering will be done to achieve the target.')
			continue  
		else:  # else ask the user.
			iter_choice = input("\nTarget was achieved. Press y if you want to do one more iteration:")
			if iter_choice == 'y':  # if the user enter y, the infinite loop should be continued!
				continue
		# This will run after we reach the target or decide not to continue labeling any more after we actively decide to label mor eafter we reach the target.
		label_rest_tweets = input("Press y if you want to label the remaining tweets without any selection criteria:")
		if label_rest_tweets == 'y':
			km, doc_feat_mtrx, word_vectorizer = output.create_clusters(tweetsDF[tok_result_col])			
			for cn, c_str, tw_ids in output.get_candidate_clusters(km, doc_feat_mtrx, word_vectorizer, tweetsDF, tok_result_col, min_dist_thres, max_dist_thres, selection=False):
				# for cn, csize in clustersizes.most_common():
				# 	cn = int(cn)
				print('\n'+c_str)
				# print("current ids:", tw_ids) # remove this after testing
				available_labels_no = [str(i)+"-"+l for i, l in enumerate(sorted(list(information_groups.keys())))]
				available_labels = [l for l in sorted(list(information_groups.keys()))]  # do not move it to somewhere else, because in explorative mode, label updates should be reflected.
				print("\n Available labels:", available_labels_no)

				cluster_label = input("Enter (number of) an available or a new label (q to quit the iteration, Enter to skip making a decision):")
				if cluster_label == 'q':
					break
				elif cluster_label == '':
					print("This group of tweets were skipped for this iteration!")
					continue
				elif cluster_label.isdigit() and int(cluster_label) < len(information_groups):
					information_groups[available_labels[int(cluster_label)]] += tw_ids  # list(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
					print("Group assigned to the available label:", available_labels[int(cluster_label)])
					print("Cluster number:", cn, "its label:", available_labels[int(cluster_label)])
				elif mchoice == '3':
					if cluster_label not in information_groups:  # in order to avoid overwriting content of a label
						information_groups[cluster_label] = []
					information_groups[cluster_label] += tw_ids  # list(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
					print("Group assigned to a NEW label:", cluster_label)
					print("Cluster number:", cn, "its label:", cluster_label)
				# print(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
				else:
					print("\nEnter a label or label number that is available. If you want to add new labels as you are exploring you should activate the explorative mode:3 from the previous step. \n Annotation of this group will be skipped.")
				print("Next cluster is coming ...")
				time.sleep(1)  # wait while showing result of the assignment
		break  # go out of the infinite while loop		
	print("Ask if they want to write the groups to a file, which features are needed, which file formats: json, tsv?")
	for k, v in information_groups.items():
		group_tweets = [tw for tw in tweetlist if tw["id_str"] in v]
		print("length of the tweets and ids (they must be equal!)", len(group_tweets), len(v))  # they should be equal!
	rlvdb[result_collection].drop()  # be sure to overwrite it! Instead of overwriting, it can be inserted by adding a date and time to this dictionary.
	rlvdb[result_collection].insert(information_groups)
	print("The result was written to the collection:", result_collection)		
