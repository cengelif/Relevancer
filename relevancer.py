import configparser
import sys
import pymongo as pm
import logging
import argparse
import json
import datetime
import time
import math as m
from collections import Counter, OrderedDict, namedtuple
from html.parser import HTMLParser

import pandas as pd
import numpy as np
import scipy as sp
import re

from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics


parser = argparse.ArgumentParser(description='Detect information groups in a microtext collection')

#parser.add_argument('integers', metavar='N', type=int, nargs='+',
#               help='an integer for the accumulator')
#parser.add_argument('--sum', dest='accumulate', action='store_const',
#               const=sum, default=max,
#               help='sum the integers (default: find the max)')


# Find a way to print the help with the default parameters when the command is: python relevancer --help
parser.add_argument('-f', '--infile', type=str, help='input file should contain the microtexts') # format of the file should be defined later.
parser.add_argument('-l', '--lang', type=str, default='en', action='store', help='language of the microtext that will be selected to be processed further.')
parser.add_argument('-t', '--tok', type=str, default=False, action='store', help='Should the tweets be tokenized? Default: False')
parser.add_argument('-d', '--mongodb', type=str, default='myconfig.ini', action='store', help='provide MongoDB credentials')
parser.add_argument('-g', '--logfile', type=str, default='myapp.log', action='store', help='provide log file name')

args = parser.parse_args()
#print (args)


#Logging
logging.basicConfig(filename=args.logfile,
                            #filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%d-%m-%Y, %H:%M',
                            level=logging.INFO)

logging.info("Script started")

def connect_mongodb(configfile="data/mongodb.ini", coll_name=None):
#Config Parser
   config = configparser.ConfigParser()
   config.read(configfile)

#MongoLab OAuth;
   client_host = config.get('mongodb', 'client_host')
   client_port = int(config.get('mongodb', 'client_port'))
   db_name = config.get('mongodb', 'db_name')
   if (coll_name == None):
      coll_name = config.get('mongodb', 'coll_name')
   if config.has_option('mongodb', 'user_name'):
      user_name = config.get('mongodb', 'user_name')
   if config.has_option('mongodb', 'passwd'):
      passwd = config.get('mongodb', 'passwd')
     
   # print(client_host, client_port,db_name,coll_name,user_name,passwd)

#Mongo query
   #mongo_query = {} # we may read this from a json file.

#Connect to database
   try:
      connection = pm.MongoClient(client_host, client_port)
      rlvdb = connection[db_name]  #Database
      if ('user_name' in locals()) and ('passwd' in locals()):
         rlvdb.authenticate(user_name, passwd)
      rlvcl = rlvdb[coll_name] #Collection
   except Exception:
      sys.exit("Database connection failed!")
      pass

   return rlvdb, rlvcl
   

min_dist_thres = 0.65 # the smallest distance of a tweet to the cluster centroid should not be bigger than that.
max_dist_thres = 0.85 # the biggest distance of a tweet to the cluster centroid should not be bigger than that.
target_labeling_ratio = 0.5 # percentage of the tweets that should be labeled, until this ratio achieved iteration will repeat automatically.
result_collection = "relevancer_result"

if args.tok: # create a function for that step!
	tok_result_col = "texttokCap"
else:
	tok_result_col = "text"
	
if args.infile is not None:
	logging.info("The tweet file is:"+args.infile) # This give None in case there is not a file provided by -f parameter. Be aware! It is not a problem now.
else:
	logging.info("There is not any tweet text file. The default MongoDB configuration file is being read!")

logging.info("The language to be processed is:"+args.lang)

class MLStripper(HTMLParser):
	def __init__(self):
		self.reset()
		self.strict = False
		self.convert_charrefs= True
		self.fed = []
	def handle_data(self, d):
		self.fed.append(d)
	def get_data(self):
		return ''.join(self.fed)

class Twtokenizer():
	
	def __init__(self):
		self.abbreviations = ['i.v.m.','a.s.','knp.']
		print('init:',self.abbreviations)

	def tokenize(self, tw):
		#abbcheck = False
		newtw = ''
		lentw = len(tw)
		#print(tw)
		for i, c in enumerate(tw):
			if (c in "'`´’‘") and ((i+1 != lentw) and (i!=0)) and ((tw[i-1].isalpha()) or tw[i-1] in "0123456789") and (tw[i+1].isalpha()):
				newtw += ' '+c+' '
			elif (c in "'`´’()+*-") and ((lentw>i+1) and (i>0)) and (tw[i-1] in ' 0123456789') and (tw[i+1].isalpha()):
				newtw += c+' '
			elif (c in '();>:') and ((lentw>i+1) and (i!=0)) and ((tw[i-1].isalpha()) or (tw[i-1] in '0123456789')) and ((tw[i+1] == ' ') or (i == lentw-1)):
				newtw += ' '+c
			elif (c in '.') and ((lentw>i+1) and (i!=0)) and ((tw[i-1].isalpha()) or (tw[i-1] in '0123456789')) and ((tw[i+1] == ' ') or (i == lentw-1)) \
							and (newtw.split()[-1]+c not in self.abbreviations):
				abbcheck = True
				newtw += " "+c
			elif (c in "'`´’‘()+*->") and (i==0) and (lentw > 1) and ((tw[1].isalpha()) or tw[1] in "0123456789"):
				newtw += c+' '
			elif (c in "'`´’‘()+*->") and (i+1 == lentw) and (lentw > 1) and ((tw[i-1].isalpha()) or tw[i-1] in "0123456789"):
				newtw += ' '+c
			elif (c in ",") and ((i != 0) and (i+1 != lentw)) and tw[i-1].isdigit() and tw[i+1].isdigit(): # for 3,5 7,5
				newtw += c
			elif (c in ",&"):
				newtw += " "+c+" "
			elif (c in "â"): # create a dictionary for character mappings. if c in dict: newtw += dict[c]
				newtw += "a"
			elif (c in "ê"):
				newtw += "e"
			elif (c in "î"):
				newtw += "i"
			elif (c in "ú"):
				newtw += "ü"
			#elif (c in ":")
			else:
				newtw += c
			#print(c in "'`´’()+*-", lentw>i+1, i>0, tw[i-1] == ' 0123456789', tw[i+1].isalpha())
			#if abbcheck:
			#	print("abbcheck is true:",newtw.split())
			#print(i,c,(c in '.'), ((lentw>i+1) and (i!=0)), ((tw[i-1].isalpha()) or (tw[i-1] in '0123456789')), ((tw[i+1] == ' ') or (i == lentw-1)) \
			#				and (newtw.split()[-1]+c not in self.abbreviations))
		#print('\n\n')
		return newtw

	def tokenize_df(self, tokdf, texcol="tweet", newtexcol='texttokCap',rescol="ttextlist", addLowerTok=True):
		#concert_table.drop_duplicates()
		# Note
		# tokdf[newtexcol] = tokdf[newtexcol].str.replace("""\xa0"""," ")
		# tokdf[newtexcol] = tokdf[newtexcol].str.replace("\n"," . ")
		
		tokdf[newtexcol] = tokdf[texcol].copy()
	
		#tokdf[newtexcol] = tokdf[newtexcol].replace(self.toReplaceDict, regex=True)
		tokdf[newtexcol][tokdf[newtexcol].str.endswith(".")] = tokdf[tokdf[newtexcol].str.endswith(".")][newtexcol].apply(lambda tw: tw[:-1]+' .') 
		tokdf[newtexcol][tokdf[newtexcol].str.endswith(".'")] = tokdf[tokdf[newtexcol].str.endswith(".'")][newtexcol].apply(lambda tw: tw[:-2]+" . '") 
		tokdf[newtexcol][tokdf[newtexcol].str.startswith("'")] = tokdf[tokdf[newtexcol].str.startswith("'")][newtexcol].apply(lambda tw: "' "+tw[1:])
	
		tokdf[newtexcol] = tokdf[newtexcol].apply(self.tokenize)
		tokdf[newtexcol] = tokdf[newtexcol].str.strip()
		#tokdf[rescol] = tokdf[newtexcol].str.split()
		
		if addLowerTok:
			tokdf[newtexcol[:-3]] = tokdf[newtexcol].str.lower()
	
		return tokdf.copy()

def strip_tags(html):
	s = MLStripper()
	s.feed(html)
	return s.get_data()

def read_json_tweets_file(myjsontweetfile, reqlang='en'):
	ftwits = []
	lang_cntr = Counter()

	
	with open(myjsontweetfile) as jfile:
		for i, ln in enumerate(jfile):

			if i == 10000: # restrict line numbers for test
				break
			
			t = json.loads(ln)
			lang_cntr[t["lang"]] += 1

			if t["lang"] == reqlang:
				t["created_at"] = datetime.datetime.strptime(t["created_at"],"%a %b %d %H:%M:%S +0000 %Y")

				#if t["created_at"].strftime("%Y-%m-%d") in flood_AnomBreakoutDaysList:

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
				#print(i, end=',')
				ftwits.append(t2)#.splitlines()
		print("Number of documents per languge:",lang_cntr)

		return ftwits


def read_json_tweets_database(rlvcl, mongo_query, tweet_count=-1, reqlang='en'):
	ftwits = []
	lang_cntr = Counter()

		
	for i, t in enumerate(rlvcl.find(mongo_query)):
		
		if i == tweet_count: # restrict line numbers for test
			break
				
      # t = json.loads(ln)
		lang_cntr[t["lang"]] += 1

		if t["lang"] == reqlang:
			t["created_at"] = datetime.datetime.strptime(t["created_at"],"%a %b %d %H:%M:%S +0000 %Y")

			#if t["created_at"].strftime("%Y-%m-%d") in flood_AnomBreakoutDaysList:

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
			#print(i, end=',')
			ftwits.append(t2)#.splitlines()
	print("Number of documents per languge:",lang_cntr)

	return ftwits

def get_cluster_sizes(kmeans_result, doclist):
	clust_len_cntr = Counter()
	for l in set(kmeans_result.labels_):
		clust_len_cntr[str(l)] = len(doclist[np.where(kmeans_result.labels_ == l)])
	return clust_len_cntr
	
def create_dataframe(tweetlist):

	dataframe = pd.DataFrame(tweetlist)
	data_cntr = Counter()
	
	logging.info("columns:"+str(dataframe.columns))
	
	dataframe.set_index("created_at", inplace=True)
	dataframe.sort_index(inplace=True)

	logging.info("Number of the tweets:"+str(len(dataframe)))
	logging.info("Available attributes of the tweets:"+str(dataframe.columns))
	
	return dataframe
	
def tok_results(tweetsDF):
	results = []
	tok_cntr = Counter()
	
	if args.tok: # create a function for that step!

		tok_result_lower_col = "texttok"

		twtknzr = Twtokenizer()
		tweetsDF = twtknzr.tokenize_df(tweetsDF, texcol='text', rescol=tok_result_col, addLowerTok=False)
		tweetsDF[tok_result_lower_col] = tweetsDF[tok_result_col].str.lower()
		print("\nAvailable attributes of the tokenized tweets:", tweetsDF.columns)
		print("\ntweet set summary:", tweetsDF.info())
		print(tweetsDF[tok_result_col][:5])

		rttext = ~tweetsDF[tok_result_lower_col].str.contains(r"\brt @")
		rtfield = tweetsDF["is_retweet"]==False
		tweetsDF["is_notrt"] = rtfield.values & rttext.values # The default setting is to eliminate retweets
		tweetsDF = tweetsDF[tweetsDF.is_notrt]

		print("Tweets are tokenized.")
	else: # do not change the text col
		
		
		tok_result_lower_col = "texttok"
		tweetsDF[tok_result_lower_col] = tweetsDF[tok_result_col].str.lower()

		print("\nAvailable attributes of the tweets:",tweetsDF.columns)
		print("\ntweet set summary:", tweetsDF.info())
		print(tweetsDF[tok_result_col][:5])

		rttext = ~tweetsDF[tok_result_lower_col].str.contains(r"\brt @")
		rtfield = tweetsDF["is_retweet"]==False
		tweetsDF["is_notrt"] = rtfield.values & rttext.values # The default setting is to eliminate retweets
		tweetsDF = tweetsDF[tweetsDF.is_notrt]

		print("\ntweets are NOT tokenized.")
		
	return results
	
def get_uni_bigrams(text, token_pattern=r"[#@]?\w+\b|[\U00010000-\U0010ffff]"):
	
	token_list = re.findall(token_pattern, text)
	
	return [" ".join((u,v)) for (u,v) in zip(token_list[:-1], token_list[1:])] + token_list
	
def reverse_index_frequency(cluster_bigram_cntr):
	reverse_index_freq_dict = {}
	
	for k, freq in cluster_bigram_cntr.most_common():
		
		if str(freq) not in reverse_index_freq_dict:
			reverse_index_freq_dict[str(freq)] = []
			
		reverse_index_freq_dict[str(freq)].append(k)

	return reverse_index_freq_dict

def create_clusters(tweetsDF, tok_result_col="text", min_dist_thres=0.6, max_dist_thres=0.8, printsize=True, nameprefix='', selection=True):
	cluster_bigram_cntr = Counter()
	
	freqcutoff = int(m.log(len(tweetsDF))/2)
	
	my_token_pattern=r"[#@]?\w+\b|[\U00010000-\U0010ffff]"
	
	word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, norm='l2', min_df=freqcutoff, token_pattern = my_token_pattern)
	text_vectors = word_vectorizer.fit_transform(tweetsDF[tok_result_col])
	
	doc_feat_mtrx = text_vectors
		
	n_clust = int(m.sqrt(len(tweetsDF)))
	n_initt = int(m.log10(len(tweetsDF)))
	
	km = KMeans(n_clusters=n_clust, init='k-means++', max_iter=1000, n_init=n_initt) # , n_jobs=16
	km.fit(doc_feat_mtrx)

	cluster_str_list = []
	#Cluster = namedtuple('Cluster', ['cno', 'cstr','tw_ids'])

	clustersizes = get_cluster_sizes(km, tweetsDF[tok_result_col].values)
	
	
	for cn, csize in clustersizes.most_common():#range(args.ksize):#clustersizes.most_common():
		cn = int(cn)
					
		similar_indices = (km.labels_== cn).nonzero()[0]
			
		similar = []
		similar_tuple_list = []
		for i in similar_indices:
			dist = sp.linalg.norm((km.cluster_centers_[cn] - doc_feat_mtrx[i]))
			similar.append(str(dist) + "\t" + tweetsDF['id_str'].values[i]+"\t"+ tweetsDF[tok_result_col].values[i])
			similar_tuple_list.append((dist, tweetsDF['id_str'].values[i], tweetsDF[tok_result_col].values[i]))
		
		similar = sorted(similar, reverse=False)
		cluster_info_str = ''
		if selection:
			if (similar_tuple_list[0][0] < min_dist_thres) and (similar_tuple_list[-1][0] < max_dist_thres) and ((similar_tuple_list[0][0]+0.5) > similar_tuple_list[-1][0]): #  # the smallest and biggest distance to the centroid should not be very different, we allow 0.4 for now!

				cluster_info_str+="cluster number and size are: "+str(cn)+'    '+str(clustersizes[str(cn)]) + "\n"
					
				
				for txt in tweetsDF[tok_result_col].values[similar_indices]:
					cluster_bigram_cntr.update(get_uni_bigrams(txt)) #regex.findall(r"\b\w+[-]?\w+\s\w+", txt, overlapped=True))
					#cluster_bigram_cntr.update(txt.split()) # unigrams
				frequency = reverse_index_frequency(cluster_bigram_cntr)
				topterms = [k+":"+str(v) for k,v in cluster_bigram_cntr.most_common() if k in word_vectorizer.get_feature_names()]  
				if len(topterms) < 2:
					continue # a term with unknown words, due to frequency threshold, may cause a cluster. We want analyze this tweets one unknown terms became known as the freq. threshold decrease.
				cluster_info_str+="Top terms are:"+", ".join(topterms) + "\n"

				cluster_info_str+="distance_to_centroid"+"\t"+"tweet_id"+"\t"+"tweet_text\n"
				
				if len(similar)>20:
					cluster_info_str+='First 10 documents:\n'
					cluster_info_str+= "\n".join(similar[:10]) + "\n"
					#print(*similar[:10], sep='\n', end='\n')

					cluster_info_str+='Last 10 documents:\n'
					cluster_info_str+= "\n".join(similar[-10:]) + "\n"
				else:
					cluster_info_str+="Tweets for this cluster are:\n"
					cluster_info_str+= "\n".join(similar) + "\n"
				
		else:
				cluster_info_str+="cluster number and size are: "+str(cn)+'    '+str(clustersizes[str(cn)]) + "\n"
					
				cluster_bigram_cntr = Counter()
				for txt in tweetsDF[tok_result_col].values[similar_indices]:
					cluster_bigram_cntr.update(get_uni_bigrams(txt))
					
				frequency = reverse_index_frequency(cluster_bigram_cntr)
				
				topterms = [k+":"+str(v) for k,v in cluster_bigram_cntr.most_common() if k in word_vectorizer.get_feature_names()]  
				if len(topterms) < 2:
					continue # a term with unknown words, due to frequency threshold, may cause a cluster. We want analyze this tweets one unknown terms became known as the freq. threshold decrease.
				cluster_info_str+="Top terms are:"+", ".join(topterms) + "\n"

				cluster_info_str+="distance_to_centroid"+"\t"+"tweet_id"+"\t"+"tweet_text\n"
					
				if len(similar)>20:
					cluster_info_str+='First 10 documents:\n'
					cluster_info_str+= "\n".join(similar[:10]) + "\n"
					#print(*similar[:10], sep='\n', end='\n')

					cluster_info_str+='Last 10 documents:\n'
					cluster_info_str+= "\n".join(similar[-10:]) + "\n"
				else:
					cluster_info_str+="Tweets for this cluster are:\n"
					cluster_info_str+= "\n".join(similar) + "\n"
		
		if len(cluster_info_str) > 0: # that means there is some information in the cluster.
			cluster_str_list.append({'cno':cn, 'cnoprefix':nameprefix+str(cn), 'rif':frequency, 'cstr':cluster_info_str, 'ctweettuplelist':similar_tuple_list,  'twids':list(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)})

	return cluster_str_list
	

        
if __name__ == "__main__":
	import output
	
	tweetlist = read_json_tweets_database(args.lang)
	logging.info("number of tweets",len(tweetlist))
	
	tweetsDF = create_dataframe(tweetlist)
	
	tok = tok_results(tweetsDF)
	
	start_tweet_size = len(tweetsDF)
	print("\nNumber of the tweets after retweet elimination:", start_tweet_size)

	print("Choose mode of the annotation.")
	
	print("1. relevant vs. irrelevant (default)")
	information_groups = {"relevant":[],"irrelevant":[]} # contains IDs of tweets, updated after each clustering/labeling cycle

	print("2. provide a list of labels")
	print("3. define labels during exploration")
	mchoice = input("Your choice:")

	if mchoice == '2':
		mylabels = input("Enter a comma seperated label list:").split(",")
		information_groups = {k:[] for k in mylabels}
		print("Labels are:", [l for l in sorted(list(information_groups.keys()))])


	identified_tweet_ids = []
	
	while True:

		km, doc_feat_mtrx, word_vectorizer = output.create_clusters(tweetsDF[tok_result_col])

		print("\nThe silhouette score (between 0 and 1, the higher is the better):", metrics.silhouette_score(doc_feat_mtrx, km.labels_, metric='euclidean',sample_size=5000))

		clustersizes = get_cluster_sizes(km, tweetsDF[tok_result_col].values)
		print("\nCluster sizes:",clustersizes.most_common())

		print("cluster candidates:", end=' ')

		#local_information_groups = {'noise':[]} # contains cluster number for this iteration. It should pass the tweet ID information to the global_information_groups at end of each cycle
		output_cluster_list = output.get_candidate_clusters(km, doc_feat_mtrx, word_vectorizer, tweetsDF, tok_result_col, min_dist_thres, max_dist_thres)

		if len(output_cluster_list) == 0:
			print("\nThere is not any good group candidate in the last clustering. New clustering with the relaxed selection conditions.")
			
			min_dist_thres += 0.01
			max_dist_thres += 0.01
			print("Relaxed distance thresholds for the group selection are:", min_dist_thres, max_dist_thres)
			time.sleep(5)
			continue

		for cn, c_str, tw_ids in output_cluster_list:
			#for cn, csize in clustersizes.most_common():
			#	cn = int(cn)
			
			print('\n'+c_str)
			#print("current ids:", tw_ids) # remove this after testing
			available_labels_no = [str(i)+"-"+l for i,l in enumerate(sorted(list(information_groups.keys())))]
			available_labels = [l for l in sorted(list(information_groups.keys()))] # do not move it to somewhere else, because in explorative mode, label updates should be reflected.
			print("\n Available labels:", available_labels_no)

			cluster_label = input("Enter (number of) an available or a new label (q to quit the iteration, Enter to skip making a decision for this group):")
			if cluster_label == 'q':
				break
			elif cluster_label == '':
				print("This group of tweets were skipped for this iteration!")
				continue

			elif cluster_label.isdigit() and int(cluster_label) < len(information_groups):
				information_groups[available_labels[int(cluster_label)]] += tw_ids #list(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
				print("Group assigned to the available label:", available_labels[int(cluster_label)])
				print("Cluster number:", cn, "its label:", available_labels[int(cluster_label)])
			elif mchoice == '3':
				if cluster_label not in information_groups: # in order to avoid overwriting content of a label
					information_groups[cluster_label] = []
				information_groups[cluster_label] += tw_ids #list(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
				print("Group assigned to a NEW label:", cluster_label)
				print("Cluster number:", cn, "its label:", cluster_label)
			#print(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
			else:
				print("\nEnter a label or label number that is available. If you want to add new labels as you are exploring you should activate the explorative mode:3 from the previous step. \n Annotation of this group will be skipped.")
				
			print("Next cluster is coming ...")
			time.sleep(1) # wait while showing result of the assignment

		print('\nEnd of Iteration, available groups:')
		print(*information_groups, sep='\n', end='\n\n')
			
		for k,v in information_groups.items():
			identified_tweet_ids += v

		identified_tweet_ids = list(set(identified_tweet_ids)) # dublicates is added in successive iterations.

		print('number of classified tweets:',len(identified_tweet_ids))

		tweetsDF = tweetsDF[~tweetsDF.id_str.isin(identified_tweet_ids)]

		print('number of remaining tweets to be identified:', len(tweetsDF))


		print('current labeled ratio is:', len(identified_tweet_ids)/len(tweetsDF))
		print('target labeling ratio is:', target_labeling_ratio)
		print("current distance thresholds are:", min_dist_thres, max_dist_thres)
		time.sleep(5)
		
		if len(identified_tweet_ids)/len(tweetsDF) < target_labeling_ratio:
			print('\nNew clustering will be done to achieve the target.')
			continue # 
		else: # else ask the user.
			iter_choice = input("\nTarget was achieved. Press y if you want to do one more iteration:")
			if iter_choice == 'y': # if the user enter y, the infinite loop should be continued!
				continue

		# This will run after we reach the target or decide not to continue labeling any more after we actively decide to label mor eafter we reach the target.
		label_rest_tweets = input("Press y if you want to label the remaining tweets without any selection criteria:")
		if label_rest_tweets == 'y':
			km, doc_feat_mtrx, word_vectorizer = output.create_clusters(tweetsDF[tok_result_col])
			
			for cn, c_str, tw_ids in output.get_candidate_clusters(km, doc_feat_mtrx, word_vectorizer, tweetsDF, tok_result_col, min_dist_thres, max_dist_thres, selection=False):
				#for cn, csize in clustersizes.most_common():
				#	cn = int(cn)
				
				print('\n'+c_str)
				#print("current ids:", tw_ids) # remove this after testing
				available_labels_no = [str(i)+"-"+l for i,l in enumerate(sorted(list(information_groups.keys())))]
				available_labels = [l for l in sorted(list(information_groups.keys()))] # do not move it to somewhere else, because in explorative mode, label updates should be reflected.
				print("\n Available labels:", available_labels_no)

				cluster_label = input("Enter (number of) an available or a new label (q to quit the iteration, Enter to skip making a decision):")
				if cluster_label == 'q':
					break
				elif cluster_label == '':
					print("This group of tweets were skipped for this iteration!")
					continue

				elif cluster_label.isdigit() and int(cluster_label) < len(information_groups):
					information_groups[available_labels[int(cluster_label)]] += tw_ids #list(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
					print("Group assigned to the available label:", available_labels[int(cluster_label)])
					print("Cluster number:", cn, "its label:", available_labels[int(cluster_label)])
				elif mchoice == '3':
					if cluster_label not in information_groups: # in order to avoid overwriting content of a label
						information_groups[cluster_label] = []
					information_groups[cluster_label] += tw_ids #list(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
					print("Group assigned to a NEW label:", cluster_label)
					print("Cluster number:", cn, "its label:", cluster_label)
				#print(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
				else:
					print("\nEnter a label or label number that is available. If you want to add new labels as you are exploring you should activate the explorative mode:3 from the previous step. \n Annotation of this group will be skipped.")
					
				print("Next cluster is coming ...")
				time.sleep(1) # wait while showing result of the assignment
		
		break # go out of the infinite while loop		
		
	print("Ask if they want to write the groups to a file, which features are needed, which file formats: json, tsv?")
	for k, v in information_groups.items():
		group_tweets = [tw for tw in tweetlist if tw["id_str"] in v]
		print("length of the tweets and ids (they must be equal!)", len(group_tweets), len(v)) # they should be equal!

	rlvdb[result_collection].drop() # be sure to overwrite it! Instead of overwriting, it can be inserted by adding a date and time to this dictionary.
	rlvdb[result_collection].insert(information_groups)
	print("The result was written to the collection:", result_collection)