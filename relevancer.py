import configparser
import sys
import pymongo
import logging
import argparse
import json
import datetime
import time
import math as m
from collections import Counter, OrderedDict
from html.parser import HTMLParser

import pandas as pd
import numpy as np
import scipy as sp
import regex

from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics


#Logging
logging.basicConfig(filename='/home/elif/relevancer/myapp.log',
                            #filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%d-%m-%Y, %H:%M',
                            level=logging.INFO)

logging.info("Script started")


#Config Parser
config = configparser.ConfigParser()
config.read('/home/elif/relevancer/myconfig.ini')

#MongoLab OAuth;
client_host = config.get('mongodb', 'client_host')
client_port = int(config.get('mongodb', 'client_port'))
db_name = config.get('mongodb', 'db_name')
user_name = config.get('mongodb', 'user_name')
passwd = config.get('mongodb', 'passwd')

#Connect to database
try:
	connection = MongoClient(client_host, client_port)
	rlvdb = connection[mydb]  #Database
	rlvdb.authenticate(user_name, passwd)
	rlvcl = rlvdb.coll123 #Collection
	logging.info('Connected to Database')
except Exception:
	logging.error("Database Connection Failed!")
	sys.exit("Database connection failed!")
	pass
 

parser = argparse.ArgumentParser(description='Detect information groups in a microtext collection')

#parser.add_argument('integers', metavar='N', type=int, nargs='+',
#				   help='an integer for the accumulator')
#parser.add_argument('--sum', dest='accumulate', action='store_const',
#				   const=sum, default=max,
#				   help='sum the integers (default: find the max)')


parser.add_argument('-f', '--infile', type=str, help='input file should contain the microtexts') # format of the file should be defined later.
parser.add_argument('-l', '--lang', type=str, default='en', help='language of the microtext that will be selected to be processed further.')

args = parser.parse_args()

min_dist_thres = 0.65 # the smallest distance of a tweet to the cluster centroid should not be bigger than that.
max_dist_thres = 0.85 # the biggest distance of a tweet to the cluster centroid should not be bigger than that.
target_labeling_ratio = 0.7 # percentage of the tweets that should be labeled, until this ratio achieved iteration will repeat automatically.

print("The microtext file is:", args.infile)
print("The language to be processed is:", args.lang)

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
		'''self.toReplaceDict = OrderedDict({'!!*':' ! ','\?':' ? ', '\"':' " ',"“":" “ ","”":" ” ", "\'\'*":"'","\' ":" ' "
	," \'":" ' ","’ ":" ’ ",'&amp;':'&','&gt;':'>','&lt;':'<', '~~*':' ~ ',"¿¿*":" ¿ ",'\.\.\.':' ... ','\.\.':' .. '
	,'…':' … ',"\(\(*":'(',"\)\)*":')',"\+\+*":'+',"\*\**":'*',"\|\|*":"|","\$\$*":"$","%%*":"%",">>*":">","<<*":"<","--*":"-" 
	,"\/\/\/*":"//","(:d)(:d)*":":d",":ddd*":" :d ",":ppp*":" :p ",";;;*":";",":\* ":" :* ",":\(":" :( ","\(:":" (: ",":\)":" :) "
	,'\):':' ): ',";\)":" ;) ","\+\+":" + ",":\|":" :| ",":-\)":" :-) ",";-\)":" ;-) ",":-\(":" :-( ",":\'\(":" :'( ",":p ":" :p "
	,";p ":" ;p ",":d ":" :d ","-_-":" -_- ",":o\)":" :o) ",":\$":" :$ ","\.@":". @",'#':' #',' \.': ' . ','	':' '
	,'   ':' ','   ':' ','  ':' ',"😡😡*":" :( ","☺️☺️*":" :) ","😄😄*":" :d ","😃😃*":" :d ","😆😆*":" :d ","😷😷*":" :d "
	,"😅😅*":" :d ","😋😋*":" :d ","😜😜*":" :p " ,"😝😝*":" :p ","😂😂*":" :'( ","😢😢*":" :'( ","😁😁*":" :( ","😞😞*":" :( "
	,"😖😖*":" :( " ,"😥😥*":" :( ","😩😩*":" :( ","😊😊*":" :) ","😉😉*":" :) ","😎😎*":" :) " ,"😇😇*":" :) ","😭😭*":" :'d " 
	,"😨😨*":" :| ","😏😏*":" :| " ,"😔😔*":" :| ","😒😒*":" :| ","😫😫*":" :( ","😪😪*":" :'( "
	,"😰😰*":" :'( " ,"😍😍*":" <3 ","😘😘*":" <3 ","<33*":" <3 ","<3(<3)*":" <3 ","😳😳*":" 😳 ", "😻😻*":" 😻 ", "\n\n*":" \n ", "♪♪*":" ♪ "
	,"💧💧*":" 💧 ", """\xa0""":" ", "\n":" . ","【【*":" 【 ","】】*":" 】 ","「「*":" 「 ","」」*":" 」 ","❤️❤️*":" <3 ","🎶🎶*":" 🎶 "
	,"😌😌*":" :) ","💖💖*":" <3 ","😐😐*":" :| ","\.: ":" .: "})'''
	
	# '\. ': ' . ' --> deleted from toReplaceDict to be able to process the abbreviations. 
	
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

def get_cluster_sizes(kmeans_result,doclist):
	clust_len_cntr = Counter()
	for l in set(kmeans_result.labels_):
		clust_len_cntr[str(l)] = len(doclist[np.where(kmeans_result.labels_ == l)])
	return clust_len_cntr

if __name__ == "__main__":

	tok_result_col = "texttokCap"
	tok_result_lower_col = "texttok"

	tweetlist = read_json_tweets_file(args.infile)
	tweetsDF = pd.DataFrame(tweetlist)
	tweetsDF.set_index("created_at", inplace=True)
	tweetsDF.sort_index(inplace=True)
	print("\nNumber of the tweets:",len(tweetsDF))
	print("\nAvailable attributes of the tweets:",tweetsDF.columns)

	twtknzr = Twtokenizer()
	tweetsDF = twtknzr.tokenize_df(tweetsDF, texcol='text', rescol=tok_result_col, addLowerTok=False)
	tweetsDF[tok_result_lower_col] = tweetsDF[tok_result_col].str.lower()
	print("\nAvailable attributes of the tokenized tweets:",tweetsDF.columns)
	print("\ntweet set summary:", tweetsDF.info())
	print(tweetsDF[tok_result_col][:5])

	rttext = ~tweetsDF[tok_result_lower_col].str.contains(r"\brt @")
	rtfield = tweetsDF["is_retweet"]==False
	tweetsDF["is_notrt"] = rtfield.values & rttext.values # The default setting is to eliminate retweets
	tweetsDF = tweetsDF[tweetsDF.is_notrt]
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

		#freqcutoff = int(m.sqrt(len(tweetsDF))/5)
		freqcutoff = int(m.log(len(tweetsDF))/2)
		print("Frequency cutoff is:", freqcutoff)
		#To use UCS-4 write in like "[\U00010000-\U0010ffff]" format and to use UCS-2 write in like "[\uD800-\uDBFF][\uDC00-\uDFFF]" format. 
		#If you use try/except statement it can be more effective.
		word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, norm='l2', min_df=freqcutoff, token_pattern=r"\b\w+\b|[\U00010000-\U0010ffff]")
		text_vectors = word_vectorizer.fit_transform(tweetsDF[tok_result_col])
		#if text_vectors is None:
		#	print('True')
		#else:
		#	print('False')
		#print("feature list:", word_vectorizer.get_feature_names()) 
		# get_feature_names()[source]
		#input('***************************************')
		#time.sleep(10)
		doc_feat_mtrx = text_vectors
		print("\nshape of the document - feature matrix:",text_vectors.shape)
		#reducer = PCA(n_components=int(text_vectors.shape[1]/5))
		#reduced_X = reducer.fit_transform(text_vectors.toarray())
		#print("\nshape of the document - feature matrix after PCA:", reduced_X.shape)
		#doc_feat_mtrx = reduced_X # assign which one to use!
		
		n_clust = int(m.sqrt(len(tweetsDF)))
		n_initt = int(m.log10(len(tweetsDF)))
		print('number of clusters:', n_clust, "number of clustering inits:", n_initt)

		km = KMeans(n_clusters=n_clust, init='k-means++', max_iter=1000, n_init=n_initt) # , n_jobs=16
		km.fit(doc_feat_mtrx)

		print("\nThe silhouette score (between 0 and 1, the higher is the better):", metrics.silhouette_score(doc_feat_mtrx, km.labels_, metric='euclidean',sample_size=5000))

		clustersizes = get_cluster_sizes(km, tweetsDF[tok_result_col].values)
		print("\nCluster sizes:",clustersizes.most_common())

		print("cluster candidates:", end=' ')

		#local_information_groups = {'noise':[]} # contains cluster number for this iteration. It should pass the tweet ID information to the global_information_groups at end of each cycle
		
		candidate_cluster_count = 0
		for cn, csize in clustersizes.most_common():#range(args.ksize):#clustersizes.most_common():
			cn = int(cn)
					
			similar_indices = (km.labels_== cn).nonzero()[0]
			
			similar = []
			for i in similar_indices:
				dist = sp.linalg.norm((km.cluster_centers_[cn] - doc_feat_mtrx[i]))
				similar.append(str(dist) + "\t" + tweetsDF['id_str'].values[i]+"\t"+ tweetsDF[tok_result_col].values[i])
				
			similar = sorted(similar, reverse=False)
			if (float(similar[0][:4]) < min_dist_thres) and (float(similar[-1][:4]) < max_dist_thres) and ((float(similar[0][:4])+0.5) > float(similar[-1][:4])): #  # the smallest and biggest distance to the centroid should not be very different, we allow 0.4 for now!
				
				candidate_cluster_count += 1
				print("cluster number and size are: "+str(cn)+'    '+str(clustersizes[str(cn)]))
				
				cluster_bigram_cntr = Counter()
				for txt in tweetsDF[tok_result_col].values[similar_indices]:
					cluster_bigram_cntr.update(regex.findall(r"\b\w+[-]?\w+\s\w+", txt, overlapped=True))
					cluster_bigram_cntr.update(txt.split()) # unigrams
				topterms = [k+":"+str(v) for k,v in cluster_bigram_cntr.most_common() if k in word_vectorizer.get_feature_names()]  
				if len(topterms) < 2:
					continue # a term with unknown words, due to frequency threshold, may cause a cluster. We want analyze this tweets one unknown terms became known as the freq. threshold decrease.
				print("\nTop terms are:", ", ".join(topterms))

				print("\ndistance_to_centroid"+"\t"+"tweet_id"+"\t"+"tweet_text")
				
				if len(similar)>20:
					print('\nFirst 10 documents:')
					print(*similar[:10], sep='\n', end='\n')

					print('\nLast 10 documents:')
					print(*similar[-10:], sep='\n', end='\n')
				else:
					print("Tweets for this cluster are:")
					print(*similar, sep='\n', end='\n')

				available_labels_no = [str(i)+"-"+l for i,l in enumerate(sorted(list(information_groups.keys())))]
				available_labels = [l for l in sorted(list(information_groups.keys()))] # do not move it to somewhere else, because in explorative mode, label updates should be reflected.
				print("\n Available labels:", available_labels_no)

				cluster_label = input("Enter (number of) an available or a new label (q to quit the iteration, Enter to postpone decision):")
				if cluster_label == 'q':
					break
				elif cluster_label == '':
					print("This group of tweets were skipped for this iteration!")
					continue

				elif cluster_label.isdigit() and int(cluster_label) < len(information_groups):
						information_groups[available_labels[int(cluster_label)]] += list(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
						print("Group assigned to the available label:", available_labels[int(cluster_label)])
						print("Cluster number:", cn, "its label:", available_labels[int(cluster_label)])
				elif mchoice == '3':
						if cluster_label not in information_groups: # in order to avoid overwriting content of a label
							information_groups[cluster_label] = []
						information_groups[cluster_label] += list(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
						print("Group assigned to a NEW label:", cluster_label)
						print("Cluster number:", cn, "its label:", cluster_label)
					#print(tweetsDF[np.in1d(km.labels_,[cn])]["id_str"].values)
				else:
					print("\nEnter a label or label number that is available. If you want to add new labels as you are exploring you should activate the explorative mode:3 from the previous step. Annotation of this group will be skipped.")
				
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

		if candidate_cluster_count == 0: # if there is not any candidate in this clustering result, repeat the clustering.
			print("\nThere is not any good group candidate, new iteration is starting ...")
			# relax the conditions for a group to be considered.
			min_dist_thres += 0.01
			max_dist_thres += 0.01
			print("Relaxed distance thresholds are:", min_dist_thres, max_dist_thres)
			time.sleep(5)
			continue
		elif len(identified_tweet_ids)/len(tweetsDF) < target_labeling_ratio:
			print('\nNew clustering will be done to achieve the target.')
			print('current labeled ratio is:', len(identified_tweet_ids)/len(tweetsDF))
			print('target labeling ratio is:', target_labeling_ratio)

			print('\ncandidate cluster count:',candidate_cluster_count)
			print("current distance thresholds are:", min_dist_thres, max_dist_thres)
			time.sleep(5)
		else: # else ask the user.
			iter_choice = input("Target was achieved. Press y if you want to do one more iteration:")
			if iter_choice != 'y':
				break

	print("Ask if they want to write the groups to a file, which features are needed, which file formats: json, tsv?")
	for k, v in information_groups.items():
		group_tweets = [tw for tw in tweetlist if tw["id_str"] in v]
		print("length of the tweets and ids (they must be equal!)", len(group_tweets), len(v)) # they should be equal!
		#with open(k,'w') as fw:
		# json.dumps
