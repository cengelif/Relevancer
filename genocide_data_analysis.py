import timeit
import importlib
import relevancer as rlv
import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB 
from bson.objectid import ObjectId
from collections import Counter
import numpy as np

pd.set_option("display.max_colwidth",200)

def eliminate_duplicates_recursively(df, duplicate_elim_func):
            
        print("starting, length:",len(df))
        df = df.reindex(np.random.permutation(df.index))
        df.reset_index(inplace=True, drop=True)
        
        tmp_df2 = pd.DataFrame()
        for i in range(0, len(df), 10000):
            tmp_unique = duplicate_elim_func(df[i:i+1000], similarity_threshold=0.20, debug=True, debug_threshold=10000)
            tmp_df2 = pd.concat([tmp_df2, tmp_unique], ignore_index=True)
            
        if len(df) > len(tmp_df2):
            print(str(len(df) - len(tmp_df2))+" tweets were eliminated!")
            return eliminate_duplicates_recursively(tmp_df2, duplicate_elim_func)
        
        return df

def clustering():

	# this is just to load the latest version of the Relevancer after we edit the code.
	importlib.reload(rlv)
	active_col = "active_text"
	rlv.set_active_column(active_col)


	my_token_pattern=r"[#@]?\w+\b|[\U00010000-\U0010ffff]"
	rlvdb, rlvcl = rlv.connect_mongodb(configfile='data/localdb.ini',coll_name="testcl")
	print("DB connected")

	# after the first iteration, the annotated clusters should be excluded from the clustering.
	# read this file from the tagged collection.
	annotated_tw_ids = ['563657483395530753', '563662532326330370', '563654330041909248', '563654944927281152', '563657924233289728', '563661021559390208', '563651950386757632', '563657164317667328', '563660271810383872', '563662538949160960'] #You should get the actual annotated tweet ids from the annotated tweets collection.
	#annotated_tw_ids = []
	# mongo_query=({'_id': {'$gte': begin, '$lte': end},'lang':'en'})

	tweetlist = rlv.read_json_tweet_fields_database(rlvcl, mongo_query=({}), read_fields={'text': 1, 'id_str': 1, '_id': 0, 'user_id': 1}, tweet_count=-1, annotated_ids=annotated_tw_ids)#=tweetsDF)

	rlv.logging.info("Number of tweets:" + str(len(tweetlist)))
	print("Number of tweets:",len(tweetlist))


	tweetsDF = rlv.create_dataframe(tweetlist)

	#tweetsDF = tweetsDFBackUP.copy()

	tweetsDF[active_col] = tweetsDF["text"].copy()
	tweetsDF = rlv.tok_results(tweetsDF, elimrt = True)



	tweetsDF = rlv.normalize_text(tweetsDF)




	tweetsDF_uniq = eliminate_duplicates_recursively(tweetsDF.copy(), rlv.get_and_eliminate_near_duplicate_tweets)

	print("eliminate_duplicates_recursively")

	tweetsDF_uniq.to_pickle("20151005_unique_genocide_tweets.pickle")


	cluster_list = rlv.create_clusters(tweetsDF_uniq, my_token_pattern, min_dist_thres=0.725, max_dist_thres=0.875, min_max_diff_thres=0.4, nameprefix='1-', min_clusters=100, user_identifier='user_id')


	rlvdb2, rlvcl2 = rlv.connect_mongodb(configfile='data/localdb.ini',coll_name="genocide_test")

	collection_name = 'genocide_test'
	rlvdb2[collection_name].insert(cluster_list) #Each iteration results with a candidate cluster list. Each iteration will have its own list. Therefore they are not mixed.

	print("db written")

	with open("genocide_clusters_20151005.pickle", 'wb') as f:
	    pickle.dump(cluster_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

	clustering()


