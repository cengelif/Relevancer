import relevancer as rlv

rlvdb, rlvcl = rlv.connect_mongodb()

tweetlist = rlv.read_json_tweets_database(rlvcl, mongo_query={}, tweet_count=1000, reqlang='en')
rlv.logging.info("number of tweets"+str(len(tweetlist)))
	
tweetsDF = rlv.create_dataframe(tweetlist)
	
tok = rlv.tok_results(tweetsDF)

start_tweet_size = len(tweetsDF)
rlv.logging.info("\nNumber of the tweets after retweet elimination:"+ str(start_tweet_size))

cluster_list = rlv.create_clusters(tweetsDF) # those comply to slection criteria
cluster_list2 = rlv.create_clusters(tweetsDF, selection=False) # get all clusters. You can consider it at the end.

print (len(cluster_list))  


a_cluster = cluster_list[1]

print("cluster_no", a_cluster['cno'] )

print("cluster_str", a_cluster['cstr'] )

print("cluster_tweet_ids", a_cluster['twids'] )

collection_name = 'clusters'
rlvdb[collection_name].insert(cluster_list)

print("Clusters were written to the collection:", collection_name)


# After excluding tweets that are annotated, you should do the same iteration as many times as the user would like.
# You can provide a percentage of annotated tweets to inform about how far is the user in annotation.
