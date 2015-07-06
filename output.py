
import math as m
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter, namedtuple
import scipy as sp
import numpy as np
import regex

def create_clusters(tweetsSrs):

	freqcutoff = int(m.log(len(tweetsSrs))/2)
	
	word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, norm='l2', min_df=freqcutoff, token_pattern=r"\b\w+\b|[\U00010000-\U0010ffff]")
	text_vectors = word_vectorizer.fit_transform(tweetsSrs)
	
	doc_feat_mtrx = text_vectors
		
	n_clust = int(m.sqrt(len(tweetsSrs)))
	n_initt = int(m.log10(len(tweetsSrs)))
	
	km = KMeans(n_clusters=n_clust, init='k-means++', max_iter=1000, n_init=n_initt) # , n_jobs=16
	km.fit(doc_feat_mtrx)

	return km, doc_feat_mtrx, word_vectorizer
	
def get_cluster_sizes(kmeans_result,doclist):
	clust_len_cntr = Counter()
	for l in set(kmeans_result.labels_):
		clust_len_cntr[str(l)] = len(doclist[np.where(kmeans_result.labels_ == l)])
	return clust_len_cntr
	

def get_candidate_clusters(clusters, doc_feat_mtrx, word_vectorizer, tweetsDF, tok_result_col, min_dist_thres, max_dist_thres, printsize=True, selection=True):
	cluster_str_list = []
	Cluster = namedtuple('Cluster', ['cno', 'cstr','tw_ids'])

	clustersizes = get_cluster_sizes(clusters, tweetsDF[tok_result_col].values)
	
	for cn, csize in clustersizes.most_common():#range(args.ksize):#clustersizes.most_common():
		cn = int(cn)
					
		similar_indices = (clusters.labels_== cn).nonzero()[0]
			
		similar = []
		for i in similar_indices:
			dist = sp.linalg.norm((clusters.cluster_centers_[cn] - doc_feat_mtrx[i]))
			similar.append(str(dist) + "\t" + tweetsDF['id_str'].values[i]+"\t"+ tweetsDF[tok_result_col].values[i])
		
		similar = sorted(similar, reverse=False)
		cluster_info_str = ''
		if selection:
			if (float(similar[0][:4]) < min_dist_thres) and (float(similar[-1][:4]) < max_dist_thres) and ((float(similar[0][:4])+0.5) > float(similar[-1][:4])): #  # the smallest and biggest distance to the centroid should not be very different, we allow 0.4 for now!

				cluster_info_str+="cluster number and size are: "+str(cn)+'    '+str(clustersizes[str(cn)]) + "\n"
					
				cluster_bigram_cntr = Counter()
				for txt in tweetsDF[tok_result_col].values[similar_indices]:
					cluster_bigram_cntr.update(regex.findall(r"\b\w+[-]?\w+\s\w+", txt, overlapped=True))
					cluster_bigram_cntr.update(txt.split()) # unigrams
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
					cluster_bigram_cntr.update(regex.findall(r"\b\w+[-]?\w+\s\w+", txt, overlapped=True))
					cluster_bigram_cntr.update(txt.split()) # unigrams
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
			cluster_str_list.append(Cluster(cno=cn, cstr=cluster_info_str, tw_ids=list(tweetsDF[np.in1d(clusters.labels_,[cn])]["id_str"].values)))

	return cluster_str_list


	
