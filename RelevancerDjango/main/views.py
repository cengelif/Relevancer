# Django
from django.shortcuts import render
from django.views.generic import View
from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse

from django.conf import settings

# Python
import re
import os
import sys
sys.path.append('../') # adds 'Relevancer' folder to PYTHONPATH to find relevancer.py etc.
import json
import time
import random
import logging
import configparser
from datetime import datetime
from bson.json_util import dumps

# DB / models
import mongoengine
from main.models import * # Clusters, CollectionList (Have to import everything(with star) because the models can be added dynamically)
from mongoengine.base.common import get_document
import pymongo

# Our Own Sources
#import genocide_data_analysis


config = configparser.ConfigParser()

config.read("data/auth.ini")

logging.basicConfig(
		format='%(asctime)s, %(levelname)s: %(message)s',
		filename='data/relevancer.log',
		datefmt='%d-%m-%Y, %H:%M',
		level=logging.INFO)


############################## FUNCTIONS #############################


def get_randomcluster(collname, is_labeled): 

# !! reorganize here.. DRY!!

	random_cluster =  None
	current_label = ""
	warning = ""
	top10 = []
	last10 = []

	model = get_document(collname)

	if(is_labeled == "True"):

		num_of_clusters = model.objects(label__exists = True).count()

		if(num_of_clusters > 0):

			rand = random.randint(0, num_of_clusters-1)

			random_cluster = model.objects(label__exists = True)[rand]

			current_label = random_cluster["label"]

			tweetlist = []
			for cl in random_cluster["ctweettuplelist"]:
				tweetlist.append(cl[2])

			if(len(tweetlist) > 20):

				top10 = tweetlist[:10]

				last10 = tweetlist[-10:]
			
			else:
				top10 = tweetlist  #All tweets

		else:

			warning = "There is not any labeled cluster yet"


	elif(is_labeled == "False"):

		num_of_clusters = model.objects(label__exists = False).count()

		if(num_of_clusters > 0):

			rand = random.randint(0, num_of_clusters-1)

			random_cluster = model.objects(label__exists = False)[rand]

			tweetlist = []
			for cl in random_cluster["ctweettuplelist"]:
				tweetlist.append(cl[2])

			if(len(tweetlist) > 20):

				top10 = tweetlist[:10]

				last10 = tweetlist[-10:]
			
			else:
				top10 = tweetlist #All tweets

		else:

			warning = "All clusters are labeled."


	return random_cluster, top10, last10, current_label, warning



def get_step_data(collname, num, page=None):

	model = get_document(collname)

	tweets = []

	if(page == "Cluster_Them"):
		for item in model.objects[5:num+5]:
			for i in item["ctweettuplelist"][:10]:
					tweets.append(i[2])
			tweets.append("------------------")

	else:
		for item in model.objects[5:num+5]:
			tweets.append(item["text"])

	
	length = model.objects.count()

	return tweets, length



def get_labels(collname):

	model = get_document(collname)

	all_labels = model.objects(label__exists = True).only("label")

	label_set = []
	for lbl in all_labels:
		if(lbl["label"] not in label_set):
			label_set.append(lbl["label"])

	num_of_cl = []
	for labl in label_set:
		num_of_cl.append(model.objects(label = labl).count())

	labellist = zip(label_set, num_of_cl)	

	return labellist
	


def get_collectionlist(choice):

	# Object is called to update in "addcollection" part
	colllist_obj = CollectionList.objects.first()

	colllist = colllist_obj["collectionlist"]

	if(choice == "info"):
		len_coll = []
		len_unlabeled = []
		len_labeled= []

		for coll in colllist:
	
			model = get_document(coll)
			len_unlbld = model.objects(label__exists = False).count()
			len_lbld = model.objects(label__exists = True).count()
	
			len_coll.append(len_unlbld + len_lbld)	
			len_unlabeled.append(len_unlbld)
			len_labeled.append(len_lbld)


		collectionlist = zip(colllist, len_coll, len_unlabeled, len_labeled)

		return collectionlist

	elif(choice == "update"):

		return colllist_obj, colllist



def backup_json(collname):

		currDate = datetime.now().strftime("%y%m%d-%H:%M") 

		filename = collname + "_" + currDate + ".json"

		model = get_document(collname)
		
		with open("data/backups/" + filename, 'w') as f:
			f.write(model.objects.to_json() + '\n')

		logging.info('Backup to ' + filename)

		return 0


def loadbackup(filename):

	collname = re.findall('(.*)_\d{6}-\d\d:\d\d.json', filename)[0]

	db_name = config.get('db', 'db_name')

	backup_json(collname)
	print("backup")

	client = pymongo.MongoClient("127.0.0.1", 27017)
	localdb = client[db_name]
	local_col = localdb[collname]
	print("connected")

	with open("data/backups/" + filename) as f:
		cluster_list = json.load(f)
	print("loaded")

	for element in cluster_list: 
		del element['_id'] 
	print("idremoved")

	local_col.remove()
	print("removed")

	local_col.insert(cluster_list)
	print("inserted")

	return 0





############################## VIEWS #############################


class Home(View):

	def get(self, request):
				
		
		return render(request, 'base.html', {	

		})

	

class Datasets(View):

	def get(self, request):
				
		collectionlist = get_collectionlist("info")

		return render(request, 'datasets.html', {	
				'collectionlist' : collectionlist,
		})


	def post(self, request):

			if "addcollection" in request.POST:				

				newcollection = request.POST['newcollection']

				colllist_obj, colllist = get_collectionlist("update")

				colllist.append(newcollection)

				colllist_obj.update(set__collectionlist = colllist)


				with open("main/models.py", "a") as myfile:
   						myfile.write("\nclass " + newcollection + "(Clusters):\n\n\t meta = {'collection': '" + newcollection + "'}\n")

				time.sleep(1) #temporary solution to prevent direct crush

				collectionlist = get_collectionlist("info")

				return render(request, 'datasets.html', {	
						'collectionlist' : collectionlist,
				})



class Backup(View):

	def get(self, request, collname):

		backup_json(collname)

		url = reverse('listbackups', kwargs={'collname': collname})
		
		return HttpResponseRedirect(url)



class ListBackups(View):

	def get(self, request, collname):

		filelist = []

		for file in os.listdir("data/backups/"):
			if file.startswith(collname):
				filelist.append(file)

		filelist.sort(reverse=True)

		return render(request, 'loadback.html', {	
				'collname' : collname,
				'filelist' : filelist,
		})		



class LoadBack(View):

	def get(self, request, filename):

		loadbackup(filename)

		return HttpResponseRedirect('/datasets')



class ResetLabels(View):

	def get(self, request, collname):


		return render(request, 'resetlabels.html', {	
				'collname' : collname,
		})


	def post(self, request, collname):

			if "confirmpass" in request.POST:				

				user_pass = request.POST['user_pass']

				reset_pass = config.get('dataset', 'reset_pass')

				if(user_pass == reset_pass):

					backup_json(collname)
	
					model = get_document(collname)
					
					model.objects.update(unset__label=1)

					logging.info('Reset All Labels.')

					confirmed = True

					return render(request, 'resetlabels.html', {	
							'collname' : collname,
							'confirmed' : confirmed,
					})

				else:

					confirmed = False

					denied_msg = "Wrong password. Please try again."

					return render(request, 'resetlabels.html', {	
							'collname' : collname,
							'confirmed' : confirmed,
							'denied_msg' : denied_msg,
					})



class Labeling(View):

	def get(self, request, collname,  is_labeled):
				
		random_cluster, top10, last10, current_label, warning = get_randomcluster(collname, is_labeled)

		labellist = get_labels(collname)

		return render(request, 'label.html', {	
				'random_cluster' : random_cluster,
				'top10' : top10,
				'last10' : last10,
				'labellist' : labellist, 
				'collname' : collname,
				'is_labeled': is_labeled,
				'current_label' : current_label,
				'warning' : warning,
		})


	def post(self, request, collname, is_labeled):

			if "labeling" in request.POST:
			
				#Add the label to DB
				input_label = request.POST['label']
				cl_id = request.POST['cl_id']

				model = get_document(collname)

				if(input_label==""):

					model.objects.get(pk=cl_id).update(unset__label = 1)

				else:
	
					model.objects.get(pk=cl_id).update(set__label = str(input_label))
				
				random_cluster, top10, last10, current_label, warning = get_randomcluster(collname, is_labeled)

				labellist = get_labels(collname)

				return render(request, 'label.html', {	
					'random_cluster' : random_cluster,
					'top10' : top10,
					'last10' : last10,
					'labellist' : labellist, 
					'collname' : collname,
					'is_labeled': is_labeled,
					'current_label' : current_label,
					'warning' : warning,
				})



class HowItWorks(View):

	def get(self, request, page):

		if(page=="Introduction"):
			intro = "True"
			tweets = "" 
			length = ""
			current_page = ""
			nextpage = ""
			next_step = ""

		if(page == "Raw_Data"):	
			intro = "False"
			tweets, length = get_step_data("testcl", 500)
			current_page = "Raw Data"
			nextpage = "Eliminate_Retweets"
			next_step = "Eliminate Retweets"			

		elif(page == "Eliminate_Retweets"):
			intro = "False"
			tweets, length = get_step_data("rt_eliminated", 500)
			current_page = "Retweets are Eliminated"
			nextpage = "Remove_Duplicates"
			next_step = "Remove Duplicates"

		elif(page == "Remove_Duplicates"):
			intro = "False"
			tweets, length = get_step_data("duplicates_eliminated", 500)
			current_page = "Duplicate Tweets are Eliminated"
			nextpage = "Cluster_Them"
			next_step = "Cluster Them"

		elif(page == "Cluster_Them"):
			intro = "False"
			tweets, length = get_step_data("genocide_clusters_20151005", 10, "Cluster_Them")
			current_page = "Tweets are Clustered"
			nextpage = "Label_the_Clusters"
			next_step = "Label the Clusters"

		elif(page == "Label_the_Clusters"):

			return HttpResponseRedirect('/datasets')#Home.as_view()(self.request)


		return render(request, 'howitworks.html', {
				'intro':intro,
				'tweets':tweets,
				'length':length,
				'current_page': current_page,
				'nextpage': nextpage,
				'next_step': next_step,
		})




class About(View):

	
	def get(self, request):


		return render(request, 'about.html', {	
		})




