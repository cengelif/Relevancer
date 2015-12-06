# Django
from django.shortcuts import render
from django.views.generic import View
from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpResponseRedirect
from django.http import HttpRequest  
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
import smtplib # mail
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


HOSTNAME = os.uname()[1]

config = configparser.ConfigParser()

config.read("data/auth.ini")

logging.basicConfig(
		format='%(asctime)s, %(levelname)s: %(message)s',
		filename='data/relevancer.log',
		datefmt='%d-%m-%Y, %H:%M',
		level=logging.INFO)


def send_mail(sbj, msg, to):
	#Mail Auth;
	fromaddr = config.get('mail', 'fromaddr')
	toaddrs  = config.get('mail', to) # opt : hurrial, ebasar
	username = config.get('mail', 'user_name') 
	password = config.get('mail', 'password')
	subject = sbj
	message = 'Subject: ' + subject + '\n\n' + msg
	server = smtplib.SMTP('smtp.gmail.com:587')  
	server.starttls()  
	server.login(username,password)  
	server.sendmail(fromaddr, toaddrs, message)  
	server.quit() 


############################## FUNCTIONS FOR VIEWS ##############################


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

	logging.info('BACKUP : Backup to ' + filename)

	return 0



def loadbackup(filename):

	collname = re.findall('(.*)_\d{6}-\d\d:\d\d.json', filename)[0]

	db_name = config.get('db', 'db_name')

	backup_json(collname)

	client = pymongo.MongoClient("127.0.0.1", 27017)
	localdb = client[db_name]
	local_col = localdb[collname]

	with open("data/backups/" + filename) as f:
		cluster_list = json.load(f)

	for element in cluster_list: 
		del element['_id'] 

	local_col.remove()

	local_col.insert(cluster_list)

	return 0



def get_client_ip(request):
    ip = request.META.get('HTTP_CF_CONNECTING_IP')
    if ip is None:
        ip = request.META.get('REMOTE_ADDR')
    return ip





############################## VIEWS ##############################


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

		action = 'backup'
		event = 'create a new backup file for '

		return render(request, 'confirmpass.html', {	
				'action': action,
				'event' : event,
				'thing' : collname,
		})

	def post(self, request, collname):

			if "confirmpass" in request.POST:				

				user_pass = request.POST['user_pass']

				reset_pass = config.get('dataset', 'reset_pass')

				if(user_pass == reset_pass):

					client_address = get_client_ip(request)

					logging.info('BACKUP : ' + client_address + ' requested to backup ' + collname)

					backup_json(collname)

					logging.info('BACKUP : Backup done with ' + collname + ', by ' + client_address)

					confirmed = True

					result = 'Backup is successfully created for : '

					return render(request, 'confirmpass.html', {	
							'thing' : collname,
							'confirmed' : confirmed,
							'result' : result,
					})

				else:

					confirmed = False
					action = 'backup'
					event = 'create a new backup file for '
					denied_msg = "Wrong password. Please try again."

					return render(request, 'confirmpass.html', {	
							'action': action,
							'event' : event,
							'thing' : filename,
							'confirmed' : confirmed,
							'denied_msg' : denied_msg,
					})



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

		action = 'loadback'
		event = 'load the database backup on'

		return render(request, 'confirmpass.html', {	
				'action': action,
				'event' : event,
				'thing' : filename,
		})


	def post(self, request, filename):

			if "confirmpass" in request.POST:				

				user_pass = request.POST['user_pass']

				reset_pass = config.get('dataset', 'reset_pass')

				if(user_pass == reset_pass):

					client_address = get_client_ip(request)

					logging.info('LOAD : ' + client_address + ' requested to load ' + filename)

					loadbackup(filename)

					logging.info('LOAD : Load done with ' + filename + ', by ' + client_address)

					sbj = "Backup Reload"
					msg = sbj + '\n\nFile loaded : ' + filename + '\nIP address : ' + client_address + "\n\nThis mail sent to : ebasar"

					if HOSTNAME[:9] == "applejack":	 
						msg = msg + ", hurrial"
						send_mail(sbj, msg, "hurrial")
						send_mail(sbj, msg, "ebasar")
					else:
						send_mail(sbj, msg, "ebasar")

					confirmed = True

					result = 'Following backup is successfully loaded back : '

					return render(request, 'confirmpass.html', {	
							'thing' : filename,
							'confirmed' : confirmed,
							'result' : result,
					})

				else:

					confirmed = False
					action = 'loadback'
					event = 'load the database backup on '
					denied_msg = "Wrong password. Please try again."

					return render(request, 'confirmpass.html', {	
							'action': action,
							'event' : event,
							'thing' : filename,
							'confirmed' : confirmed,
							'denied_msg' : denied_msg,
					})




class ResetLabels(View):

	def get(self, request, collname):

		action = 'resetlabels'
		event = 'reset all labels for'

		return render(request, 'confirmpass.html', {	
				'action': action,
				'event' : event,
				'thing' : collname,
		})


	def post(self, request, collname):

			if "confirmpass" in request.POST:				

				user_pass = request.POST['user_pass']

				reset_pass = config.get('dataset', 'reset_pass')

				if(user_pass == reset_pass):

					client_address = get_client_ip(request)

					logging.info('RESET : ' + client_address + ' requested to reset all labels on '   + collname )

					backup_json(collname)
	
					model = get_document(collname)
					
					model.objects.update(unset__label=1)

					logging.info('RESET : Reset done for all labels on ' + collname + ', by ' + client_address)

					sbj = "Reset All Labels"
					msg = sbj + '\n\nCollection : ' + collname + '\nIP address : ' + client_address + "\n\nThis mail sent to : ebasar"

					send_mail(sbj, msg, "ebasar")

					if HOSTNAME[:9] == "applejack":	 
						msg = msg + ", hurrial"
						send_mail(sbj, msg, "hurrial")
						send_mail(sbj, msg, "ebasar")
					else:
						send_mail(sbj, msg, "ebasar")

					confirmed = True

					result = 'All labels are successfully removed from '

					return render(request, 'confirmpass.html', {	
							'thing' : collname,
							'confirmed' : confirmed,
							'result' : result,
					})

				else:

					confirmed = False
					action = 'resetlabels'
					event = 'reset all labels for '
					denied_msg = "Wrong password. Please try again."

					return render(request, 'confirmpass.html', {
							'action': action,	
							'event' : event,
							'thing' : collname,
							'confirmed' : confirmed,
							'denied_msg' : denied_msg,
					})



class Labeling(View):


	def get(self, request, collname, is_labeled):

		action = 'labeling'
		event = 'see the clusters for '

		return render(request, 'confirmpass.html', {	
				'action': action,
				'event' : event,
				'thing' : collname,
				'is_labeled' : is_labeled,
		})


	def post(self, request, collname, is_labeled):

			if "confirmpass" in request.POST:

				user_pass = request.POST['user_pass']

				reset_pass = config.get('dataset', 'reset_pass')

				if(user_pass == reset_pass):

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

				else:

					confirmed = False
					action = "labeling"
					event = 'see the clusters for '
					denied_msg = "Wrong password. Please try again."

					return render(request, 'confirmpass.html', {	
							'action': action,
							'event' : event,
							'thing' : collname,
							'confirmed' : confirmed,
							'denied_msg' : denied_msg,
							'is_labeled' : is_labeled,
					})


			elif "label" in request.POST:

				client_address = get_client_ip(request)
			
				#Add the label to DB
				input_label = request.POST['label']
				cl_id = request.POST['cl_id']

				model = get_document(collname)

				if(input_label==""):

					model.objects.get(pk=cl_id).update(unset__label = 1)

					logging.info('LABEL: Label deleted from ' + cl_id + ', by ' + client_address + ' for ' + collname)

				else:
	
					model.objects.get(pk=cl_id).update(set__label = str(input_label))

					logging.info('LABEL: ' + cl_id + ' labeled as "' + input_label + '", by ' + client_address + ' for ' + collname)
				
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


			elif "directlabel" in request.POST:

				client_address = get_client_ip(request)

				#Add the label to DB
				input_label = request.POST['directlabel']
				cl_id = request.POST['cl_id']

				model = get_document(collname)

				model.objects.get(pk=cl_id).update(set__label = str(input_label))

				logging.info('LABEL: ' + cl_id + ' labeled as "' + input_label + '", by ' + client_address + ' for ' + collname)
				
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
				


			elif "nextcl" in request.POST:

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




