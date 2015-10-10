# Django
from django.shortcuts import render
from django.views.generic import View
from django.core.exceptions import ObjectDoesNotExist

from django.conf import settings

# Python
import random
import time

# DB / models
import mongoengine
from main.models import * # Clusters, CollectionList (Have to import everything(with star) because the models can be added dynamically)
from mongoengine.base.common import get_document


############################## FUNCTIONS #############################


def get_randomcluster(collname, is_labeled):

	random_cluster =  None
	current_label = ""
	warning = ""
	top10 = []
	last10 = []

	model = get_document(collname)

	if(is_labeled == "labeled"):

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


	elif(is_labeled == "unlabeled"):

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



############################## VIEWS #############################


class Home(View):

	def get(self, request):
				
		collectionlist = get_collectionlist("info")

		return render(request, 'base.html', {	
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

				return render(request, 'base.html', {	
						'collectionlist' : collectionlist,
				})
	



class ClusterView(View):

	def get(self, request, collname, is_labeled):
				
		random_cluster, top10, last10, current_label, warning = get_randomcluster(collname, is_labeled)

		labellist = get_labels(collname)

		return render(request, 'cluster.html', {	
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

			if "addlabel" in request.POST:
			
				#Add the label to DB
				input_label = request.POST['label']
				cl_id = request.POST['cl_id']

				model = get_document(collname)

				model.objects.get(pk=cl_id).update(set__label = str(input_label))
				
				random_cluster, top10, last10, current_label, warning = get_randomcluster(collname, is_labeled)

				labellist = get_labels(collname)

				return render(request, 'cluster.html', {	
					'random_cluster' : random_cluster,
					'top10' : top10,
					'last10' : last10,
					'labellist' : labellist, 
					'collname' : collname,
					'is_labeled': is_labeled,
					'current_label' : current_label,
					'warning' : warning,
				})
				

