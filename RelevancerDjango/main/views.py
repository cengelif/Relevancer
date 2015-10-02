# Django
from django.shortcuts import render
from django.views.generic import View
from django.core.exceptions import ObjectDoesNotExist

from django.conf import settings

# Python
import random

# DB / models
import mongoengine
from main.models import * #All models
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

		clusters = model.objects(label__exists = True)

		if(clusters):

			random_cluster = random.choice(clusters)

			current_label = random_cluster["label"]

			ctweettuplelist = []
			for cl in random_cluster["ctweettuplelist"]:
				ctweettuplelist.append(cl[2])

			top10 = ctweettuplelist[:10]

			last10 = ctweettuplelist[-10:]

		else:

			warning = "There is not any labeled cluster yet"


	elif(is_labeled == "unlabeled"):

		clusters = model.objects(label__exists = False)

		if(clusters):

			random_cluster = random.choice(clusters)

			ctweettuplelist = []
			for cl in random_cluster["ctweettuplelist"]:
				ctweettuplelist.append(cl[2])

			top10 = ctweettuplelist[:10]

			last10 = ctweettuplelist[-10:]

		else:

			warning = "All clusters are labeled."


	return random_cluster, top10, last10, current_label, warning



def get_labels(collname):

	model = get_document(collname)

	clusters = model.objects(label__exists = True)

	all_labels = []
	for lbl in clusters:
		if(lbl["label"] not in all_labels):
			all_labels.append(lbl["label"])

	num_of_cl = []
	for labl in all_labels:
		num_of_cl.append(len(model.objects(label = labl)))

	labellist = zip(all_labels, num_of_cl)	

	return labellist



def get_collectioninfo():

	colllist_obj = CollectionList.objects.get(pk = "560e57ade4b097feaeb9f951")

	colllist = []
	for colls in colllist_obj["collectionlist"]:
		colllist.append(colls)

	len_coll = []
	len_unlabeled = []
	len_labeled= []
	for coll in colllist:
		model = get_document(coll)
		len_unlbld = len(model.objects(label__exists = False))
		len_lbld =len(model.objects(label__exists = True))
	
		len_coll.append(len_unlbld + len_lbld)	
		len_unlabeled.append(len_unlbld)
		len_labeled.append(len_lbld)


	collectionlist = zip(colllist, len_coll, len_unlabeled, len_labeled)

	return collectionlist, colllist_obj



############################## VIEWS #############################


class Home(View):

	def get(self, request):
				
		collectionlist, colllist_obj = get_collectioninfo()

		return render(request, 'base.html', {	
				'collectionlist' : collectionlist,
		})


	def post(self, request):

			if "addcollection" in request.POST:				

				newcollection = request.POST['newcollection']

				collectionlist, colllist_obj = get_collectioninfo()

				collectionlist.append(newcollection)

				colllist_obj.update(set__collectionlist = collectionlist)


				with open("main/models.py", "a") as myfile:
   						myfile.write("\nclass " + newcollection + "(Clusters):\n\n\t meta = {'collection': '" + newcollection + "'}")


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
				

