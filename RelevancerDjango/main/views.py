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



def getRandomClusterandLabels(collname):

	## GET RANDOM CLUSTER

	model = get_document(collname)

	unlabeled_clusters = model.objects(label__exists = False)

	if(unlabeled_clusters):
		random_cluster = random.choice(unlabeled_clusters)
		
		ctweettuplelist = []
		for cl in random_cluster["ctweettuplelist"]:
			ctweettuplelist.append(cl[2])

		top10 = ctweettuplelist[:10]

		last10 = ctweettuplelist[-10:]

		warning = ""

	else:
		random_cluster =  None
		top10 = []
		last10 = []

	## GET ALL LABELS FOR A CLUSTER

	label_coll = Labels.objects.get(coll_name = collname)

	all_labels = []
	for lbl in label_coll["all_labels"]:
		all_labels.append(lbl)

	warning = "All clusters labeled in this collection"

	return random_cluster, top10, last10, all_labels, warning


class Home(View):

	def get(self, request):
				
		colllistdb = CollectionList.objects.get(pk = "560e57ade4b097feaeb9f951")

		collectionlist = []
		for colls in colllistdb["collectionlist"]:
			collectionlist.append(colls)

		return render(request, 'base.html', {	
				'collectionlist' : collectionlist,
		})


	def post(self, request):

			if "addcollection" in request.POST:				

				newcollection = request.POST['newcollection']

				colllistdb = CollectionList.objects.get(pk = "560e57ade4b097feaeb9f951")

				collectionlist = []
				for colls in colllistdb["collectionlist"]:
					collectionlist.append(colls)

				collectionlist.append(newcollection)

				colllistdb.update(set__collectionlist = collectionlist)

				with open("main/models.py", "a") as myfile:
   						myfile.write("\nclass " + newcollection + "(Clusters):\n\n\t meta = {'collection': 'all_data_clusters2'}")


				Labels(coll_name = newcollection, all_labels = []).save()

				return render(request, 'base.html', {	
						'collectionlist' : collectionlist,
				})
	



class ClusterView(View):

	def get(self, request, collname):
				
		cluster, top10, last10, all_labels, warning = getRandomClusterandLabels(collname)

		return render(request, 'cluster.html', {	
				'cluster' : cluster,
				'top10' : top10,
				'last10' : last10,
				'all_labels' : all_labels, 
				'warning' : warning,
				'collname' : collname,
		})


	def post(self, request, collname):

			if "labeler" in request.POST:
			
				#Add the label to DB
				input_label = request.POST['label']
				cl_id = request.POST['cl_id']

				model = get_document(collname)

				model.objects.get(pk=cl_id).update(set__label = str(input_label))
				
				# New Cluster to label and labels to update
				new_cluster, new_top10, new_last10, all_labels, warning = getRandomClusterandLabels(collname)

				if (input_label not in all_labels):
					all_labels.append(input_label)

				Labels.objects.get(coll_name = collname).update(set__all_labels = all_labels)

				return render(request, 'cluster.html', {	
					'cluster' : new_cluster,
					'top10' : new_top10,
					'last10' : new_last10,
					'label' : input_label,
					'cl_id' : cl_id,
					'all_labels' : all_labels, 
					'warning' : warning,
					'collname' : collname,
				})



