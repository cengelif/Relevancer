
from django.shortcuts import render
from django.views.generic import View
from django.core.exceptions import ObjectDoesNotExist

from django.conf import settings

import mongoengine

import random

from main.models import Clusters, Labels


def getRandomClusterandLabels():

	## GET RANDOM CLUSTER

	unlabeled_clusters = Clusters.objects(label__exists = False)

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

	label_coll = Labels.objects.get(coll_name = 'all_data_clusters')

	all_labels = []
	for lbl in label_coll["all_labels"]:
		all_labels.append(lbl)

	warning = "All clusters labeled in this collection"

	return random_cluster, top10, last10, all_labels, warning


class Home(View):

	def get(self, request):
				
		

		return render(request, 'base.html', {	

		})
	



class ClusterView(View):

	def get(self, request):
				
		cluster, top10, last10, all_labels, warning = getRandomClusterandLabels()

		return render(request, 'cluster.html', {	
				'cluster' : cluster,
				'top10' : top10,
				'last10' : last10,
				'all_labels' : all_labels, 
				'warning' : warning,
		})


	def post(self, request):

			if "labeler" in request.POST:
			
				#Add the label to DB
				input_label = request.POST['label']
				cl_id = request.POST['cl_id']

				Clusters.objects.get(pk=cl_id).update(set__label = str(input_label))
				
				# New Cluster to label and labels to update
				new_cluster, new_top10, new_last10, all_labels, warning = getRandomClusterandLabels()

				if (input_label not in all_labels):
					all_labels.append(input_label)

				Labels.objects.get(coll_name = "all_data_clusters").update(set__all_labels = all_labels)

				return render(request, 'cluster.html', {	
					'cluster' : new_cluster,
					'top10' : new_top10,
					'last10' : new_last10,
					'label' : input_label,
					'cl_id' : cl_id,
					'all_labels' : all_labels, 
					'warning' : warning,
				})














