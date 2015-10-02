
from django.db import models

from mongoengine import *


class Labels(DynamicDocument):

	meta = {'collection' : 'labels'}
	all_labels = ListField(StringField())
	coll_name = StringField()


class CollectionList(DynamicDocument):

	meta = {'collection' : 'CollectionList'}
	collectionlist = ListField(StringField())


class Clusters(DynamicDocument):

	meta = {'abstract': True,}

	ctweettuplelist  = ListField(StringField())
	cstr = StringField()
	cno = StringField()
	cnoprefix = StringField()
	rif = ListField(StringField())
	twids = ListField(StringField())
	user_entropy = StringField()
	label = StringField()


class all_data_clusters(Clusters):

	meta = {'collection': 'all_data_clusters'}


class all_data_clusters2(Clusters):

	meta = {'collection': 'all_data_clusters2'}
