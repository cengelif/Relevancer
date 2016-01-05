
from django.db import models

from mongoengine import DynamicDocument, ListField, StringField


class CollectionList(DynamicDocument):

	meta = {'collection' : 'CollectionList'}
	collectionlist = ListField(StringField())


class testcl(DynamicDocument): # raw tweet data

	meta = {'collection' : 'testcl'}
	text = StringField()

class rt_eliminated(DynamicDocument): # retweets are eliminated 

	meta = {'collection' : 'rt_eliminated'}
	text = StringField()

class duplicates_eliminated(DynamicDocument): # duplicates are eliminated 

	meta = {'collection' : 'duplicates_eliminated'}
	text = StringField()


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


class genocide_clusters_20151005(Clusters):

	meta = {'collection': 'genocide_clusters_20151005'}

