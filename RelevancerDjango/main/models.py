
from django.db import models

from mongoengine import * #DynamicDocument come from here.


class Clusters(Document):
	"""
	It represents MongoDB structure and inherits the Document Class from MongoEngine.
	"""
	meta = {'collection' : 'all_data_clusters'}
	ctweettuplelist  = ListField(StringField())
	cstr = StringField()
	cno = StringField()
	cnoprefix = StringField()
	rif = ListField(StringField())
	twids = ListField(StringField())
	user_entropy = StringField()
	label = StringField()



class Labels(Document):

	meta = {'collection' : 'labels'}
	all_labels = ListField(StringField())
	coll_name = StringField()
