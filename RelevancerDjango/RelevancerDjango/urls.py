from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

from main.views import *

urlpatterns = patterns('',

	url(r'^$', Home.as_view(), name='home'),

	url(r'^how_does_it_work/step:(?P<page>\w+)$', HowItWorks.as_view(), name='howdoesitwork'),

	url(r'^datasets$', Datasets.as_view(), name='datasets'),

	url(r'^datasets/(?P<collname>\w+)/backup$', Backup.as_view(), name='backup'),

	url(r'^datasets/(?P<collname>\w+)/listbackups$', ListBackups.as_view(), name='listbackups'),

	url(r'^datasets/(?P<filename>.*)/loadback$', LoadBack.as_view(), name='loadback'),

	url(r'^datasets/(?P<collname>\w+)/reset_labels$', ResetLabels.as_view(), name='resetlabels'),

	url(r'^datasets/(?P<collname>\w+)/labeled:(?P<is_labeled>\w+)$', Labeling.as_view(), name='labeling'),

	url(r'^about$', About.as_view(), name='about')

)
