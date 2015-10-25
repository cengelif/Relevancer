from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

from main.views import *

urlpatterns = patterns('',

	url(r'^$', Home.as_view(), name='home'),

	url(r'^(?P<collname>\w+)/(?P<is_labeled>\w+)$', ClusterView.as_view(), name='cluster'),

	url(r'^clustering$', Clustering.as_view(), name='clustering'),

	url(r'^how_it_works/step:(?P<page>\w+)$', HowItWorks.as_view(), name='howitworks'),

	url(r'^about$', About.as_view(), name='about')

)
