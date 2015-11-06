from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

from main.views import *

urlpatterns = patterns('',

	url(r'^$', Home.as_view(), name='home'),

	url(r'^datasets/(?P<collname>\w+)/labeled:(?P<is_labeled>\w+)$', LabelView.as_view(), name='label'),

	url(r'^datasets$', Datasets.as_view(), name='datasets'),

	url(r'^how_it_works/step:(?P<page>\w+)$', HowItWorks.as_view(), name='howitworks'),

	url(r'^about$', About.as_view(), name='about')

)
