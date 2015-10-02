from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

from main.views import *

urlpatterns = patterns('',

	url(r'^$', Home.as_view(), name='home'),

	url(r'^cluster/(?P<collname>\w+)$', ClusterView.as_view(), name='cluster'),

)
