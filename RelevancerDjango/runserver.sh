#!/bin/bash

ROOTDIR=/scratch2/www/Relevancer/RelevancerDjango/

. /scratch2/www/Relevancer/relenv/bin/activate

#python3 manage.py runserver 3029
uwsgi --virtualenv $VIRTUAL_ENV --socket 127.0.0.1:3029 --chdir $ROOTDIR --wsgi-file $ROOTDIR/RelevancerDjango/wsgi.py --logto $ROOTDIR/relevancerdjango.uwsgi.log --log-date --log-5xx --master --processes 4 --threads 2 --need-app --pidfile ./relevancer.pid
