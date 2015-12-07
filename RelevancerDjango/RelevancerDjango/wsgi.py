
import os
import sys

sys.path.append('/scratch2/www/Relevancer/RelevancerDjango/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "RelevancerDjango.settings")

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
