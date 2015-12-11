
import os

import mongoengine

import configparser


HOSTNAME = os.uname()[1]

which_db = "localdb" 


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

config = configparser.ConfigParser()

if(which_db == "localdb"):
	config.read("data/localdb.ini")

elif(which_db == "mongolab_ebasar"):
	config.read("data/ebasar_rel.ini")


SECRET_KEY = config.get('rel_settings', 'secret_key')


if HOSTNAME[:9] == "applejack":		#to work on the server
	DEBUG = False
else:								#to work on local
	DEBUG = True


ALLOWED_HOSTS = ['localhost', '127.0.0.1', 'relevancer.science.ru.nl']


INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'mongoengine.django.mongo_auth',
    'main',
)

MIDDLEWARE_CLASSES = (
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
)

ROOT_URLCONF = 'RelevancerDjango.urls'

WSGI_APPLICATION = 'RelevancerDjango.wsgi.application'



DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.dummy',
    },
}

AUTHENTICATION_BACKENDS = (
    'mongoengine.django.auth.MongoEngineBackend',
)

SESSION_ENGINE = 'mongoengine.django.sessions'
SESSION_SERIALIZER = 'mongoengine.django.sessions.BSONSerializer'


AUTH_USER_MODEL = 'mongo_auth.MongoUser'
MONGOENGINE_USER_DOCUMENT = 'mongoengine.django.auth.User'


db_name = config.get('rel_mongo_db', 'db_name')
db_host = config.get('rel_mongo_db', 'client_host')
db_port = int(config.get('rel_mongo_db', 'client_port'))

if(which_db == "localdb"):
	mongoengine.connect(db_name, host=db_host, port=db_port)

elif(which_db == "mongolab_ebasar"):
	db_uname = config.get('rel_mongo_db', 'user_name')
	db_passwd = config.get('rel_mongo_db', 'passwd')
	mongoengine.connect(db_name, host=db_host, port=db_port, username=db_uname , password=db_passwd)


LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Europe/Amsterdam'

USE_I18N = True

USE_L10N = True

USE_TZ = True


STATIC_URL = '/static/'


STATICFILES_DIRS = (
    os.path.join(BASE_DIR, "static"),
)

TEMPLATE_DIRS = (
    os.path.join(BASE_DIR, "templates"),
)
