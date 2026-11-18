from celery import Celery
from . import settings

app = Celery()
#print(settings)
app.config_from_object(settings)