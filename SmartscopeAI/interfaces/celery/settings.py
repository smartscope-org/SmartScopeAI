import os

REDIS_HOST = os.getenv('REDIS_HOST', "cache")
REDIS_PORT = os.getenv('REDIS_PORT', "6379")
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
CALLBACK_SITE = os.getenv('CALLBACK_SITE', "http://localhost:8000") 

if REDIS_PASSWORD is not None:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"
else:
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"

accept_content = ['json']
result_accept_content = ['json']
result_backend = f"{REDIS_URL}/1"
broker_url = f"{REDIS_URL}/0"

task_track_started = True

task_default_queue = os.getenv('WORKER_QUEUES', 'celery')
# tasks_routes = {
#     'smartscope.core.tasks.*': {'queue': 'smartscope'},
#     'Smartscope.finders.tasks.*': {'queue': 'finders'},
#     'Smartscope.tasks.*': 'default'
# }

include = ['SmartscopeAI.interfaces.celery.tasks']

