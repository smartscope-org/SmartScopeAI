import os

REDIS_HOST = os.getenv('REDIS_HOST', "cache")
REDIS_PORT = os.getenv('REDIS_PORT', "6379")

accept_content = ['json']
result_accept_content = ['json']
result_backend = f"redis://{REDIS_HOST}:{REDIS_PORT}/1"
broker_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# tasks_routes = {
#     'smartscope.core.tasks.*': {'queue': 'smartscope'},
#     'Smartscope.finders.tasks.*': {'queue': 'finders'},
#     'Smartscope.tasks.*': 'default'
# }

include = ['SmartscopeAI.interfaces.celery.tasks']