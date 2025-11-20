import json
import requests
from celery.signals import task_postrun
from .app import app
from . import settings as celery_settings

from ...microservice.main import get_method, run_method
from ...microservice.sim_siam_wrapper import siam_siam_inference, siam_siam_training


@app.task
def ping():
    return True

@app.task
def find_squares(data: str):
    result = run_method('find_squares',data)
    print(result)
    return result

@app.task
def find_holes(data: str):
    result = run_method('find_holes',data)
    print(result)
    return result

@app.task
def sim_siam_data(data: str):
    result = siam_siam_inference(data)
    print(result)
    return result

@app.task
def sim_siam_training(data: str):
    result = siam_siam_training(data)
    print(result)
    return result


@task_postrun.connect
def callback_sim_siam_training(sender=None, task_id=None, state=None, retval=None, **kwargs):
    # Check that the signal is coming from the specific task
    if sender != sim_siam_training:
        print(f"[{task_id}] Ignoring callback for task {sender.name}")
        return  # Ignore other tasks
    
    callback_url = f"{celery_settings}/sim_siam/training_callback/"
    payload = {
        "process_id": task_id,
    }

    try:
        requests.post(callback_url, json=payload)
    except requests.RequestException as e:
        print(f"[{task_id}] Callback failed: {e}")