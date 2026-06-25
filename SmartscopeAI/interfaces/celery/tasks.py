import json
import requests
import logging
from celery.signals import task_postrun
from .app import app
from . import settings as celery_settings

from ...microservice.main import get_method, run_method
from ...microservice.sim_siam_wrapper import siam_siam_inference, siam_siam_training


logger = logging.getLogger(__name__)

@app.task
def ping():
    return True

@app.task(bind=True) # Gives access to the task instance and task metadata
def find_squares(self, data: str):
    try:
        result = run_method('find_squares', data)
        logger.debug('find_squares returned %s targets', len(result) if result else 0)
        return result
    except Exception: # Log the error along with the current Celery task ID
        logger.exception('find_squares failed (task_id=%s)', self.request.id)
        raise # Re-raise the exception so Celery marks the task as failed


@app.task(bind=True)
def find_holes(self, data: str):
    try:
        result = run_method('find_holes', data)
        logger.debug('find_holes returned %s targets', len(result) if result else 0)
        return result
    except Exception:
        logger.exception('find_holes failed (task_id=%s)', self.request.id)
        raise

@app.task(bind=True)
def sim_siam_data(self, data: str):
    try:
        result = siam_siam_inference(data)
        logger.debug('sim_siam_inference returned %s targets', len(result) if result else 0)
        return result
    except Exception:
        logger.exception('sim_siam_inference failed (task_id=%s)', self.request.id)
        raise

@app.task(bind=True)
def sim_siam_training(self, data: str):
    try:
        result = siam_siam_training(data)
        logger.debug('sim_siam_training returned %s targets', len(result) if result else 0)
        return result
    except Exception:
        logger.exception('sim_siam_training failed (task_id=%s)', self.request.id)
        raise

@task_postrun.connect
def callback_sim_siam_training(sender=None, task_id=None, state=None, retval=None, **kwargs):
    # Check that the signal is coming from the specific task
    if sender != sim_siam_training:
        logger.debug("[%s] Ignoring callback for task %s", task_id, sender.name)
        return  # Ignore other tasks
    
    callback_url = f"{celery_settings}/sim_siam/training_callback/"
    payload = {
        "process_id": task_id,
    }

    try:
        requests.post(callback_url, json=payload)
    except requests.RequestException as e:
        logger.exception("[%s] Callback failed: {e}", task_id) #log the exception