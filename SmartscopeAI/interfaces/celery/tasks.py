import json
from .app import app
from ...microservice.main import get_method, run_method
from ...microservice.sim_siam_wrapper import siam_siam_inference, siam_siam_training


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