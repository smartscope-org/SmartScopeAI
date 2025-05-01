from typing import Callable, Dict, Optional
from .wrapper import find_holes_from_image, find_squares_from_image
from pydantic import BaseModel, Base64Bytes

class FindSquareKwargs(BaseModel):
    imgsz: int = 2048
    thresh: float = 0.2
    iou: float = 0.3
    weights: str = 'square_weights/model_large_atlas.pth'

class FindHolesKwargs(BaseModel):
    imgsz: int = 1280
    conf_thres: float = 0.7
    iou_thres: float = 0.7
    weights_circle: str = 'circle_weights/20250129_best.pt'
    scaling_factor: float =  1
    class_mapping: Optional[Dict]
    


class FindSquaresRequest(BaseModel):
    image: Base64Bytes
    kwargs: FindHolesKwargs
    method: Callable= find_squares_from_image

class FindHolesRequest(BaseModel):
    image: Base64Bytes
    kwargs: FindHolesKwargs
    method: Callable= find_holes_from_image


METHODS_MAPPING = {
    'find_squares': FindSquaresRequest,
    'find_holes': FindHolesRequest
}



