from typing import Callable, Dict, Optional, List
from pathlib import Path
import os
from .wrapper import find_holes_from_image, find_squares_from_image
from pydantic import BaseModel, Base64Bytes, model_validator


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
    
class SimSiamTargets(BaseModel):
    x: int
    y: int
    pk: str

    @property
    def image(self):
        return Path(self.pk).with_suffix('.jpg')
    

class SimSiamData(BaseModel):
    dataset_name: str
    mag_level: str
    checkpoint_path: Optional[str]

    # @model_validator(mode='before')
    # def checkpoint_path_validator(cls,data):
    #     if not 'checkpoint_path' in data or data['checkpoint_path'] is None:
    #         data['checkpoint_path'] = f"/mnt/smartscope/jo-dev/ai_microservice/SmartscopeAI/smartscope_simsiam/example/pretrained_checkpoints/{data['mag_level']}s/model_best.pth"
    #     return data


    @property
    def all_target_pks(self):
        return [target.stem for target in self.data_dir.glob(f'*.jpg')]

    @property
    def data_dir(self):
        scratch_dir = os.getenv('SCRATCH_DIR', '/mnt/scratch')
        return Path(scratch_dir, self.dataset_name)
    
    @property
    def image_directory(self):
        return self.data_dir / 'images'

    @property
    def output_directory(self):
        return self.data_dir / 'output'
    
    @property
    def scratch_checkpoint_path(self) -> Path:
        if self.checkpoint_path is None:
            return Path('/mnt/smartscope/jo-dev/ai_microservice/SmartscopeAI/smartscope_simsiam/example/pretrained_checkpoints', f'{self.mag_level}s/model_best.pth')
        return self.output_directory / 'checkpoints' / 'model_best.pth'
    
    @property
    def output_data_file(self) -> Path:
        return Path(self.output_directory / f'sim_siam_embeddings_{self.mag_level}.parquet.zip',)
    
class SimSiamKwargs(BaseModel):
    config_file: str
    data_dir: str
    output_dir: str
    checkpoint_path: Optional[str] = None
    debug: bool = False
    debug_subset_size: int = 8
    download: bool = False
    log_dir: Optional[str] = None
    ckpt_dir: Optional[str] = None
    eval_from: Optional[str] = None
    hide_progress: bool = False

# class SimSiamSuggestSimilarKwargs(BaseModel):
#     grid_directory: Path
#     similar_to_ids: List[str]
#     mag_level: str

#     @property
#     def embeddings_file(self):
#         return self.grid_directory / f'sim_siam_embeddings_{self.mag_level}.parquet.zip'


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
    'find_holes': FindHolesRequest,
}



