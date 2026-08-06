from .data import SimSiamData, SimSiamKwargs
import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List
from ..smartscope_simsiam import map_embeddings, main, inference, cluster
from ..smartscope_simsiam.arguments import get_args

logger = logging.getLogger(__name__)


def load_inferences(file_path: Path) -> pd.DataFrame:

    # Load the parquet file into a DataFrame
    if not file_path.is_file():
        logger.warning('File %s not found, creating empty dataframe', file_path)
        df = pd.DataFrame()
        return df
    df = pd.read_parquet(file_path)
    return df

def extract_targets_from_inference(df: pd.DataFrame, all_pks:List[str]) -> pd.DataFrame:

    # Filter the targets list to include only those not in existing_images
    present_mask = df.index.isin(all_pks)
    filtered_targets = df[present_mask]
    all_pks_in_df = set(filtered_targets.index.tolist())
    missing_pks = set(all_pks) - all_pks_in_df
    if len(missing_pks) > 0:
        logger.warning('Warning: %s targets not found in inference results', len(missing_pks))
        logger.warning('Missing targets: %s', missing_pks)
        return filtered_targets, missing_pks
    else:
        logger.info('All targets found in inference results')
        return filtered_targets, []



def siam_siam_inference(data):
    validated_data = SimSiamData.model_validate_json(data)
    logger.info('Validated data: %s', validated_data.model_dump())
    logger.info('Current directory: %s', Path('.').resolve())
    logger.info('Scratch checkpoint path: %s', str(validated_data.scratch_checkpoint_path))
    image_directory = validated_data.image_directory
    # extract_directory = validated_data.extract_directory
    output_directory = validated_data.output_directory
    for directory in [output_directory]:
        if not directory.is_dir():
            logger.info('Creating directory: %s', directory)
            directory.mkdir(parents=True, exist_ok=True)

    ### check if images were already processed with the same checkpoint
    df = load_inferences(validated_data.output_data_file)
    # df.drop(df.index[0], inplace=True)
    df.drop(columns=['assignments', 'umap', 'tsne', 'pca'], inplace=True, errors='ignore')
    logger.info('Inference results:')
    logger.info(df.head())
    if not df.empty:
        logger.info('Found %s previous inferences', len(df))
        filtered_df, missing_pks = extract_targets_from_inference(df, validated_data.all_target_pks)
        logger.info('Found %s targets in previous inferences. Missing %s targets', len(filtered_df), len(missing_pks))
        if len(missing_pks) == 0:
            is_checkpoint_up_to_date = set(filtered_df.checkpoint_path.tolist()) == set([validated_data.checkpoint_path])
            if is_checkpoint_up_to_date:
                logger.info('Images already processed with checkpoint \"%s\", skipping', validated_data.checkpoint_path)
                return str(validated_data.output_data_file_relative_to_scratch)
        
        #check which images where processing with a different checkpoint
        filtered_df = filtered_df[filtered_df.checkpoint_path != validated_data.checkpoint_path]
        logger.info('Filtered %s images that were processed with a different checkpoint', len(filtered_df))

        df = df.drop(filtered_df.index)
        missing_pks = set(missing_pks) | set(filtered_df.index.tolist())
        logger.info('Updated missing pks: %s, only running inference on these targets', missing_pks)
        temp_dir = validated_data.data_dir / 'tmp'
        if not temp_dir.is_dir():
            logger.info('Creating temporary directory: %s', temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        #make sure the temp directory is empty
        for file in temp_dir.glob('*'):
            file.unlink()
        #link the missing pks to the temp directory
        for pk in missing_pks:
            image_file = list(validated_data.image_directory.glob(f'*/{pk}.jpg'))
            if image_file and image_file[0].is_file():
                temp_file = temp_dir / f'{pk}.jpg'
                temp_file.symlink_to(image_file[0])
            else:
                logger.warning('Warning: Image file for %s not found, skipping', pk)
        image_directory = temp_dir
        #linking the missing pk images to a temp directory
          

    logger.info('Starting inference')

    args = SimSiamKwargs(
            config_file= f'./SmartscopeAI/smartscope_simsiam/example/config/simsiam_smartscope_{validated_data.mag_level}s.yaml',
            data_dir= str(image_directory),
            output_dir= str(output_directory),
            checkpoint_path= str(validated_data.scratch_checkpoint_path),
    )
    args = get_args(args)
    hardware = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Using device: %s', hardware)
    image_files, embeddings =  map_embeddings.main(hardware, args)
    logger.info('Found %s image files and %s embeddings', len(image_files), len(embeddings))


    # assignments = cluster.run(args, embeddings)

    data = dict(image_files=image_files)
    inference_df = pd.DataFrame(data)
    inference_df['pk'] = inference_df['image_files'].apply(lambda x: Path(x).stem)
    inference_df['checkpoint_path'] = validated_data.checkpoint_path
    inference_df['embeddings'] = embeddings.tolist()
    inference_df.set_index('pk', inplace=True)
    df = pd.concat([df, inference_df])

    if len(df) > 3:
        all_embeddings = np.array(df['embeddings'].to_list())
        args.disable_plotting = False
        args.data_dir = str(validated_data.image_directory)
        args.embeddings = all_embeddings
        args.fit_only = True  # Set to True to skip plotting
        df['assignments'], df['umap'], df['pca'] = map_embeddings.main(hardware, args)
    else:
        df['assignments'] = 0
        df['umap'] = 0
        df['pca'] = 0



    df.to_parquet(validated_data.output_data_file, compression='gzip')
    return str(validated_data.output_data_file_relative_to_scratch)

def siam_siam_training(data):
    validated_data = SimSiamData.model_validate_json(data)

    # extract_directory = validated_data.extract_directory
    output_directory = validated_data.output_directory
    for directory in [output_directory]:
        if not directory.is_dir():
            logger.info('Creating directory: %s', directory)
            directory.mkdir(parents=True, exist_ok=True)

    logger.info('Starting training')

    args = SimSiamKwargs(
            config_file= f'./SmartscopeAI/smartscope_simsiam/example/config/simsiam_smartscope_{validated_data.mag_level}s.yaml',
            data_dir= str(validated_data.image_directory),
            output_dir= str(output_directory),
    )
    args = get_args(args)
    hardware = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Using device: %s', hardware)
    main.main(hardware, args)
    return str(output_directory)
    
