from .data import SimSiamData, SimSiamKwargs
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from ..smartscope_simsiam import map_embeddings, main, inference, cluster
from ..smartscope_simsiam.arguments import get_args



def load_inferences(file_path: Path) -> pd.DataFrame:

    # Load the parquet file into a DataFrame
    if not file_path.is_file():
        print(f'File {file_path} not found, creating empty dataframe')
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
        print(f'Warning: {len(missing_pks)} targets not found in inference results')
        print(f'Missing targets: {missing_pks}')
        return filtered_targets, missing_pks
    else:
        print('All targets found in inference results')
        return filtered_targets, []



def siam_siam_inference(data):
    validated_data = SimSiamData.model_validate_json(data)
    print(f'Validated data: {validated_data.model_dump()}')
    print('Current directory:', Path('.').resolve())
    print(str(validated_data.scratch_checkpoint_path))
    image_directory = validated_data.image_directory
    # extract_directory = validated_data.extract_directory
    output_directory = validated_data.output_directory
    for directory in [output_directory]:
        if not directory.is_dir():
            print(f'Creating directory {directory}')
            directory.mkdir(parents=True, exist_ok=True)

    ### check if images were already processed with the same checkpoint
    df = load_inferences(validated_data.output_data_file)
    # df.drop(df.index[0], inplace=True)
    df.drop(columns=['assignments', 'umap', 'tsne', 'pca'], inplace=True, errors='ignore')
    print(df.head())
    if not df.empty:
        print(f'Found {len(df)} previous inferences')
        filtered_df, missing_pks = extract_targets_from_inference(df, validated_data.all_target_pks)
        print(f'Found {len(filtered_df)} targets in previous inferences. Missing {len(missing_pks)} targets')
        if len(missing_pks) == 0:
            is_checkpoint_up_to_date = set(filtered_df.checkpoint_path.tolist()) == set([validated_data.checkpoint_path])
            if is_checkpoint_up_to_date:
                print(f'Images already processed with checkpoint \"{validated_data.checkpoint_path}\", skipping')
                return str(validated_data.output_data_file_relative_to_scratch)
        
        #check which images where processing with a different checkpoint
        filtered_df = filtered_df[filtered_df.checkpoint_path != validated_data.checkpoint_path]
        print(f'Filtered {len(filtered_df)} images that were processed with a different checkpoint')

        df = df.drop(filtered_df.index)
        missing_pks = missing_pks | set(filtered_df.index.tolist())
        print(f'Updated missing pks: {missing_pks}, only running inference on these targets')
        temp_dir = validated_data.data_dir / 'tmp'
        if not temp_dir.is_dir():
            print(f'Creating temporary directory {temp_dir}')
            temp_dir.mkdir(parents=True, exist_ok=True)
        #make sure the temp directory is empty
        for file in temp_dir.glob('*'):
            file.unlink()
        #link the missing pks to the temp directory
        for pk in missing_pks:
            image_file = validated_data.image_directory / f'{pk}.jpg'
            if image_file.is_file():
                temp_file = temp_dir / f'{pk}.jpg'
                temp_file.symlink_to(image_file)
            else:
                print(f'Warning: Image file {image_file} not found, skipping')
        image_directory = temp_dir
        #linking the missing pk images to a temp directory
          

    print(f'Starting inference')

    args = SimSiamKwargs(
            config_file= './SmartscopeAI/smartscope_simsiam/example/config/simsiam_smartscope_squares.yaml',
            data_dir= str(image_directory),
            output_dir= str(output_directory),
            checkpoint_path= str(validated_data.scratch_checkpoint_path),
    )
    args = get_args(args)
    hardware = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {hardware}')
    image_files, embeddings =  map_embeddings.main(hardware, args)
    print(len(image_files), len(embeddings))


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
            print(f'Creating directory {directory}')
            directory.mkdir(parents=True, exist_ok=True)

    print(f'Starting training')

    args = SimSiamKwargs(
            config_file= f'./SmartscopeAI/smartscope_simsiam/example/config/simsiam_smartscope_{validated_data.mag_level}s.yaml',
            data_dir= str(validated_data.image_directory),
            output_dir= str(output_directory),
    )
    args = get_args(args)
    hardware = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {hardware}')
    main.main(hardware, args)
    return str(output_directory)
    