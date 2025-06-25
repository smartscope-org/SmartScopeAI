from .data import SimSiamData, SimSiamKwargs
import pandas as pd
from pathlib import Path
from typing import List
from ..smartscope_simsiam import map_embeddings, main
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

    # extract_directory = validated_data.extract_directory
    output_directory = validated_data.output_directory
    for directory in [output_directory]:
        if not directory.is_dir():
            print(f'Creating directory {directory}')
            directory.mkdir(parents=True, exist_ok=True)

    ### check if images were already processed with the same checkpoint
    df = load_inferences(validated_data.output_data_file)
    if df.empty:
        print(f'No previous inferences found, starting fresh')
    else:
        print(f'Found {len(df)} previous inferences')
        filtered_df, missing_pks = extract_targets_from_inference(df, validated_data.all_target_pks)
        if len(missing_pks) == 0:
            is_checkpoint_up_to_date = set(filtered_df.checkpoint_path.tolist()) == set([validated_data.checkpoint_path])
            if is_checkpoint_up_to_date:
                print(f'Images already processed with checkpoint \"{validated_data.checkpoint_path}\", skipping')
                return str(validated_data.output_data_file)
        df = df.drop(filtered_df.index)
         
    
    print(f'Starting inference')

    args = SimSiamKwargs(
            config_file= '/mnt/smartscope/jo-dev/ai_microservice/SmartscopeAI/smartscope_simsiam/example/config/simsiam_smartscope_squares.yaml',
            data_dir= str(validated_data.image_directory),
            output_dir= str(output_directory),
            checkpoint_path= str(validated_data.scratch_checkpoint_path),
    )
    args = get_args(args)
    image_files, embeddings, assignments =  map_embeddings.main('cuda', args)
    print(len(image_files), len(embeddings), len(assignments))

    data = dict(image_files=image_files, assignments=assignments)
    inference_df = pd.DataFrame(data)
    inference_df['pk'] = inference_df['image_files'].apply(lambda x: Path(x).stem)
    inference_df['checkpoint_path'] = validated_data.checkpoint_path
    inference_df['embeddings'] = embeddings.tolist()
    inference_df.set_index('pk', inplace=True)
    df = pd.concat([df, inference_df])
    df.to_parquet(validated_data.output_data_file, compression='gzip')
    return str(validated_data.output_data_file)

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
            config_file= f'/mnt/smartscope/jo-dev/ai_microservice/SmartscopeAI/smartscope_simsiam/example/config/simsiam_smartscope_{validated_data.mag_level}s.yaml',
            data_dir= str(validated_data.data_dir),
            output_dir= str(output_directory),
            # checkpoint_path= validated_data.checkpoint_path,
    )
    args = get_args(args)
    main.main('cuda', args)
    return str(output_directory)
    