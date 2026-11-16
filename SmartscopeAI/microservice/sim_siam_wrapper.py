from .data import SimSiamData, SimSiamKwargs
import torch
import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import List
from ..smartscope_simsiam import map_embeddings, main, inference, cluster
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import umap
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

def _as_float32(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def atomic_to_parquet(df: pd.DataFrame, path: Path):
    """Write next to the target then os.replace, so readers never see a partially written file."""
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        # zstd writes several times faster than gzip; float embeddings barely compress either way.
        df.to_parquet(tmp, compression='zstd')
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def extract_targets_from_inference(df: pd.DataFrame, all_pks:List[str]) -> pd.DataFrame:

    # Filter the targets list to include only those not in existing_images
    present_mask = df.index.isin(all_pks)
    filtered_targets = df[present_mask]
    all_pks_in_df = set(filtered_targets.index.tolist())
    missing_pks = set(all_pks) - all_pks_in_df
    # New targets are expected on every run; the caller logs how many.
    if len(missing_pks) > 0:
        return filtered_targets, missing_pks
    return filtered_targets, []



def siam_siam_inference(data):
    validated_data = SimSiamData.model_validate_json(data)
    logger.info('SimSiam inference for %s (%s), checkpoint %s',
                validated_data.dataset_name, validated_data.mag_level, validated_data.scratch_checkpoint_path)
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
    if not df.empty:
        filtered_df, missing_pks = extract_targets_from_inference(df, validated_data.all_target_pks)
        logger.info('%s previous inferences; %s current targets already embedded, %s new',
                    len(df), len(filtered_df), len(missing_pks))
        if len(missing_pks) == 0:
            is_checkpoint_up_to_date = set(filtered_df.checkpoint_path.tolist()) == set([validated_data.checkpoint_path])
            if is_checkpoint_up_to_date:
                logger.info('Images already processed with checkpoint \"%s\", skipping', validated_data.checkpoint_path)
                return str(validated_data.output_data_file_relative_to_scratch)
        
        #check which images where processing with a different checkpoint
        filtered_df = filtered_df[filtered_df.checkpoint_path != validated_data.checkpoint_path]
        if len(filtered_df):
            logger.info('%s targets were embedded with a different checkpoint, redoing them', len(filtered_df))

        df = df.drop(filtered_df.index)
        missing_pks = set(missing_pks) | set(filtered_df.index.tolist())
        logger.info('Running inference on %s targets', len(missing_pks))
        temp_dir = validated_data.data_dir / 'tmp'
        if not temp_dir.is_dir():
            logger.info('Creating temporary directory: %s', temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        #make sure the temp directory is empty
        for file in temp_dir.glob('*'):
            file.unlink()
        #link the missing pks to the temp directory
        # One listing of the image folders rather than a search per target.
        images_by_pk = {f.stem: f for f in validated_data.image_directory.glob('*/*.jpg')}
        not_found = []
        for pk in missing_pks:
            if pk in images_by_pk:
                (temp_dir / f'{pk}.jpg').symlink_to(images_by_pk[pk])
            else:
                not_found.append(pk)
        if not_found:
            logger.warning('%s image files not found, skipping them (first few: %s)', len(not_found), not_found[:5])
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
    # float32 arrays rather than Python lists: half the data, and pyarrow converts arrays
    # without walking every float.
    inference_df['embeddings'] = list(_as_float32(embeddings))
    inference_df.set_index('pk', inplace=True)
    if not df.empty:
        # Rows from files written before this change come back as float64.
        df['embeddings'] = [_as_float32(e) for e in df['embeddings']]
    df = pd.concat([df, inference_df])

    # Clustering / UMAP / PCA are intentionally NOT run here anymore: they scale with the
    # whole grid and are only used for WebUI display. See siam_siam_cluster.

    atomic_to_parquet(df, validated_data.output_data_file)
    return str(validated_data.output_data_file_relative_to_scratch)

def fit_clusters(args, embeddings: np.ndarray):
    """Cluster assignments + 2D UMAP/PCA projections for the given embeddings.
    Same math as map_embeddings.main(fit_only=True) but without building a DataLoader over the images."""
    normed_embeddings = normalize(embeddings, axis=1, norm='l2')
    os.makedirs(os.path.join(args.output_dir, 'inference'), exist_ok=True)
    assignments = cluster.run(args, normed_embeddings)
    pca_fit = PCA(n_components=2).fit_transform(normed_embeddings)
    umap_fit = umap.UMAP(n_neighbors=args.umap['num_neighbors'],
                         min_dist=args.umap['min_dist'],
                         n_components=2,
                         random_state=args.umap['random_state']).fit_transform(normed_embeddings)
    return assignments, umap_fit.tolist(), pca_fit.tolist()


def siam_siam_cluster(data):
    validated_data = SimSiamData.model_validate_json(data)
    input_file = validated_data.cluster_input_file
    if not input_file.is_file():
        input_file = validated_data.output_data_file
    df = load_inferences(input_file)
    if df.empty:
        raise FileNotFoundError(f'No embeddings found at {input_file}')
    logger.info('Clustering %s embeddings from %s', len(df), input_file)

    out = pd.DataFrame(index=df.index)
    if len(df) > 3:
        args = SimSiamKwargs(
            config_file=f'./SmartscopeAI/smartscope_simsiam/example/config/simsiam_smartscope_{validated_data.mag_level}s.yaml',
            data_dir=str(validated_data.image_directory),
            output_dir=str(validated_data.output_directory),
        )
        args = get_args(args)
        all_embeddings = np.array(df['embeddings'].to_list())
        out['assignments'], out['umap'], out['pca'] = fit_clusters(args, all_embeddings)
    else:
        out['assignments'] = 0
        out['umap'] = 0
        out['pca'] = 0

    atomic_to_parquet(out, validated_data.output_cluster_file)
    return str(validated_data.output_cluster_file_relative_to_scratch)


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
    
