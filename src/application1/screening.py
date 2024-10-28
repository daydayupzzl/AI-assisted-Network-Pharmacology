# This script is used for drug repurposing using a trained GraphkmerDTA model.
# It takes in a dataset, CUDA device information, and version identifier as inputs.
# The script reads drug and target information from provided paths, loads a pre-trained model,
# and then performs repurposing for each target to predict possible drug-target interactions.
# The results are saved in Excel format for each target. Logging is used throughout to provide
# information about the execution process, including device details, model loading, and repurposing progress.

import argparse
import os
import sys

import numpy as np
import torch
from src.models.GraphkmerDTA.GraphkmerDTA import GraphkmerDTA as GNNNet
import time
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.tools.repurpose_tool import repurpose

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

parser = argparse.ArgumentParser(description='Repurposing using trained model')
parser.add_argument('traindata_index', type=int, choices=[0, 1], help='Index for train data: 0 for davis, 1 for kiba')
parser.add_argument('dataset_name', type=str, help='Name of the dataset for repurposing')
parser.add_argument('cuda_index', type=int, choices=[0, 1, 2, 3, 4], help='Index for CUDA device (0-3, 4 for CPU)')
parser.add_argument('version', type=str, help='Version identifier for the model')
parser.add_argument('--drug_library_path', type=str, default='case_study/{}/ligands.txt', help='Path to the drug library file')
parser.add_argument('--target_library_path', type=str, default='case_study/{}/receptor.txt', help='Path to the target library file')
parser.add_argument('--output_path', type=str, default='case_study/{}/', help='Path to save repurposing results')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for parallel processing')
args = parser.parse_args()

traindata = ['davis', 'kiba'][args.traindata_index]
dataset = args.dataset_name
cuda_name = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'][args.cuda_index]
version = args.version

available_gpus = torch.cuda.device_count()
if args.cuda_index < available_gpus:
    device = torch.device(f'cuda:{args.cuda_index}')
else:
    logger.warning(f'CUDA device index {args.cuda_index} is out of range. Using CPU instead.')
    device = torch.device('cpu')
logger.info(f'Using device: {device}')

model_file_name = f'weights/nofold/{traindata}/model_km2_dp02_v{version}_GNNNet_{traindata}.model'
model = GNNNet().to(device)
try:
    model.load_state_dict(torch.load(model_file_name, map_location=device), strict=False)
    logger.info(f'Model loaded successfully from {model_file_name}')
    logger.info(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')
except Exception as e:
    logger.error(f'Failed to load model from {model_file_name}: {e}')
    sys.exit(1)

def read_file_library(path, is_drug=True):
    try:
        with open(path, 'r') as file:
            lines = file.readlines()
            if is_drug:
                drug_names, drug_smiles = zip(*(line.strip().split() for line in lines if len(line.strip().split()) == 2))
                return np.array(drug_smiles), np.array(drug_names)
            else:
                target_names, target_seqs = zip(*(line.strip().split() for line in lines if len(line.strip().split()) == 2))
                return np.array(target_seqs), np.array(target_names)
    except FileNotFoundError:
        logger.error(f'File Not Found: {path}, please double check')
        raise
    except ValueError:
        logger.error(f'Unexpected format in file: {path}, please ensure each line contains exactly two elements')
        raise

X_drug_seqs, X_drug_names = read_file_library(args.drug_library_path.format(dataset))
Target_seq_array, Target_name_array = read_file_library(args.target_library_path.format(dataset), is_drug=False)

output_directory = args.output_path.format(dataset) + f'{traindata}_{model.__class__.__name__}_{version}/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

start_time = time.time()

def repurpose_target(target_idx):
    target_name = Target_name_array[target_idx]
    logger.info(f'Repurposing target {target_name} (index {target_idx})...')
    file_path = f'{output_directory}repurpose_{traindata}_{model.__class__.__name__}_{target_name}.xlsx'

    if not os.path.exists(file_path):
        repurpose(X_repurpose=X_drug_seqs, target=Target_seq_array[target_idx], drug_names=X_drug_names,
                  target_name=target_name, model=model, device=device,
                  dataset=dataset, model_st=model.__class__.__name__, traindata=traindata, version=version)
        logger.info(f'Repurposing for target {target_name} completed and saved to {file_path}')
    else:
        logger.info(f'File {file_path} already exists, skipping repurposing for target {target_name}')

with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    list(tqdm(executor.map(repurpose_target, range(len(Target_name_array))), total=len(Target_name_array), desc="Repurposing targets"))

logger.info('Repurposing all targets completed!')
end_time = time.time()
elapsed_time_minutes = (end_time - start_time) / 60
logger.info(f'Total execution time: {elapsed_time_minutes:.2f} minutes')