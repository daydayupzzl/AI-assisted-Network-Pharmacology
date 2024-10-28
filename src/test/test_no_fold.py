import argparse
import torch
import logging
from torch_geometric.data import DataLoader
from src.tools.emetrics import get_cindex, get_rm2, get_ci, get_mse, get_rmse, get_pearson, get_spearman
from src.tools.dataset_test import create_dataset_for_test
from src.tools.utils import *
import time
from src.models.GraphkmerDTA.GraphkmerDTA import GraphkmerDTA as GNNNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

parser = argparse.ArgumentParser(description='Test a trained model for drug-target interaction prediction')
parser.add_argument('dataset_index', type=int, choices=[0, 1], help='Index for dataset: 0 for davis, 1 for kiba')
parser.add_argument('version', type=str, help='Version identifier for the model')
parser.add_argument('cuda_index', type=int, choices=[0, 1, 2, 3], help='Index for CUDA device (0-3)')
parser.add_argument('--test_batch_size', type=int, default=512, help='Batch size for testing')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
args = parser.parse_args()

dataset = ['davis', 'kiba'][args.dataset_index]
version = args.version
cuda_name = f'cuda:{args.cuda_index}'

logger.info(f'Testing dataset: {dataset}')
logger.info(f'CUDA name: {cuda_name}')
logger.info(f'Version: {version}')

device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')


model_st = GNNNet.__name__
model_file_name = f'weights/nofold/{dataset}/v_{version}_{model_st}_{dataset}.model'
result_file_name = f'results/nofold/{dataset}/v_{version}_test_on_{dataset}.txt'
model = GNNNet().to(device)
model.load_state_dict(torch.load(model_file_name, map_location=cuda_name), strict=False)


test_data = create_dataset_for_test(dataset, model_st)


test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    logger.info(f'Make prediction for {len(loader.dataset)} samples...')
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    logger.info('Prediction done!')
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def calculate_metrics(Y, P, result_file_name, dataset='davis'):
    cindex = get_cindex(Y, P)
    cindex2 = get_ci(Y, P)
    rm2 = get_rm2(Y, P)
    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)
    rmse = get_rmse(Y, P)

    logger.info(f'Metrics for {dataset}:')
    logger.info(f'C-index: {cindex}')
    logger.info(f'C-index2: {cindex2}')
    logger.info(f'RM2: {rm2}')
    logger.info(f'MSE: {mse}')
    logger.info(f'Pearson: {pearson}')
    logger.info(f'Spearman: {spearman}')
    logger.info(f'RMSE: {rmse}')

    result_str = f'{dataset}\nrmse: {rmse} mse: {mse} pearson: {pearson} spearman: {spearman} ci: {cindex} rm2: {rm2}'
    with open(result_file_name, 'w') as f:
        f.write(result_str)


if __name__ == '__main__':
    start_time = time.time()

    Y, P = predicting(model, device, test_loader)
    logger.info('Test done, begin evaluation')

    calculate_metrics(Y, P, result_file_name, dataset)

    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60
    logger.info(f'Total execution time: {elapsed_time_minutes:.2f} minutes')