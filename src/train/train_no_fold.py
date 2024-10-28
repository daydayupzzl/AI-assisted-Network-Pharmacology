import argparse
import torch
import torch.nn as nn
import logging
from torch.optim import lr_scheduler

from src.tools.emetrics import get_mse
from src.models.GraphkmerDTA.GraphkmerDTA import GraphkmerDTA as GNNNet
import time
from src.tools.utils import *
import datetime

from src.tools.dataset_nofolds import create_dataset_for_NoFolds

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

parser = argparse.ArgumentParser(description='Train a model for drug-target interaction prediction')
parser.add_argument('dataset_index', type=int, choices=[0, 1], help='Index for dataset: 0 for davis, 1 for kiba')
parser.add_argument('version', type=str, help='Version identifier for the model')
parser.add_argument('cuda_index', type=int, help='Index for CUDA device (0, 1, 2, 3)')
parser.add_argument('--save_path', type=str, default='weights/nofold/', help='Path to save model weights')
parser.add_argument('--save_interval', type=int, default=50, help='Interval (in epochs) to save model checkpoints')
parser.add_argument('--train_batch_size', type=int, default=512, help='Training batch size')
parser.add_argument('--test_batch_size', type=int, default=512, help='Testing batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--num_epochs', type=int, default=2000, help='Number of training epochs')
parser.add_argument('--patience', type=int, default=300, help='Patience for early stopping')
parser.add_argument('--scheduler_step_size', type=int, default=500, help='Step size for the learning rate scheduler')
parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Gamma for the learning rate scheduler')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
args = parser.parse_args()

datasets = [['davis', 'kiba'][args.dataset_index]]
version = args.version
cuda_name = f'cuda:{args.cuda_index}'

device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')

logger.info(f'Learning rate: {args.learning_rate}')
logger.info(f'Epochs: {args.num_epochs}')
logger.info(f'cuda_name: {cuda_name}')
logger.info(f'device: {device}')

model = GNNNet().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

for dataset in datasets:
    train_data, test_data = create_dataset_for_NoFolds(dataset, model.__class__.__name__)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True,
                                               collate_fn=collate, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False,
                                              collate_fn=collate, num_workers=args.num_workers)

    no_improve_count = 0
    start_time = time.time()
    best_mse = float('inf')
    best_epoch = -1
    model_file_name = f'{args.save_path}/{dataset}/v_{version}_{model.__class__.__name__}_{dataset}.model'

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        train(model, device, train_loader, optimizer, epoch + 1)
        logger.info('Predicting for test data')
        G, P = predicting(model, device, test_loader)
        ret = get_mse(G, P)

        if ret < best_mse:
            best_mse = ret
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            logger.info(f'MSE improved at epoch {best_epoch}; best_test_mse: {best_mse}, {dataset}')
            no_improve_count = 0
        else:
            no_improve_count += 1
            logger.info(f'No improvement since epoch {best_epoch}; best_test_mse: {best_mse}, {dataset}')

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_file_name = f'{args.save_path}/{dataset}/v_{version}_{model.__class__.__name__}_{dataset}_epoch_{epoch + 1}.model'
            torch.save(model.state_dict(), checkpoint_file_name)
            logger.info(f'Saved model checkpoint at epoch {epoch + 1}')

        scheduler.step()

        if no_improve_count >= args.patience:
            logger.info(f'Early stopping at epoch {epoch + 1} due to no improvement in the last {args.patience} epochs.')
            break

        epoch_end_time = time.time()
        epoch_elapsed_time = datetime.timedelta(seconds=(epoch_end_time - epoch_start_time))
        logger.info(f'Epoch {epoch + 1} completed in {epoch_elapsed_time}')

    end_time = time.time()
    elapsed_time = datetime.timedelta(seconds=(end_time - start_time))
    logger.info(f'Total execution time: {elapsed_time}')