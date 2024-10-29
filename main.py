from models import DSANIB
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataProcess import DTIDataset
from torch.utils.data import DataLoader
from train import Training
import torch
import argparse
import warnings, os
import pandas as pd
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DSANIB for DTI prediction")
parser.add_argument('--cfg', default='configs/DSANIB.yaml', type=str)
parser.add_argument('--data', default='human', type=str, metavar='TASK', choices=['bindingdb', 'biosnap', 'human'])
parser.add_argument('--split', default='random', type=str, metavar='S')
args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SETUP.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.PATH)
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = DTIDataset(df_train.index.values, df_train)
    val_dataset = DTIDataset(df_val.index.values, df_val)
    test_dataset = DTIDataset(df_test.index.values, df_test)

    params = {'batch_size': cfg.SETUP.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SETUP.NUM_WORKERS, 'drop_last': True, 'collate_fn': graph_collate_func}

    train_data = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False

    val_data = DataLoader(val_dataset, **params)
    test_data = DataLoader(test_dataset, **params)
    model = DSANIB(**cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SETUP.LR)
    torch.backends.cudnn.benchmark = True

    trainer = Training(model, opt, device, train_data, val_data, test_data, opt_da=None,
                          discriminator=None, **cfg)
    result = trainer.train()

    print()
    print(f"Directory for saving result: {cfg.RESULT.PATH}")

    return result

if __name__ == '__main__':
    torch.cuda.empty_cache()
    result = main()
