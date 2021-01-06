import os
import sys
import argparse

import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils
import defaults.EPIC_JPOSE as EPIC_JPOSE

from models.jpose import JPOSE
from datasets.jpose_dataset import create_epic_jpose_dataset
from evaluation import nDCG
from evaluation import mAP
from evaluation import average_rank
from evaluation import recall_at_k
from train.train_mmen_triplet import initialise_nDCG_values
from train.train_jpose_triplet import initialise_jpose_nDCG_values, create_modality_dicts

def main(args):
    arg_file_path = args.MODEL_PATH.rsplit('/', 2)[0] + '/args.txt'
    model_args = utils.output.load_args(arg_file_path)

    print(model_args)

    if not hasattr(model_args, 'num_triplets'):
        setattr(model_args, 'num_triplets', 0)
    test_ds = create_epic_jpose_dataset(is_train=False, batch_size=model_args.batch_size, num_triplets=model_args.num_triplets)
    test_ds = initialise_jpose_nDCG_values(test_ds)

    vis_nDCG = []
    txt_nDCG = []
    vis_mAP = []
    txt_mAP = []
    for i in tqdm(range(10)):
        rand_sim = np.random.randn(test_ds.x_len, test_ds.y_len)
        vis_nDCG.append(nDCG.calculate_nDCG(rand_sim,
                test_ds['action'].relevancy_matrix, test_ds['action'].k_values['v'],
                IDCG=test_ds['action'].IDCG['v']))
        txt_nDCG.append(nDCG.calculate_nDCG(rand_sim.T,
                test_ds['action'].relevancy_matrix.T, test_ds['action'].k_values['t'],
                IDCG=test_ds['action'].IDCG['t']))
        vis_mAP.append(mAP.calculate_mAP(rand_sim, test_ds['action'].relevancy_matrix))
        txt_mAP.append(mAP.calculate_mAP(rand_sim.T, test_ds['action'].relevancy_matrix.T))
    print('nDCG: {:.3f} {:.3f}'.format(np.mean(vis_nDCG), np.mean(txt_nDCG)))
    print('mAP: {:.3f} {:.3f}'.format(np.mean(vis_mAP), np.mean(txt_mAP)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test Joint Part-of-Speech Embedding Network (JPoSE) using Triplets")

    parser.add_argument('MODEL_PATH', type=str, help='Path of model to load')
    parser.add_argument('--gpu', type=bool, help='Whether or not to use the gpu for testin. [False]')

    parser.set_defaults(
            gpu=False,
    )

    main(parser.parse_args())
