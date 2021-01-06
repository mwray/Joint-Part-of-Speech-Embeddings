import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch as th
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils
import defaults.EPIC_MMEN as EPIC_MMEN

from models.mmen import MMEN
from datasets.mmen_dataset import create_epic_mmen_dataset
from evaluation import nDCG
from evaluation import mAP
from train.train_mmen_triplet import initialise_nDCG_values

def test_epoch(model, dataset, gpu=False):
    model.eval()

    vis_feat, txt_feat = dataset.get_eval_batch(gpu=gpu)
    out_dict = model.forward([{'v': vis_feat}, {'t': txt_feat}])

    vis_feat = out_dict[0]['v']
    txt_feat = out_dict[1]['t']

    if gpu:
        vis_feat = vis_feat.cpu()
        txt_feat = txt_feat.cpu()
    vis_feat = vis_feat.detach().numpy()
    txt_feat = txt_feat.detach().numpy()

    vis_sim_matrix = vis_feat.dot(txt_feat.T)
    txt_sim_matrix = vis_sim_matrix.T

    vis_nDCG = nDCG.calculate_nDCG(vis_sim_matrix, dataset.relevancy_matrix,
            dataset.k_counts['v'], IDCG=dataset.IDCG_values['v'])
    txt_nDCG = nDCG.calculate_nDCG(txt_sim_matrix, dataset.relevancy_matrix.T,
            dataset.k_counts['t'], IDCG=dataset.IDCG_values['t'])

    print('nDCG: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_nDCG, txt_nDCG, (vis_nDCG + txt_nDCG) / 2))

    vis_mAP = mAP.calculate_mAP(vis_sim_matrix, dataset.relevancy_matrix)
    txt_mAP = mAP.calculate_mAP(txt_sim_matrix, dataset.relevancy_matrix.T)

    print('mAP: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_mAP, txt_mAP, (vis_mAP + txt_mAP) / 2))


def main(args):
    arg_file_path = args.MODEL_PATH.rsplit('/', 2)[0] + '/args.txt'
    model_args = utils.output.load_args(arg_file_path)

    print(model_args)

    test_ds = create_epic_mmen_dataset(model_args.caption_type, is_train=False, batch_size=model_args.batch_size, num_triplets=model_args.num_triplets)
    test_IDCG_values, test_k_counts = initialise_nDCG_values(test_ds)
    test_ds.k_counts = test_k_counts
    test_ds.IDCG_values = test_IDCG_values

    modality_dict = {
            't': {
                'num_layers': model_args.num_layers,
                'layer_sizes': [test_ds.y_size, model_args.embedding_size]
            },
            'v': {
                'num_layers': model_args.num_layers,
                'layer_sizes': [test_ds.x_size, model_args.embedding_size]
            }
    }
    mmen = MMEN(modality_dict)

    mmen.load_state_dict(th.load(args.MODEL_PATH))

    if args.gpu:
        mmen.cuda()

    test_epoch(mmen, test_ds, gpu=args.gpu)
    if args.challenge_submission != '':
        test_ds = create_epic_mmen_dataset(model_args.caption_type, is_train=False, batch_size=model_args.batch_size, num_triplets=model_args.num_triplets, is_test=True)
        test_df = pd.read_pickle('./data/dataframes/EPIC_100_retrieval_test.pkl')
        test_sentence_df = pd.read_pickle('./data/dataframes/EPIC_100_retrieval_test_sentence.pkl')
        vis_feat, txt_feat = test_ds.get_eval_batch(gpu=args.gpu)
        out_dict = mmen.forward([{'v': vis_feat}, {'t': txt_feat}])

        vis_feat = out_dict[0]['v']
        txt_feat = out_dict[1]['t']

        if args.gpu:
            vis_feat = vis_feat.cpu()
            txt_feat = txt_feat.cpu()
        test_vis = vis_feat.detach().numpy()
        test_txt = txt_feat.detach().numpy()

        sim_mat = test_vis.dot(test_txt.T)
        out_dict = {}
        out_dict['version'] = 0.1
        out_dict['challenge'] = 'multi_instance_retrieval'
        out_dict['sim_mat'] = sim_mat
        out_dict['vis_ids'] = test_df.index
        out_dict['txt_ids'] = test_sentence_df.index
        out_dict['sls_pt'] = 2
        out_dict['sls_tl'] = 3
        out_dict['sls_td'] = 3
        pd.to_pickle(out_dict, args.challenge_submission, protocol=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test Multi-Modal Embedding Network (MMEN) using Triplets")

    parser.add_argument('MODEL_PATH', type=str, help='Path of model to load')
    parser.add_argument('--gpu', type=bool, help='Whether or not to use the gpu for testin. [False]')
    parser.add_argument('--challenge-submission', type=str, help='Whether or not to create a challenge submission with given output path. ['']')

    parser.set_defaults(
            gpu=False,
            challenge_submission=''
    )

    main(parser.parse_args())
