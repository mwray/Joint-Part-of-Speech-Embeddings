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
import defaults.EPIC_JPOSE as EPIC_JPOSE

from models.jpose import JPOSE
from datasets.jpose_dataset import create_epic_jpose_dataset
from evaluation import nDCG
from evaluation import mAP
from train.train_mmen_triplet import initialise_nDCG_values
from train.train_jpose_triplet import initialise_jpose_nDCG_values, create_modality_dicts



def get_features(model, dataset, PoS_list, gpu=False, use_learned_comb_func=True):
    model.eval()

    vis_feat, txt_feat = dataset.get_eval_batch(PoS_list, gpu=gpu)


    if use_learned_comb_func:
        comb_func = None
    else:
        comb_func = th.cat
    out_dict = model.forward({PoS: [{'v': vis_feat[PoS]}, {'t': txt_feat[PoS]}] for PoS in PoS_list if PoS != 'action'}, action_output=True,comb_func=comb_func)

    vis_out = out_dict[0]['v']
    txt_out = out_dict[1]['t']

    if gpu:
        vis_out = vis_out.cpu()
        txt_out = txt_out.cpu()
    vis_out = vis_out.detach().numpy()
    txt_out = txt_out.detach().numpy()

    return vis_out, txt_out



def test_epoch(model, dataset, PoS_list, gpu=False, use_learned_comb_func=True):
    model.eval()

    vis_out, txt_out = get_features(model, dataset, PoS_list, gpu=False, use_learned_comb_func=True)

    vis_sim_matrix = vis_out.dot(txt_out.T)
    txt_sim_matrix = vis_sim_matrix.T


    vis_nDCG = nDCG.calculate_nDCG(vis_sim_matrix,
            dataset['action'].relevancy_matrix, dataset['action'].k_values['v'],
            IDCG=dataset['action'].IDCG['v'])
    txt_nDCG = nDCG.calculate_nDCG(txt_sim_matrix,
            dataset['action'].relevancy_matrix.T, dataset['action'].k_values['t'],
            IDCG=dataset['action'].IDCG['t'])
    print('nDCG: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_nDCG, txt_nDCG, (vis_nDCG + txt_nDCG) / 2))

    vis_mAP = mAP.calculate_mAP(vis_sim_matrix,
            dataset['action'].relevancy_matrix)
    txt_mAP = mAP.calculate_mAP(txt_sim_matrix,
            dataset['action'].relevancy_matrix.T)
    print('mAP: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_mAP, txt_mAP, (vis_mAP + txt_mAP) / 2))


def main(args):
    arg_file_path = args.MODEL_PATH.rsplit('/', 2)[0] + '/args.txt'
    model_args = utils.output.load_args(arg_file_path)

    print(model_args)

    if not hasattr(model_args, 'num_triplets'):
        setattr(model_args, 'num_triplets', 0)
    test_ds = create_epic_jpose_dataset(is_train=False, batch_size=model_args.batch_size, num_triplets=model_args.num_triplets)
    test_ds = initialise_jpose_nDCG_values(test_ds)

    modality_dicts, comb_func = create_modality_dicts(model_args, test_ds.x_size, test_ds.y_size)

    PoS_list = ['verb', 'noun', 'action']

    jpose = JPOSE(modality_dicts, comb_func=comb_func)
    jpose.load_state_dict(th.load(args.MODEL_PATH))

    if args.gpu:
        jpose.cuda()

    test_epoch(jpose, test_ds, PoS_list, gpu=args.gpu, use_learned_comb_func=args.comb_func)
    if args.challenge_submission != '':
        test_ds = create_epic_jpose_dataset(is_train=False, batch_size=model_args.batch_size, num_triplets=model_args.num_triplets, is_test=True)
        test_df = pd.read_pickle('./data/dataframes/EPIC_100_retrieval_test.pkl')
        test_sentence_df = pd.read_pickle('./data/dataframes/EPIC_100_retrieval_test_sentence.pkl')
        test_vis, test_txt = get_features(jpose, test_ds, PoS_list, gpu=args.gpu, use_learned_comb_func=args.comb_func)
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
    parser = argparse.ArgumentParser("Test Joint Part-of-Speech Embedding Network (JPoSE) using Triplets")

    parser.add_argument('MODEL_PATH', type=str, help='Path of model to load')
    parser.add_argument('--gpu', type=bool, help='Whether or not to use the gpu for testin. [False]')
    parser.add_argument('--comb-func', type=bool, help='Whether or not to use the combination func for testing. [False]')
    parser.add_argument('--challenge-submission', type=str, help='Whether or not to create a challenge submission with given output path. ['']')

    parser.set_defaults(
            gpu=False,
            comb_func=False,
            challenge_submission=''
    )

    main(parser.parse_args())
