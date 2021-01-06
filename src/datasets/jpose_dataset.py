import os
import sys
import numpy as np
import pandas as pd

from datasets import to_tensor, sample_triplets, convert_rel_dicts_to_uids 
from datasets.mmen_dataset import MMEN_Dataset, create_epic_mmen_dataset
import defaults.EPIC_MMEN as EPIC_MMEN

def create_epic_jpose_dataset(is_train=True, batch_size=EPIC_MMEN.batch_size, num_triplets=EPIC_MMEN.num_triplets, is_test=False):
    verb_ds = create_epic_mmen_dataset('verb', is_train=is_train,
            batch_size=batch_size, num_triplets=num_triplets, action_dataset=False, is_test=is_test)
    noun_ds = create_epic_mmen_dataset('noun', is_train=is_train,
            batch_size=batch_size, num_triplets=num_triplets, action_dataset=False, is_test=is_test)
    action_ds = create_epic_mmen_dataset('caption', is_train=is_train,
            batch_size=batch_size, num_triplets=num_triplets, action_dataset=True, is_test=is_test)

    return JPOSE_Dataset(verb_ds, noun_ds, action_ds, batch_size=batch_size, num_triplets=num_triplets)


class JPOSE_Dataset:
    def __init__(self, verb_ds, noun_ds, action_ds, batch_size=64, num_triplets=10, x_name='v', y_name='t'):
        self.verb = verb_ds
        self.noun = noun_ds
        self.action = action_ds

        self.batch_size = batch_size
        self.num_triplets = num_triplets

        self.x_name = x_name
        self.y_name = y_name

        assert self.verb.x_size == self.noun.x_size == self.action.x_size
        assert self.verb.y_size == self.noun.y_size == self.action.y_size
        self.x_size = self.verb.x_size
        self.y_size = self.verb.y_size

        assert self.verb.x_len == self.noun.x_len == self.action.x_len
        assert self.verb.y_len == self.noun.y_len == self.action.y_len
        self.x_len = self.verb.x_len
        self.y_len = self.verb.y_len

        self.num_batches = int(np.ceil(len(self) / self.batch_size))

    def sample_triplets(self, pos_list, cross_modal_pair, sampling_method='random'):
        for pos in pos_list:
            self[pos].sample_triplets(cross_modal_pair, sampling_method=sampling_method)

    def get_triplet_batch(self, PoS_list, cross_modal_pairs, gpu=False):
        generator_dict = {}
        assert isinstance(cross_modal_pairs, list)
        for PoS in PoS_list:
            if PoS == 'action':
                PoS_datasets = {'verb': self.verb, 'noun': self.noun}
                generator_dict[PoS] = self[PoS].get_triplet_batch(cross_modal_pairs, PoS_datasets, gpu=gpu)
            else:
                generator_dict[PoS] = self[PoS].get_triplet_batch(cross_modal_pairs, gpu=gpu)

        for i in range(self.num_batches):
            batch_dict = {}
            for PoS in PoS_list:
                batch_dict[PoS] = next(generator_dict[PoS])
            yield batch_dict

    def get_eval_batch(self, pos_list, gpu=False):
        vis_batch = {}
        txt_batch = {}
        for pos in pos_list:
            vis_batch[pos], txt_batch[pos] = self[pos].get_eval_batch(gpu=gpu)
        return vis_batch, txt_batch

    def __getitem__(self, idx):
        if idx == 'verb':
            return self.verb
        elif idx == 'noun':
            return self.noun
        elif idx == 'action':
            return self.action

    def __len__(self):
        return max(self.x_len, self.y_len)


if __name__ == '__main__':
    jpose_ds = create_epic_jpose_dataset()
    jpose_ds.sample_triplets(['verb', 'noun', 'action'], 'tv')
    for batch in jpose_ds.get_triplet_batch(['verb', 'noun', 'action'], ['tv']):
        pass
    jpose_ds.get_eval_batch(['verb', 'noun', 'action'])
