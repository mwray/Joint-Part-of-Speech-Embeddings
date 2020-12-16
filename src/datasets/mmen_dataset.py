import os
import sys
import numpy as np
import pandas as pd

from datasets import to_tensor, sample_triplets, convert_rel_dicts_to_uids
import defaults.EPIC_MMEN as EPIC_MMEN


def create_epic_mmen_dataset(caption_type, is_train=True,
        batch_size=EPIC_MMEN.batch_size, num_triplets=EPIC_MMEN.num_triplets,
        action_dataset=False):
    """
    Creates a mmen dataset object for EPIC2020 using default locations of feature files and relational dicts files.
    """
    is_train_str = 'train' if is_train else 'test'
    #Find location of word features based on caption type and load
    if caption_type in ['caption', 'verb', 'noun']:
        word_features_path = 'EPIC_100_retrieval_{}_text_features_{}.pkl'.format(caption_type, is_train_str)
    else:
        raise NotImplementedError(caption_type)
    word_features_path = './data/text_features/{}'.format(word_features_path)
    text_features = pd.read_pickle(word_features_path)

    #load video features
    video_features = pd.read_pickle('./data/video_features/EPIC_100_retrieval_{}_features_mean.pkl'.format(is_train_str))
    video_features = np.array(video_features['features'])

    #Load relational dictionaries
    rel_dicts = pd.read_pickle('./data/relational/EPIC_100_retrieval_{}_relational_dict_{}.pkl'.format(caption_type, is_train_str))
    rel_dicts = [rel_dicts['vid2class'], rel_dicts['class2vid'],
            rel_dicts['sent2class'], rel_dicts['class2sent']]

    #Load relevancy matrix
    rel_matrix = None
    if not is_train:
        rel_matrix = pd.read_pickle('./data/relevancy/EPIC_100_retrieval_{}_relevancy_mat.pkl'.format(is_train_str))
    if action_dataset:
        return MMEN_Dataset_Action(video_features, text_features, rel_dicts,
                batch_size=batch_size, num_triplets=num_triplets,
                relevancy_matrix=rel_matrix)
    else:
        return MMEN_Dataset(video_features, text_features, rel_dicts,
                batch_size=batch_size, num_triplets=num_triplets,
                relevancy_matrix=rel_matrix)


class MMEN_Dataset:
    """
    Dataset wrapper for a multi-modal embedding Network Dataset.
    """
    def __init__(self, x, y, relation_dicts, batch_size=64, num_triplets=10, x_name='v', y_name='t', relevancy_matrix=None):
        x_to_class, class_to_x, y_to_class, class_to_y = relation_dicts
        self._x = x
        self._y = y

        self.x_name = x_name
        self.y_name = y_name
        self.x_to_x_name = '{}{}'.format(x_name, x_name)
        self.x_to_y_name = '{}{}'.format(x_name, y_name)
        self.y_to_x_name = '{}{}'.format(y_name, x_name)
        self.y_to_y_name = '{}{}'.format(y_name, y_name)

        self.triplets = {self.x_to_x_name: [], self.x_to_y_name: [],
                self.y_to_x_name: [], self.y_to_y_name: []}

        self.x_size = x.shape[-1]
        self.x_len = x.shape[0]
        self.y_size = y.shape[-1]
        self.y_len = y.shape[0]

        x_uids = list(x_to_class.keys())
        self._x_idxs = np.array(list(range(len(x_uids))))
        self.x_uid_to_idx = {uid: idx for idx, uid in enumerate(x_uids)}
        self.x_idx_to_uid = {idx: uid for idx, uid in enumerate(x_uids)}
        self._y_idxs = np.array(list(y_to_class.keys()))

        x_to_class, class_to_x = convert_rel_dicts_to_uids(x_to_class, class_to_x, self.x_uid_to_idx)

        self.x_to_class = x_to_class
        self.class_to_x = class_to_x
        self.y_to_class = y_to_class
        self.class_to_y = class_to_y

        self.batch_size = batch_size
        self.num_triplets = num_triplets

        self.relevancy_matrix = relevancy_matrix

    def sample_triplets(self, cross_modal_pair, sampling_method='random'):
        if cross_modal_pair[0] == self.x_name:
            b1 = self._x
            to_class = self.x_to_class
            b_idxs = self._x_idxs
        elif cross_modal_pair[0] == self.y_name:
            b1 = self._y
            to_class = self.y_to_class
            b_idxs = self._y_idxs
        else:
            raise Exception('Unknown cross_modal_pair: {}'.format(cross_modal_pair))

        if cross_modal_pair[1] == self.x_name:
            b2 = self._x
            from_class = self.class_to_x
        elif cross_modal_pair[1] == self.y_name:
            b2 = self._y
            from_class = self.class_to_y
        else:
            raise Exception('Unknown cross_modal_pair: {}'.format(cross_modal_pair))

        self.triplets[cross_modal_pair] = []

        anchors = b_idxs
        pos_idxs, neg_idxs = sample_triplets(anchors, to_class,
                from_class, self.num_triplets, sampling_method)
        positives = pos_idxs
        negatives = neg_idxs
        self.triplets[cross_modal_pair] = (anchors, positives, negatives)
        self.sampling_method = sampling_method

    def _get_triplet_batch_start_end(self, i, modality_length):
        start = i * self.batch_size
        end = (i + 1) * self.batch_size

        if start > modality_length:
            start = start % modality_length
            end = end % modality_length
            if start > end:
                end = modality_length
        if end > modality_length:
            end = modality_length
        return start, end

    def get_triplet_batch(self, cross_modal_pairs, gpu=False):
        assert isinstance(cross_modal_pairs, list)
        modalities = set([pair[0] for pair in cross_modal_pairs])
        if self.x_to_y_name[0] in modalities and self.y_to_x_name[0] in modalities:
            longest_modality = self.x_len if self.x_len > self.y_len else self.y_len
        elif self.x_to_y_name[0] in modalities:
            longest_modality = self.x_len
        elif self.y_to_x_name[0] in modalities:
            longest_modality = self.y_len
        else:
            raise Exception('No modalities found in cross_modal_pairs: {}.'.format(cross_modal_pair))

        num_batches = int(np.ceil(longest_modality / self.batch_size))

        x_idxs = np.array(range(self.x_len))
        y_idxs = np.array(range(self.y_len))
        np.random.shuffle(x_idxs)
        np.random.shuffle(y_idxs)
        for i in range(num_batches):
            x_start, x_end = self._get_triplet_batch_start_end(i, self.x_len)
            y_start, y_end = self._get_triplet_batch_start_end(i, self.y_len)
            batch_dict = {}
            for cross_modal_pair in cross_modal_pairs:
                if cross_modal_pair[0] == self.x_to_y_name[0]:
                    anch_vals = self._x
                    start = x_start
                    end = x_end
                    idxs = x_idxs
                else:
                    anch_vals = self._y
                    start = y_start
                    end = y_end
                    idxs = y_idxs
                if cross_modal_pair[1] == self.x_to_y_name[1]:
                    other_vals = self._y
                else:
                    other_vals = self._x
                triplets = self.triplets[cross_modal_pair]
                batch_anchors_idxs = np.repeat(triplets[0][idxs[start:end]], self.num_triplets)
                batch_pos_idxs = triplets[1][idxs[start:end]].flatten()
                if self.sampling_method in ['random']:
                    batch_neg_idxs = triplets[2][idxs[start:end]].flatten()
                else:
                    neg = np.repeat(triplets[2][idxs[start:end]], self.num_triplets)
                anch = to_tensor(anch_vals[batch_anchors_idxs, :], gpu=gpu)
                pos = to_tensor(other_vals[batch_pos_idxs, :], gpu=gpu)
                if self.sampling_method in ['random']:
                    neg = to_tensor(other_vals[batch_neg_idxs, :], gpu=gpu)
                batch_dict[cross_modal_pair] = (anch, pos, neg)
            yield batch_dict

    def get_eval_batch(self, gpu=False):
        return to_tensor(self._x, gpu=gpu), to_tensor(self._y, gpu=gpu)


class MMEN_Dataset_Action(MMEN_Dataset):
    def __init__(self, x, y, relation_dicts, batch_size=64, num_triplets=10, x_name='v', y_name='t', relevancy_matrix=None):
        super(MMEN_Dataset_Action, self).__init__(x, y, relation_dicts,
                batch_size=batch_size, num_triplets=num_triplets,
                x_name=x_name, y_name=y_name, relevancy_matrix=relevancy_matrix)

    def get_triplet_batch(self, cross_modal_pairs, PoS_datasets, gpu=False):
        assert isinstance(cross_modal_pairs, list)
        modalities = set([pair[0] for pair in cross_modal_pairs])
        if self.x_to_y_name[0] in modalities and self.y_to_x_name[0] in modalities:
            longest_modality = self.x_len if self.x_len > self.y_len else self.y_len
        elif self.x_to_y_name[0] in modalities:
            longest_modality = self.x_len
        elif self.y_to_x_name[0] in modalities:
            longest_modality = self.y_len
        else:
            raise Exception('No modalities found in cross_modal_pairs: {}.'.format(cross_modal_pair))

        num_batches = int(np.ceil(longest_modality / self.batch_size))

        x_idxs = np.array(range(self.x_len))
        y_idxs = np.array(range(self.y_len))
        np.random.shuffle(x_idxs)
        np.random.shuffle(y_idxs)
        for i in range(num_batches):
            x_start, x_end = self._get_triplet_batch_start_end(i, self.x_len)
            y_start, y_end = self._get_triplet_batch_start_end(i, self.y_len)
            batch_dict = {}
            for cross_modal_pair in cross_modal_pairs:
                if cross_modal_pair[0] == self.x_to_y_name[0]:
                    anch_vals = {PoS: PoS_datasets[PoS]._x for PoS in PoS_datasets}
                    start = x_start
                    end = x_end
                    idxs = x_idxs
                else:
                    anch_vals = {PoS: PoS_datasets[PoS]._y for PoS in PoS_datasets}
                    start = y_start
                    end = y_end
                    idxs = y_idxs
                if cross_modal_pair[1] == self.y_name:
                    other_vals = {PoS: PoS_datasets[PoS]._y for PoS in PoS_datasets}
                else:
                    other_vals = {PoS: PoS_datasets[PoS]._x for PoS in PoS_datasets}
                triplets = self.triplets[cross_modal_pair]
                batch_anchors_idxs = np.repeat(triplets[0][idxs[start:end]], self.num_triplets)
                batch_pos_idxs = triplets[1][idxs[start:end]].flatten()
                if self.sampling_method in ['random']:
                    batch_neg_idxs = triplets[2][idxs[start:end]].flatten()
                else:
                    neg = np.repeat(triplets[2][idxs[start:end]], self.num_triplets)
                anch = {PoS: to_tensor(anch_vals[PoS][batch_anchors_idxs, :], gpu=gpu) for PoS in PoS_datasets}
                pos = {PoS: to_tensor(other_vals[PoS][batch_pos_idxs, :], gpu=gpu) for PoS in PoS_datasets}
                if self.sampling_method in ['random']:
                    neg = {PoS: to_tensor(other_vals[PoS][batch_neg_idxs, :], gpu=gpu) for PoS in PoS_datasets}
                batch_dict[cross_modal_pair] = (anch, pos, neg)
            yield batch_dict


if __name__ == '__main__':
    for caption_type in ['caption', 'verb', 'noun']:
        print('LOADING: {}'.format(caption_type))
        mm_ds = create_epic_mmen_dataset(caption_type)
        mm_ds.sample_triplets('vt')
        mm_ds.sample_triplets('tv')
        for batch in mm_ds.get_triplet_batch(['vt', 'tv']):
            vt_batch = batch['vt']
            tv_batch = batch['tv']
            anch, pos, neg = vt_batch
            assert pos.shape == neg.shape
            assert anch.shape[0] == pos.shape[0]
            anch, pos, neg = tv_batch
            assert pos.shape == neg.shape
            assert anch.shape[0] == pos.shape[0]
        eval_batch = mm_ds.get_eval_batch()
