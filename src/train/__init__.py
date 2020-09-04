import pandas as pd

def load_dataset(dir_, is_train=True):
    dataset_type = "train" if is_train else "test"
    return pd.read_pickle("{}/{}_pre-release_v4_features.pkl".format(dir_, dataset_type))

def load_labels(dir_, is_train=True, is_verb=True):
    dataset_type = "train" if is_train else "test"
    pos_type = "verb" if is_verb else "noun"
    return pd.read_pickle("{}/{}_{}_pre-release_v4.pkl".format(dir_, dataset_type, pos_type))
