import os
import sys
import pickle
import pandas as pd
import numpy as np
import word2vec as wv
from tqdm import tqdm


def get_new_dataframe_names(df_path):
    file_name = df_path.rsplit('/')[-1]
    return 'sentence_' + file_name


def create_sentence_df(df):
    sentences = {}
    indices = {}
    classes = {}
    verb_class = {}
    noun_class = {}
    verbs = {}
    nouns = {}
    unique_sents = df.narration.unique()
    for i, sent in tqdm(enumerate(unique_sents), total=len(unique_sents)):
        sentences[i] = sent
        subset_df = df[df.narration.apply(lambda x: x == sent)]
        indices[i] = subset_df.index[0]
        classes[i] = subset_df.iloc[0].action_class
        verb_class[i] = subset_df.iloc[0].verb_class
        noun_class[i] = subset_df.iloc[0].all_noun_classes
        verbs[i] = subset_df.iloc[0].verb
        nouns[i] = subset_df.iloc[0].all_nouns
    missing_classes = set(df.action_class.unique()) - set(classes.values())
    i = len(indices)
    for class_ in missing_classes:
        sentences[i] = subset_df.iloc[0].narration
        subset_df = df[df.action_class.apply(lambda x: x == class_)]
        indices[i] = subset_df.index[0]
        classes[i] = subset_df.iloc[0].action_class
        verb_class[i] = subset_df.iloc[0].verb_class
        noun_class[i] = subset_df.iloc[0].all_noun_classes
        verbs[i] = subset_df.iloc[0].verb
        nouns[i] = subset_df.iloc[0].all_nouns
        i += 1
    sentence_df = pd.DataFrame([sentences]).T
    sentence_df.columns = ['sentence']
    sentence_df['action_class'] = pd.Series(classes)
    sentence_df['index'] = pd.Series(indices)
    sentence_df['verb_class'] = pd.Series(verb_class)
    sentence_df['noun_class'] = pd.Series(noun_class)
    sentence_df['verb'] = pd.Series(verbs)
    sentence_df['nouns'] = pd.Series(nouns)
    return sentence_df


def fix_sentence_df(df, sentence_df):
    missing_actions = set(df.action_uid) - set(sentence_df.action_uid)
    for uid in missing_actions:
        subset_df = df[df.action_uid.apply(lambda x: x == uid)]
        new_row = {}
        new_row['index'] = subset_df.index[0]
        new_row['sentence'] = subset_df.iloc[0].sentence
        new_row['action_class'] = subset_df.iloc[0].action_class
        new_row['verb_class'] = subset_df.iloc[0].verb_class
        new_row['noun_class'] = subset_df.iloc[0].all_noun_classes
        new_row['verb'] = subset_df.iloc[0].verb
        new_row['nouns'] = subset_df.iloc[0].all_nouns
        new_row['noun'] = subset_df.iloc[0].noun
        new_row['action_uid'] = subset_df.iloc[0].action_uid
        new_row['verb_uid'] = subset_df.iloc[0].verb_uid
        new_row['noun_uid'] = subset_df.iloc[0].noun_uid
        sentence_df = sentence_df.append(new_row, ignore_index=True)
    return sentence_df


def pre_process_word_idx(df, verb_to_index=None, noun_to_index=None):
    if verb_to_index is None:
        verb_to_index = {verb: i for i, verb in enumerate(df.verb.unique())}
    if noun_to_index is None:
        noun_to_index = {noun: i for i, noun in enumerate(df.noun.unique())}
    df['verb_uid'] = df.verb.apply(lambda x: verb_to_index[x])
    df['noun_uid'] = df.noun.apply(lambda x: noun_to_index[x])
    df['action_uid'] = df.apply(lambda x: '{}_{}'.format(x.verb_uid, x.noun_uid), axis=1)
    return df, verb_to_index, noun_to_index


def main(args):
    df = pd.read_pickle(args.dataframe)
    dataframe_name = get_new_dataframe_names(args.dataframe)
    #df, v2idx, n2idx = pre_process_word_idx(df)
    if 'narration_id' in df.columns:
        df = df.set_index('narration_id')
    df['action_class'] = df.apply(lambda x: '{}_{}'.format(x.verb_class, x.noun_class), axis=1)
    sentence_df = create_sentence_df(df)
    sentence_df['noun'] = sentence_df.nouns.apply(lambda x: x[0])
    #sentence_df, _, _ = pre_process_word_idx(sentence_df, v2idx, n2idx)
    #sentence_df = fix_sentence_df(df, sentence_df)

    pd.to_pickle(sentence_df, dataframe_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Creates Sentence dataframes for retrieval')
    parser.add_argument('dataframe', type=str, help='Annotations dataframe to create sentence dataframe from')
    parser.add_argument('--out-dir', type=str, help='Output directory to save file to. [./]')

    parser.set_defaults(
            out_dir='./',
    )
    not_found = {}
    main(parser.parse_args())
