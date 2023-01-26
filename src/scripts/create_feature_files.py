import os
import sys
import pickle
import pandas as pd
import numpy as np
import word2vec as wv
from tqdm import tqdm


def get_new_dataframe_names(df_path, class_):
    file_name = df_path.rsplit('/')[-1]
    class_str = '' if class_ else 'no-class_'
    return class_str + '{}_features_' + file_name


def save_output_pickle(df, out_dir, df_name, type_):
    out_path = '/'.join([out_dir, df_name.format(type_)])
    out_arr = np.array(df)
    with open(out_path, 'wb') as out_f:
        pickle.dump(out_arr, out_f)


def create_sentence_df(df):
    df['action_class'] = df.apply(lambda x: '{}_{}'.format(x.verb_class, x.noun_class), axis=1)
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
        subset_df = df[df.action_class.apply(lambda x: x == class_)]
        sentences[i] = subset_df.iloc[0].narration
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


def get_word_embedding(model, word):
    global not_found
    def make_replacements(word, list_rep):
        for search, replacement in list_rep:
            word = word.replace(search, replacement)
        return word
    word = make_replacements(word, [('.', ''), ('(', ''), (')', ''), (',', ''), ("'", '')])
    word = word.lower()
    if '/' in word:
        word = word.split('/')[0]
    try:
        return model[word]
    except:
        if word not in not_found:
            print('{} not found in model'.format(word))
            vec = np.random.rand(model.vectors.shape[1])
            not_found[word] = [vec, 1]
            return vec
        else:
            not_found[word][1] += 1
            return not_found[word][0]


def create_sentence_only_vector(model, sentence, type_='average', split_str=' '):
    if type_ == 'average':
        col_func = np.mean
    words = sentence.split(split_str)
    sentence_vector = col_func(np.array([get_word_embedding(model, w) for w in words]), axis=0)
    return sentence_vector


def pre_process_word_idx(df, verb_to_index=None, noun_to_index=None):
    if verb_to_index is None:
        verb_to_index = {verb: i for i, verb in enumerate(df.verb.unique())}
    if noun_to_index is None:
        noun_to_index = {noun: i for i, noun in enumerate(df.noun.unique())}
    df['verb_uid'] = df.verb.apply(lambda x: verb_to_index[x])
    df['noun_uid'] = df.noun.apply(lambda x: noun_to_index[x])
    df['action_uid'] = df.apply(lambda x: '{}_{}'.format(x.verb_uid, x.noun_uid), axis=1)
    return df, verb_to_index, noun_to_index


def create_noun_only_vector(model, nouns, type_='average'):
    if type_ == 'average':
        col_func = np.mean
    final_nouns = []
    for noun in nouns:
        if ':' in noun:
            final_nouns += noun.split(':')
        else:
            final_nouns.append(noun)
    sentence_vector = col_func(np.array([get_word_embedding(model, n) for n in final_nouns]), axis=0)
    return sentence_vector


def main(args):
    df = pd.read_pickle(args.dataframe)
    datafame_name = get_new_dataframe_names(args.dataframe, args.class_)
    if not args.class_:
        df, v2idx, n2idx = pre_process_word_idx(df)
        sentence_df = create_sentence_df(df)
        sentence_df['noun'] = sentence_df.nouns.apply(lambda x: x[0])
        sentence_df, _, _ = pre_process_word_idx(sentence_df, v2idx, n2idx)
        sentence_df = fix_sentence_df(df, sentence_df)
    else:
        if args.sentence_dataframe == '':
            sentence_df = create_sentence_df(df)
        else:
            sentence_df = pd.read_pickle(args.sentence_dataframe)
    model = wv.load(args.word2vec_model)

    if args.caption:
        caption_features = []
    if args.verb:
        verb_features = []
    if args.noun:
        noun_features = []


    for i, row in tqdm(sentence_df.iterrows(), total=len(sentence_df)):
        if args.caption:
            caption_features.append(create_sentence_only_vector(model, row.narration))
        if args.verb:
            verb_features.append(create_sentence_only_vector(model, row.verb, split_str='-'))
        if args.noun:
            noun_features.append(create_noun_only_vector(model, row.nouns))

    if args.caption:
        save_output_pickle(caption_features, args.out_dir, datafame_name, 'caption')
    if args.verb:
        save_output_pickle(verb_features, args.out_dir, datafame_name, 'verb')
    if args.noun:
        save_output_pickle(noun_features, args.out_dir, datafame_name, 'noun')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Creates word2vec feature files for MMEN/JPoSE')
    parser.add_argument('dataframe', type=str, help='Annotations dataframe to create feature files from')
    parser.add_argument('word2vec_model', type=str, help='Path to word2vec model')
    parser.add_argument('--sentence-dataframe', type=str, help='Sentence dataframe to use. ['']')
    parser.add_argument('--out-dir', type=str, help='Output directory to save files to. [./]')
    parser.add_argument('--caption', action='store_true', help='Create caption word2vec file. [True]')
    parser.add_argument('--no-caption', action='store_false', dest='caption', help='Do not create caption word2vec file.')
    parser.add_argument('--verb', action='store_true', help='Create verb word2vec file. [True]')
    parser.add_argument('--no-verb', action='store_false', dest='verb', help='Do not create verb word2vec file.')
    parser.add_argument('--noun', action='store_true', help='Create noun word2vec file. [True]')
    parser.add_argument('--no-noun', action='store_false', dest='noun', help='Do not noun caption word2vec file.')
    parser.add_argument('--class', action='store_true', dest='class_', help='Use classes when creating relational files. [True]')
    parser.add_argument('--no-class', action='store_false', dest='class_', help='Do not use classes when creating relational files.')

    parser.set_defaults(
            sentence_dataframe='',
            out_dir='./',
            caption=True,
            verb=True,
            noun=True,
            class_=True
    )
    not_found = {}
    main(parser.parse_args())
