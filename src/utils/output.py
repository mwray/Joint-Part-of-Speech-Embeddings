import os
import string
import argparse
import pickle

import torch as th
import random as r


def load_args(arg_file):
    with open(arg_file, 'r') as in_f:
        dict_str = in_f.readline()
    arg_dict = eval(dict_str)

    args = argparse.Namespace()
    for k in arg_dict:
        setattr(args, k, arg_dict[k])
    return args


def save_args(args, arg_file):
    with open(arg_file, 'w') as out_f:
        out_f.write(str(vars(args)))


def random_string(length):
    letters = string.ascii_letters + string.digits
    return ''.join(r.choices(letters, k=length))


def get_out_dir(args):
    out_dir = args.out_dir
    dir_name = random_string(12)

    final_out_dir = os.path.join(out_dir, dir_name)
    model_out_dir = os.path.join(out_dir, dir_name, 'model')
    results_out_dir = os.path.join(out_dir, dir_name, 'results')

    os.makedirs(final_out_dir)
    os.makedirs(model_out_dir)
    os.makedirs(results_out_dir)

    save_args(args, os.path.join(final_out_dir, 'args.txt'))
    return final_out_dir


def save_model(out_dir, model, epoch):
    model_name = '{}_{}.pth'.format(str(model.name), str(epoch))
    final_out_path = os.path.join(out_dir, 'model', model_name)
    th.save(model.state_dict(), final_out_path)


def save_results(out_dir, out_dict, name=''):
    if name == '':
        name = random_string(6) + '.pkl'
    else:
        name += '.pkl'
    out_filepath = os.path.join(out_dir, 'results', name)
    with open(out_filepath, 'wb') as out_f:
        pickle.dump(out_dict, out_f)
