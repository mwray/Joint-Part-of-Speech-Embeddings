import argparse
from defaults import EPIC_JPOSE, EPIC_MMEN

def get_JPoSE_parser(info_str):
    parser = get_base_parser(info_str, EPIC_JPOSE)
    parser.add_argument('--action-weight', type=float, help='Weight of the action losses. [{}]'.format(EPIC_JPOSE.action_weight))
    parser.add_argument('--comb-func', type=str, help='What combination function to use for the action embedding. [{}]'.format(EPIC_JPOSE.comb_func))
    parser.add_argument('--comb-func-start', type=int, help='When to start using a learned combine function instead of concatenate. [{}]'.format(EPIC_JPOSE.comb_func_start))
    parser.add_argument('--noun-weight', type=float, help='Weight of the noun losses. [{}]'.format(EPIC_JPOSE.noun_weight))
    parser.add_argument('--verb-weight', type=float, help='Weight of the verb losses. [{}]'.format(EPIC_JPOSE.verb_weight))
    parser.set_defaults(
            action_weight=EPIC_JPOSE.action_weight,
            comb_func=EPIC_JPOSE.comb_func,
            comb_func_start=EPIC_JPOSE.comb_func_start,
            noun_weight=EPIC_JPOSE.noun_weight,
            verb_weight=EPIC_JPOSE.verb_weight
    )
    return parser


def get_MMEN_parser(info_str):
    parser = get_base_parser(info_str, EPIC_MMEN)
    parser.add_argument('caption_type', type=str, help='Type of captions to use {caption, verb, noun}.')
    parser.set_defaults(
    )
    return parser


def get_base_parser(info_str, defaults_):
    parser = argparse.ArgumentParser(info_str)
    parser.add_argument('--batch-size', type=int, help='Size of the batch during training. [{}]'.format(defaults_.batch_size))
    parser.add_argument('--checkpoint-rate', type=int, help='How many epochs between saving a model checlpoint. [{}]'.format(defaults_.checkpoint_rate))
    parser.add_argument('--embedding-size', type=int, help='Size of the resulting embedding. [{}]'.format(defaults_.embedding_size))
    parser.add_argument('--gpu', type=bool, help='Whether or not to use the gpu for training. [False]')
    parser.add_argument('--learning-rate', type=float, help='Value for the learning rate. [{}]'.format(defaults_.learning_rate))
    parser.add_argument('--margin', type=float, help='The size of the margin for the triplet losses. [{}]'.format(defaults_.margin))
    parser.add_argument('--momentum', type=float, help='Momentum used for the SGD optimiser. [{}]'.format(defaults_.momentum))
    parser.add_argument('--num-epochs', type=int, help='Number of epochs to train for. [{}]'.format(defaults_.num_epochs))
    parser.add_argument('--num-layers', type=int, help='Number of layers for each embedding network. [{}]'.format(defaults_.num_layers))
    parser.add_argument('--optimiser', type=str, help='Which optimiser to use, SGD or adam. [{}]'.format(defaults_.optimiser))
    parser.add_argument('--out-dir', type=str, help='Where to save the model and outputs. [{}]'.format(defaults_.out_dir))
    parser.add_argument('--tt-weight', type=float, help='Weight of the text to text weight. [{}]'.format(defaults_.tt_weight))
    parser.add_argument('--tv-weight', type=float, help='Weight of the text to visual weight. [{}]'.format(defaults_.tv_weight))
    parser.add_argument('--vt-weight', type=float, help='Weight of the visual to text weight. [{}]'.format(defaults_.vt_weight))
    parser.add_argument('--vv-weight', type=float, help='Weight of the visual to visual weight. [{}]'.format(defaults_.vv_weight))

    parser.set_defaults(
            batch_size=defaults_.batch_size,
            checkpoint_rate=defaults_.checkpoint_rate,
            embedding_size=defaults_.embedding_size,
            gpu=False,
            learning_rate=defaults_.learning_rate,
            margin=defaults_.margin,
            momentum=defaults_.momentum,
            num_epochs=defaults_.num_epochs,
            num_layers=defaults_.num_layers,
            optimiser=defaults_.optimiser,
            out_dir=defaults_.out_dir,
            tt_weight=defaults_.tt_weight,
            tv_weight=defaults_.tv_weight,
            vt_weight=defaults_.vt_weight,
            vv_weight=defaults_.vv_weight,
    )

    return parser
