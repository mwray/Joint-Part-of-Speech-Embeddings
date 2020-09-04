import os

out_dir = os.path.join('.', 'logs', 'runs')
checkpoint_rate = 10

# Training Parameters
batch_size = 64
learning_rate = 0.01
margin = 1.0
momentum = 0.9
num_triplets = 10
num_epochs = 100
optimiser = 'SGD'
## Triplet Training Parameters
triplet_sampling_rate = 10


# Model Parameters
embedding_size = 256
num_layers = 2
comb_func = 'cat'
comb_func_start = 0

#modality weights
tt_weight = 1.0
tv_weight = 2.0
vt_weight = 1.0
vv_weight = 1.0

#Part of Speech weights
action_weight = 1.0
verb_weight = 1.0
noun_weight = 1.0
