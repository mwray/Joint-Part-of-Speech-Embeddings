import os
import sys
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MMEN(nn.Module):
    def __init__(self, modality_dict):
        """
        modality_dict: {modality: {'layer_sizes': [], 'num_layers': \#}}
        """
        super(MMEN, self).__init__()
        self.SMEs = nn.ModuleDict()
        self.name='MMEN'
        for modality in modality_dict:
            layer_sizes = modality_dict[modality]['layer_sizes']
            num_layers = modality_dict[modality]['num_layers']
            self.SMEs[modality] = Single_Modality_Embedding_Gated(layer_sizes)

    def forward(self, modality_batches, final_norm=True):
        """
        modality_batch is a list of dictionaries which are modality: [batch]
        """
        xs = []
        for modality_dict in modality_batches:
            assert len(modality_dict) == 1
            modality = list(modality_dict.keys())[0]
            assert modality in self.SMEs
            x = F.normalize(modality_dict[modality])
            x = self.SMEs[modality](x)
            if final_norm:
                x = F.normalize(x)
            xs.append({modality: x})
        return xs

class Single_Modality_Embedding(nn.Module):
    """
    Single basic embedding function.
    Used to project a single modality into a different space.
    """
    def __init__(self, layer_sizes, num_layers=2):
        """
        Inputs:
        - layer_sizes: list of layer sizes, must be of length 2 (i.e. start/end
          embedding space size) or of length number of layers + 1 (size of
          start, end and all intermediary layers)
        - num_layers: number of fully connected layers in the embedding.
        """
        super(Single_Modality_Embedding, self).__init__()
        assert len(layer_sizes) == 2 or len(layer_sizes) == num_layers + 1
        if len(layer_sizes) != num_layers + 1:
            if len(layer_sizes) > 2:
                print('Warning. Single Modality Embedding has been given more \
                        than two layer sizes ({}) but this does not match the \
                        number required for {} layers (requires \
                            {})'.format(len(layer_sizes), num_layers,
                        num_layers + 1))
            #Linearly interpolate the layer sizes
            final_layer_sizes = [0 for i in range(num_layers + 1)]
            start = layer_sizes[0]
            diff = (layer_sizes[-1] - start) // num_layers
            for i in range(num_layers):
                final_layer_sizes[i] = start + (i * diff)
            final_layer_sizes[-1] = layer_sizes[-1]
        else:
            final_layer_sizes = layer_sizes
        self.fc_layers = nn.ModuleList()
        for i in range(num_layers):
            self.fc_layers.append(nn.Linear(final_layer_sizes[i], final_layer_sizes[i+1]))
        self.batch_norm = nn.BatchNorm1d(final_layer_sizes[-1])

    def forward(self, x, last_relu=False):
        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)
            if i == len(self.fc_layers) - 1:
                if last_relu:
                    x = F.relu(x)
            else:
                x = F.relu(x)
        x = self.batch_norm(x)
        return x


class Single_Modality_Embedding_Gated(nn.Module):
    def __init__(self, layer_sizes):
        super(Single_Modality_Embedding_Gated, self).__init__()
        self.fc = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.cg = Context_Gating(layer_sizes[1])

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        return F.normalize(x)

class Context_Gating(nn.Module):
    def __init__(self, dimension, add_batch_norm=False):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)
        
    def forward(self,x):
        x1 = self.fc(x)

        if self.add_batch_norm:
            x1 = self.batch_norm(x1) 

        x = th.cat((x, x1), 1)
        
        return F.glu(x,1)

if __name__ == '__main__':
    sme = Single_Modality_Embedding([1024, 256], 4)
    from datasets import to_tensor
    sme(to_tensor(np.random.rand(64, 1024)))
    modality_dict = {'text': {'layer_sizes': [200, 256], 'num_layers': 2}, 
            'visual': {'layer_sizes': [1024, 256], 'num_layers': 2}}
    mmen = MMEN(modality_dict)
    xs = mmen([{'text': to_tensor(np.random.rand(64, 200))}, {'visual':
        to_tensor(np.random.rand(64, 1024))}])
