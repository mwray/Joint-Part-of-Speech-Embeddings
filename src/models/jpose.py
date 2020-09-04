import os
import sys
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.mmen import MMEN, Single_Modality_Embedding

class JPOSE(nn.Module):
    def __init__(self, modality_dicts, comb_func={'cat':[]}):
        super(JPOSE, self).__init__()
        self.MMENs = nn.ModuleDict()
        self.name='JPoSE'
        for PoS in modality_dicts:
            self.MMENs[PoS] = MMEN(modality_dicts[PoS])
        assert len(comb_func) == 1
        if 'cat' in comb_func:
            self.comb_func = th.cat
        elif 'sum' in comb_func:
            self.comb_func = th.sum
        elif 'max' in comb_func:
            self.comb_func = th.max
        elif 'fc' in comb_func:
            start_dim, end_dim = comb_func['fc']
            self.comb_func = Shared_FC_Layer(start_dim, end_dim)
        elif 'res' in comb_func:
            start_dim, end_dim = comb_func['res']
            self.comb_func = Shared_FC_Layer(start_dim, end_dim)
        else:
            raise NotImplementedError('Combined function {} not implemented'.format(comb_func.keys()))

    def forward(self, PoS_batches, action_output=False, final_norm=True, comb_func=None):
        if action_output:
            return self._forward_jpose(PoS_batches, final_norm=final_norm, comb_func=comb_func)
        else:
            PoS = list(PoS_batches.keys())[0]
            return self._forward_pos(PoS_batches[PoS], PoS, final_norm=final_norm)

    def _forward_pos(self, modality_batches, PoS, final_norm=True):
        return self.MMENs[PoS].forward(modality_batches, final_norm=final_norm)

    def _forward_jpose(self, PoS_batches, final_norm=True, comb_func=None):
        """
        PoS_batches is a dictionary of the following form:
        {PoS_1 : [
                    {'modality': tensor},
                    ...
                    {'modality': tensor}
                 ],
         PoS_2 : [
                    {'modality': tensor},
                    ...
                    {'modality': tensor}
                 ]
        ...
        }
        Note that the list for each PoS must be the same length
        """
        PoS_out = {}
        for PoS in PoS_batches:
            PoS_out[PoS] = self._forward_pos(PoS_batches[PoS], PoS, final_norm=False)
        comb_out = []
        num_modalities = len(PoS_out[list(PoS_out.keys())[0]])
        for i in range(num_modalities):
            comb_tensors = []
            for PoS in PoS_out:
                for modality_type in PoS_out[PoS][i]:
                    comb_tensors.append(PoS_out[PoS][i][modality_type])
            if not comb_func:
                c_func = self.comb_func
            else:
                c_func = comb_func
            comb_tnsr = c_func(comb_tensors, axis=1)
            if final_norm:
                comb_tnsr = F.normalize(comb_tnsr)
            comb_out.append({modality_type: comb_tnsr})
        return comb_out


class Shared_FC_Layer(nn.Module):
    def __init__(self, start_dim, end_dim, residual=False):
        super(Shared_FC_Layer, self).__init__()
        if residual:
            assert start_dim == end_dim
        self.residual = residual
        self.fc = Single_Modality_Embedding([start_dim, end_dim], num_layers=1)

    def forward(self, comb_tensor, axis=1):
        x = th.cat(comb_tensor, axis=1)
        if self.residual:
            x += self.fc(x)
        else:
            x = self.fc(x)
        return x


if __name__ == '__main__':
    modality_dict = {'verb': 
        {
            't': {
                'num_layers': 2,
                'layer_sizes': [200, 256]
            },
            'v': {
                'num_layers': 2,
                'layer_sizes': [1024, 256]
            }
        }, 'noun':
        {
            't': {
                'num_layers': 2,
                'layer_sizes': [200, 256]
            },
            'v': {
                'num_layers': 2,
                'layer_sizes': [1024, 256]
            }
        }
    }
    jpose = JPOSE(modality_dict)
    from datasets import to_tensor
    v = to_tensor(np.zeros((64, 1024)))
    t = to_tensor(np.zeros((64, 200)))
    verb = {'verb': [{'v': v}, {'t': t}, {'t': t+1}]}
    noun = {'noun': [{'v': v}, {'t': t}, {'t': t+1}]}
    jpose(verb)
    jpose(noun)
    action = {**verb, **noun}
    jpose(action, action_output=True)
