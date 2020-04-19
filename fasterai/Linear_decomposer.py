import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def decompose_fc(model, percent_removed=0.5):

    new_model = copy.deepcopy(model)

    module_names = list(new_model._modules)

    for k, name in enumerate(module_names):
        
        if len(list(new_model._modules[name]._modules)) > 0:
            new_model._modules[name] = decompose_fc(new_model._modules[name])
            
        else:
            if isinstance(new_model._modules[name], nn.Linear):
                # Folded BN
                layer = SVD(new_model._modules[name])

                # Replace old weight values
                new_model._modules[name] = layer # Replace the FC Layer by the decomposedversion
                    
                    
    return new_model



def SVD(layer, percent_removed=0.5):
    
    W = layer.weight.data
    U, S, V = torch.svd(W)
    L = int(percent_removed*U.shape[0])
    W1 = U[:,:L]
    W2 = torch.diag(S[:L]) @ V[:,:L].t()
    layer_1 = nn.Linear(in_features=layer.in_features, 
                out_features=L, bias=False)
    layer_1.weight.data = W2

    layer_2 = nn.Linear(in_features=L, 
                out_features=layer.out_features, bias=True)
    layer_2.weight.data = W1
    
    if layer.bias.data is None: 
        layer_2.bias.data = torch.zeros(*layer.out_features.shape)
    else:
        layer_2.bias.data = layer.bias.data
    
    return nn.Sequential(layer_1, layer_2)