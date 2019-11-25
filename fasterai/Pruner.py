import torch
import torch.nn as nn
import copy
import numpy as np

class Pruner():
    def __init__(self):
        super().__init__()
        
    def filters_to_keep(self, layer, nxt_layer):
    
        is_cuda = layer.weight.is_cuda
    
        filters = layer.weight
        biases = layer.bias
        nz_filters = filters.data.view(layer.out_channels, -1).sum(dim=1) # Flatten the filters to compare them
        ixs = torch.LongTensor(np.argwhere(nz_filters!=0)) # Get which filters are not equal to zero

        ixs = ixs.cuda() if is_cuda else ixs
    
        filters_keep = filters.index_select(0, ixs[0]).data # keep only the non_zero filters
        biases_keep = biases.index_select(0, ixs[0]).data
        
        if nxt_layer is not None:
            nxt_filters = nxt_layer.weight
            nxt_filters_keep = nxt_filters.index_select(1, ixs[0]).data
        else:
            nxt_filters_keep = None
            
        return filters_keep, biases_keep, nxt_filters_keep
    
    
    def prune_conv(self, layer, nxt_layer):
        assert layer.__class__.__name__ == 'Conv2d'
    
        new_weights, new_biases, new_next_weights = self.filters_to_keep(layer, nxt_layer)

        new_out_channels = new_weights.shape[0]
        new_in_channels = new_weights.shape[1]

    
        layer.out_channels = new_out_channels
        layer.in_channels = new_in_channels
    
        layer.weight = nn.Parameter(new_weights)
        layer.bias = nn.Parameter(new_biases)
    

        if new_next_weights is not None:
            new_next_in_channels = new_next_weights.shape[1]
            nxt_layer.weight = nn.Parameter(new_next_weights)
            nxt_layer.in_channels = new_next_in_channels
    
        return layer, nxt_layer

    def delete_fc_weights(self, layer, last_conv):
        
        is_cuda = last_conv.weight.is_cuda

        
        filters = last_conv.weight
        nz_filters = filters.data.view(last_conv.out_channels, -1).sum(dim=1) # Flatten the filters to compare them
        ixs = torch.LongTensor(np.argwhere(nz_filters!=0))
        
        ixs = ixs.cuda() if is_cuda else ixs
        
        weights = layer.weight.data
        
        #biases = layer.bias.data
        weights_keep = weights.index_select(1, ixs[0]).data
        
        
        layer.in_features = weights_keep.shape[1]
        layer.weight = nn.Parameter(weights_keep)
    

        return layer
    
    def _find_next_conv(self, model, conv_ix):
        for k,m in enumerate(model.children()):
            if k > conv_ix and m.__class__.__name__ == 'Conv2d':
                next_conv_ix = k
                break
            else:
                next_conv_ix = None
            
        return next_conv_ix
    
    def _get_last_conv_ix(self, model):
        layer_names = list(dict(model.named_children()).keys())
        last_conv_ix = 0
        for i in range(len(layer_names)):
            if getattr(model, layer_names[i]).__class__.__name__ == 'Conv2d':
                last_conv_ix = i
                
        return last_conv_ix
    
    def _get_first_fc_ix(self, model):
        layer_names = list(dict(model.named_children()).keys())
        first_fc_ix = 0
        for i in range(len(layer_names)):
            if getattr(model, layer_names[i]).__class__.__name__ == 'Linear':
                first_fc_ix = i
                break
                
        return first_fc_ix
    
    def prune_model(self, model):
        pruned_model = copy.deepcopy(model)
        
        layer_names = list(dict(pruned_model.named_children()).keys())
        
        for k,m in enumerate(list(pruned_model.children())):
            last_conv_ix = self._get_last_conv_ix(pruned_model)
            first_fc_ix = self._get_first_fc_ix(pruned_model)
            
            if isinstance(m, nn.Conv2d):
                next_conv_ix = self._find_next_conv(model, k)
                if next_conv_ix is not None: # The conv layer is not the last one
                    next_conv = getattr(pruned_model, layer_names[next_conv_ix]) # Get the next_conv_layer
                    new_m, new_next_m = self.prune_conv(m, next_conv) # Prune the current conv layer
                    
                    setattr(pruned_model, layer_names[k], new_m) # Apply the changes to the model
                    setattr(pruned_model, layer_names[next_conv_ix], new_next_m)

                else:
                    new_m, _ = self.prune_conv(m, None) # Prune the current conv layer without changing the next one
                    setattr(pruned_model, layer_names[k], new_m) # Apply the changes to the model
                    
            if isinstance(m, nn.Linear) and k==first_fc_ix:
                new_m = self.delete_fc_weights(m, getattr(model, layer_names[last_conv_ix]))
            
            else:
                pass
        return pruned_model