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
        ixs = torch.nonzero(nz_filters).T # Get which filters are not equal to zero

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
    
    def prune_bn(self, layer, prev_conv):
             
        is_cuda = prev_conv.weight.is_cuda

        filters = prev_conv.weight
        nz_filters = filters.data.view(prev_conv.out_channels, -1).sum(dim=1) # Flatten the filters to compare them
        ixs = torch.nonzero(nz_filters).T
        
        ixs = ixs.cuda() if is_cuda else ixs
        
        weights = layer.weight.data
        running_mean = layer.running_mean.data
        running_var = layer.running_var.data
        biases = layer.bias.data
        
        weights_keep = weights.index_select(0, ixs[0]).data
        biases_keep = biases.index_select(0, ixs[0]).data
        mean_keep = running_mean.index_select(0, ixs[0]).data
        var_keep = running_var.index_select(0, ixs[0]).data
        
        new_num_features = weights_keep.shape[0]
    
        layer.num_features = new_num_features
        
        layer.weight = nn.Parameter(weights_keep)
        layer.bias = nn.Parameter(biases_keep)
        layer.running_mean = mean_keep
        layer.running_var = var_keep
        
        return layer

    def delete_fc_weights(self, layer, last_conv, pool_shape):
        
        is_cuda = last_conv.weight.is_cuda

        filters = last_conv.weight
        nz_filters = filters.data.view(last_conv.out_channels, -1).sum(dim=1) # Flatten the filters to compare them
        ixs = torch.nonzero(nz_filters).T
        
        ixs = ixs.cuda() if is_cuda else ixs
        
        weights = layer.weight.data
        
        if pool_shape:
            new_ixs = torch.cat([torch.arange(i*pool_shape**2,((i+1)*pool_shape**2)) for i in ixs[0]]) # The pooling size affects the number of vectors to remove in the fc layer.
        
        else: new_ixs=ixs[0]

        new_ixs =  torch.LongTensor(new_ixs).cuda() if is_cuda else torch.LongTensor(new_ixs)

        weights_keep = weights.index_select(1, new_ixs).data
        
        layer.in_features = weights_keep.shape[1]
        layer.weight = nn.Parameter(weights_keep)
    
        return layer
    
    def _find_next_conv(self, model, conv_ix):
        for k,m in enumerate(model.modules()):
            if k > conv_ix and isinstance(m, nn.Conv2d):
                next_conv_ix = k
                break
            else:
                next_conv_ix = None
            
        return next_conv_ix
    
    def _find_previous_conv(self, model, layer_ix):
        for k,m in reversed(list(enumerate(model.modules()))):
            if k < layer_ix and isinstance(m, nn.Conv2d):
                prev_conv_ix = k
                break
            else:
                prev_conv_ix = None
        return prev_conv_ix    
    
    def _get_last_conv_ix(self, model):
        for k,m in enumerate(list(model.modules())):
            if isinstance(m, nn.Conv2d):
                last_conv_ix = k
        return last_conv_ix

    
    def _get_first_fc_ix(self, model):
        for k,m in enumerate(list(model.modules())):
            if isinstance(m, nn.Linear):
                first_fc_ix = k
                break       
        return first_fc_ix
    
    def _find_pool_shape(self, model):
        for k,m in enumerate(model.modules()):
            if isinstance(m, nn.AdaptiveAvgPool2d):
                output_shape = m.output_size
                break
            else: output_shape=None

        return output_shape    
    
    def prune_model(self, model):
        pruned_model = copy.deepcopy(model)
        
        layer_names = list(dict(pruned_model.named_modules()).keys())
        layers = dict(pruned_model.named_modules())
        old_layers = dict(model.named_modules())
        
        last_conv_ix = self._get_last_conv_ix(pruned_model)
        first_fc_ix = self._get_first_fc_ix(pruned_model)
        
        
        for k,m in enumerate(list(pruned_model.modules())):
        #for k,m in enumerate(list(model.modules())):
            
            if isinstance(m, nn.Conv2d):
                next_conv_ix = self._find_next_conv(model, k)
                if next_conv_ix is not None: # The conv layer is not the last one

                    next_conv = layers[layer_names[next_conv_ix]] # Get the next_conv_layer
                    new_m, new_next_m = self.prune_conv(m, next_conv) # Prune the current conv layer

                else:
                    new_m, _ = self.prune_conv(m, None) # Prune the current conv layer without changing the next one
                    
            if isinstance(m, nn.BatchNorm2d):
                new_m = self.prune_bn(m, old_layers[layer_names[self._find_previous_conv(model, k)]])             
                    
            if isinstance(m, nn.Linear) and k==first_fc_ix:
                pool_shape = self._find_pool_shape(model)
                new_m = self.delete_fc_weights(m, old_layers[layer_names[last_conv_ix]], pool_shape[0])
            else:
                pass
        return pruned_model