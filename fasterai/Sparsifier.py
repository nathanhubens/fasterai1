from fastai.vision import *
from fastai.callbacks import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sparsifier():
    def __init__(self, model, method, method2):
        self.model = model
        self.method = method
        self.method2 = method2
        
    def prune(self, sparsity):
        
        for k, m in enumerate(self.model.modules()):
            
            if self.method == 'filter' and isinstance(m, nn.Conv2d):
                orig_shape = m.weight.shape # Store original weight shape         
                weight = m.weight.data.abs().sum(dim=(1,2,3)).clone()
                mask = self._compute_mask(weight, sparsity).view(-1, 1, 1, 1) # Expand dimensions to allow broadcasting
                m.register_buffer("_mask", mask) # Put the mask into a buffer
                self._apply(m)
                
            elif self.method == 'kernel' and isinstance(m, nn.Conv2d):
                orig_shape = m.weight.shape
                weight = m.weight.data.abs().sum(dim=(2,3)).clone()
                mask = self._compute_mask(weight, sparsity).view(*orig_shape[:2], 1, 1)
                m.register_buffer("_mask", mask)
                self._apply(m)
                
            elif self.method == 'weight' and isinstance(m, nn.Conv2d):
                orig_shape = m.weight.shape
                weight = m.weight.data.view(-1).abs().clone()
                mask = self._compute_mask(weight, sparsity).view(*orig_shape)
                m.register_buffer("_mask", mask)
                self._apply(m)

        return self.model
    
    def _apply(self, module):
        '''
        Apply the mask and freeze the gradient so the corresponding weights are not updated anymore
        '''
        mask = getattr(module, "_mask")
        module.weight.data.mul_(mask)
        if module.weight.grad is not None: # In case some layers are freezed
            module.weight.grad.mul_(mask)
            
    def _compute_thresh(self, weight, sparsity):
        '''
        Compute the threshold value under which we should prune
        '''
        y, i = torch.sort(weight) # Sort the weights by ascending norm    
        spars_index = int(weight.shape[0]*sparsity/100)
        threshold = y[spars_index]
        return threshold
        
    
    def _compute_mask(self, weight, sparsity):
        '''
        Compute the binary masks
        '''
        if self.method2 == 'global':
            global_weight = []
            
            for k, m in enumerate(self.model.modules()):
            
                if self.method == 'filter' and isinstance(m, nn.Conv2d):        
                    w = m.weight.data.abs().sum(dim=(1,2,3)).clone()
                    global_weight.append(w)

                elif self.method == 'kernel' and isinstance(m, nn.Conv2d):
                    w = m.weight.data.abs().sum(dim=(2,3)).clone()
                    global_weight.append(w)

                elif self.method == 'weight' and isinstance(m, nn.Conv2d):
                    w = m.weight.data.view(-1).abs().clone()
                    global_weight.append(w)
            
            global_weight = torch.cat(global_weight)
            threshold = self._compute_thresh(global_weight, sparsity) # Compute the threshold globally
        else: 
            threshold = self._compute_thresh(weight, sparsity)
            
        # Make sure we don't remove every weight of a given layer
        if threshold > weight.max():
            threshold = weight.max()
            mask = weight.ge(threshold).to(dtype=weight.dtype)
        else:
            mask = weight.ge(threshold).to(dtype=weight.dtype)
        return mask
        

class SparsifyCallback(LearnerCallback):
        
    def __init__(self, learn:Learner, sparsity, method, method2, sched_func):
        super().__init__(learn)
        self.sparsity, self.method, self.method2, self.sched_func = sparsity, method, method2, sched_func
        self.sparsifier = Sparsifier(self.learn.model, self.method, self.method2)
        self.batches = math.floor(len(learn.data.train_ds)/learn.data.train_dl.batch_size)
    
    def on_train_begin(self, n_epochs:int, **kwargs):
        print(f'Pruning of {self.method} until a sparsity of {self.sparsity}%')
        self.total_iters = n_epochs * self.batches
        
    def on_epoch_end(self, epoch, **kwargs):
        print(f'Sparsity at epoch {epoch}: {self.current_sparsity:.2f}%')
        
    def on_batch_begin(self,iteration, **kwargs):
        self.set_sparsity(iteration)
        self.sparsifier.prune(self.current_sparsity)
        
    def set_sparsity(self, iteration):
        self.current_sparsity = self.sched_func(start=0, end=self.sparsity, pct=iteration/self.total_iters)
    
    def on_train_end(self, **kwargs):
        print(f'Final Sparsity: {self.current_sparsity:.2f}')