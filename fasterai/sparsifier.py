from fasterai.utils import *

from fastai.vision import *
from fastai.core import *
from fastai.callbacks import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class Sparsifier():
    def __init__(self, granularity, method, criteria):
        self.granularity = granularity
        self.method = method
        self.criteria = criteria
        
    def prune(self, model, sparsity):
        
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d):
                if self.criteria == 'l1':
                    weight = self._l1_norm(m.weight)
                elif self.criteria == 'taylor':
                    weight = self._taylor_crit(m.weight)
                else: raise NameError('Invalid Criteria')
                
                mask = self._compute_mask(model, weight, sparsity)
                mask = make_broadcastable(mask, m.weight)
                m.register_buffer("_mask", mask) # Put the mask into a buffer
                self._apply(m)
            
        return model
    
    def _apply(self, module):
        '''
        Apply the mask and freeze the gradient so the corresponding weights are not updated anymore
        '''
        mask = getattr(module, "_mask")
        module.weight.data.mul_(mask)
        if module.weight.grad is not None: # In case some layers are freezed
            module.weight.grad.mul_(mask)

        if self.granularity == 'filter': # If we remove complete filters, we want to remove the bias as well
            module.bias.data.mul_(mask.squeeze())
            if module.bias.grad is not None: # In case some layers are freezed
                module.bias.grad.mul_(mask.squeeze())
    
    def _l1_norm(self, weight):
        
        if self.granularity == 'filter':       
            w = weight.abs().sum(dim=(1,2,3)).clone()

        elif self.granularity == 'weight':
            w = weight.view(-1).abs().clone()

        elif self.granularity == 'kernel':
            w = weight.abs().sum(dim=(2,3)).view(-1).clone()       

        elif self.granularity == 'vector':
            w = weight.abs().sum(dim=(3)).view(-1).clone()
        
        else: raise NameError('Invalid Granularity') 
        
        return w
        
    def _taylor_crit(self, weight):
        if weight.grad is not None:
            if self.granularity == 'filter':       
                w = (weight*weight.grad).data.pow(2).sum(dim=(1,2,3))
                #w = (weight*weight.grad).data.sum(dim=(1,2,3)).pow(2)

            elif self.granularity == 'weight':
                w = (weight*weight.grad).data.pow(2).view(-1)

            elif self.granularity == 'kernel':
                w = weight.abs().sum(dim=(2,3)).view(-1).clone()       

            elif self.granularity == 'vector':
                w = weight.abs().sum(dim=(3)).view(-1).clone()

            else: raise NameError('Invalid Granularity') 

            return w
            
    def _compute_thresh(self, weight, sparsity):
        '''
        Compute the threshold value under which we should prune
        '''
        y, i = torch.sort(weight) # Sort the weights by ascending norm    
        spars_index = int(weight.shape[0]*sparsity/100)
        threshold = y[spars_index]
        return threshold
    
    def _compute_mask(self, model, weight, sparsity):
        '''
        Compute the binary masks
        '''
        if self.method == 'global':
            global_weight = []
            
            for k, m in enumerate(model.modules()):
                if isinstance(m, nn.Conv2d):
                    if self.criteria == 'l1':
                        w = self._l1_norm(m.weight)
                    elif self.criteria == 'taylor':
                        w = self._taylor_crit(m.weight)
                        
                    global_weight.append(w)

            global_weight = torch.cat(global_weight)
            threshold = self._compute_thresh(global_weight, sparsity) # Compute the threshold globally
            
        elif self.method == 'local': 
            threshold = self._compute_thresh(weight, sparsity)
            
        else: raise NameError('Invalid Method')
            
        # Make sure we don't remove every weight of a given layer
        if threshold > weight.max(): threshold = weight.max()

        mask = weight.ge(threshold).to(dtype=weight.dtype)

        return mask
        

class SparsifyCallback(LearnerCallback):
        
    def __init__(self, learn:Learner, sparsity, granularity, method, criteria, sched_func):
        super().__init__(learn)
        self.sparsity, self.granularity, self.method, self.criteria, self.sched_func = sparsity, granularity, method, criteria, sched_func
        self.sparsifier = Sparsifier(self.granularity, self.method, self.criteria)
        self.batches = math.floor(len(learn.data.train_ds)/learn.data.train_dl.batch_size)
    
    def on_train_begin(self, n_epochs:int, **kwargs):
        print(f'Pruning of {self.granularity} until a sparsity of {self.sparsity}%')
        self.total_iters = n_epochs * self.batches
        
    def on_epoch_end(self, epoch, **kwargs):
        print(f'Sparsity at epoch {epoch}: {self.current_sparsity:.2f}%')
        
    def on_batch_begin(self,iteration, **kwargs):
        self.set_sparsity(iteration)
        #self.sparsifier.prune(self.learn.model, self.current_sparsity)
    def on_step_end(self, iteration, **kwargs):
        #self.set_sparsity(iteration)
        self.sparsifier.prune(self.learn.model, self.current_sparsity)
        
    def set_sparsity(self, iteration):
        self.current_sparsity = self.sched_func(start=0.0001, end=self.sparsity, pct=(iteration+1)/self.total_iters)
    
    def on_train_end(self, **kwargs):
        print(f'Final Sparsity: {self.current_sparsity:.2f}')