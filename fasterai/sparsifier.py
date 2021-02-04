from fasterai.utils import *

from fastai.vision import *
from fastai.core import *
from fastai.callbacks import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sparsifier():
    def __init__(self, model, granularity, method, criteria):
        self.granularity, self.method, self.criteria, self.model = granularity, method, criteria, model
        self._save_weights() # Save the original weights
        
    def prune(self, sparsity):
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d):
                if self.criteria == 'l1':
                    weight = self._l1_norm(m.weight)
                elif self.criteria == 'grad':
                    weight = self._grad_crit(m.weight)
                elif self.criteria == 'movement':
                    weight = self._movement(m)
                else: raise NameError('Invalid Criteria')
                
                mask = self._compute_mask(self.model, weight, sparsity)
                #mask = make_broadcastable(mask, m.weight) # Reshape the mask to be broadcastable with the weights
                m.register_buffer("_mask", mask) # Put the mask into a buffer
                self._apply(m) 
            
    def _apply(self, module):
        '''
        Apply the mask and freeze the gradient so the corresponding weights are not updated anymore
        '''
        mask = getattr(module, "_mask")
        module.weight.data.mul_(mask)
        if module.weight.grad is not None: # In case some layers are freezed
            module.weight.grad.mul_(mask)

        if self.granularity == 'filter': # If we remove complete filters, we want to remove the bias as well
            if module.bias is not None:
                module.bias.data.mul_(mask.squeeze())
                if module.bias.grad is not None: # In case some layers are freezed
                    module.bias.grad.mul_(mask.squeeze())


    def _l1_norm(self, weight):

        if self.granularity == 'weight':
            w = weight.abs()
            
        elif self.granularity == 'vector':
            dim = 3 # dim=1 -> channel vector, dim=2 -> column vector, dim=3 -> row vector
            w = (torch.norm(weight, p=1, dim=dim)/(weight.shape[3])).unsqueeze(dim) # Normalize the norm to be consistent for different dimensions

        elif self.granularity == 'kernel':
            w = (torch.norm(weight, p=1, dim=(2,3))/(weight.shape[2]*weight.shape[3]))[:,:, None, None]
        
        elif self.granularity == 'filter':       
            w = (torch.norm(weight, p=1, dim=(1,2,3))/(weight.shape[1]*weight.shape[2]*weight.shape[3]))[:, None, None, None]

        else: raise NameError('Invalid Granularity') 
        
        return w
        
    def _grad_crit(self, weight):
        if weight.grad is not None:
            if self.granularity == 'weight':
                w = (weight*weight.grad).data.pow(2).view(-1)

            elif self.granularity == 'vector':
                w = (weight*weight.grad).data.pow(2).sum(dim=(3)).view(-1).clone()/(weight.shape[3])

            elif self.granularity == 'kernel':
                w = (weight*weight.grad).data.pow(2).sum(dim=(2,3)).view(-1).clone()/(weight.shape[2]*weight.shape[3])    
                
            elif self.granularity == 'filter':       
                w = (weight*weight.grad).data.pow(2).sum(dim=(1,2,3))/(weight.shape[1]*weight.shape[2]*weight.shape[3])

            else: raise NameError('Invalid Granularity') 

            return w
        
    def _movement(self, module):
        if hasattr(module, '_old_weights') == False:
            module.register_buffer("_old_weights", module._init_weights.clone()) # If the previous value of weights is not known, take the initial value
            
        old_weights = getattr(module, "_init_weights")

        if self.granularity == 'weight': 
            w = torch.abs((module.weight.view(-1)).clone()) - torch.abs(old_weights.view(-1).clone())

        elif self.granularity == 'vector': 
            w = torch.abs(module.weight.sum(dim=(3)).clone()) - torch.abs(old_weights.sum(dim=(3).clone()))

        elif self.granularity == 'kernel': 
            w = torch.abs(module.weight.sum(dim=(2,3)).clone()) - torch.abs(old_weights.sum(dim=(2,3).clone()))           
        
        elif self.granularity == 'filter': 
            w = torch.abs(module.weight.sum(dim=(1,2,3)).clone()) - torch.abs(old_weights.sum(dim=(1,2,3).clone()))

        else: raise NameError('Invalid Granularity')

        module._old_weights = module.weight.clone() # The current value becomes the old one for the next iteration
            
        return w
    
    def _reset_weights(self):
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Linear):
                init_weights = getattr(m, "_init_weights")
                m.weight.data = init_weights.clone()
            if isinstance(m, nn.Conv2d):
                init_weights = getattr(m, "_init_weights")
                m.weight.data = init_weights.clone()
                self._apply(m) # Reset the weights and apply the current mask
                
    def _save_weights(self):
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.register_buffer("_init_weights", m.weight.clone())
                
    def _clean_buffers(self):
        for k, m in enumerate(self.model.modules()):
            if isinstance(m, nn.Conv2d):
                del m._buffers["_mask"]

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                del m._buffers["_init_weights"]                
                    
    
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
                    elif self.criteria == 'grad':
                        w = self._grad_crit(m.weight)
                        
                    global_weight.append(w)

            global_weight = torch.cat([i.view(-1) for i in global_weight])
            threshold = torch.quantile(global_weight, sparsity/100) # Compute the threshold globally
            
        elif self.method == 'local': 
            threshold = torch.quantile(weight.view(-1), sparsity/100) # Compute the threshold locally
            
        else: raise NameError('Invalid Method')
            
        # Make sure we don't remove every weight of a given layer
        if threshold > weight.max(): threshold = weight.max()

        mask = weight.ge(threshold).to(dtype=weight.dtype)

        return mask


class SparsifyCallback(LearnerCallback):
    '''
    sparsity: The percentage of sparsity you want in your final model (between 0 and 100)
    granularity: The granularity the pruning will be operating on ('weights', 'vector', 'kernel', 'filters')
    method: The method of selection of the parameters ('local' or 'global')
    criteria: The criteria of selection ('l1', 'grad', 'movement')
    sched_func: The scheduling function for the pruning ('one_shot', 'iterative', 'annealing_cos', 'gradual', ...)
    start_epoch: The epoch you want to start pruning the network
    start_reset: When doing Lottery Ticket Hypothesis, the epoch you want to start resetting weights to their original values (set to 0 if you don't want to reset the weights)
    rewind: When doing Lottery Ticket Hypothesis with Rewind, the epoch you want to reset you weights to.
    reset_end: If you want to reset your weights at the end of training to get your winning ticket.
    '''
        
    def __init__(self, learn:Learner, sparsity, granularity, method, criteria, sched_func, start_epoch=0, lth_reset=False, rewind_epoch=0, reset_end=False):
        super().__init__(learn)
        self.sparsity, self.granularity, self.method, self.criteria, self.sched_func = sparsity, granularity, method, criteria, sched_func
        self.reset_end, self.rewind_epoch, self.start_epoch, self.lth_reset = reset_end, rewind_epoch, start_epoch, lth_reset
        self.sparsifier = Sparsifier(self.learn.model, self.granularity, self.method, self.criteria)
        self.batches = math.floor(len(learn.data.train_ds)/learn.data.train_dl.batch_size)
        self.current_sparsity, self.previous_sparsity = 0,0

        assert self.start_epoch>=self.rewind_epoch, 'You must rewind to an epoch before the start of the pruning process'
    
    def on_train_begin(self, n_epochs:int, **kwargs):
        print(f'Pruning of {self.granularity} until a sparsity of {self.sparsity}%')
        self.total_iters = n_epochs * self.batches
        self.start_iter = self.start_epoch * self.batches
        
    def on_epoch_end(self, epoch, **kwargs):
        print(f'Sparsity at the end of epoch {epoch}: {self.current_sparsity:.2f}%')
    
    def on_epoch_begin(self, epoch, **kwargs):
        if epoch == self.rewind_epoch:
            print(f'Saving Weights at epoch {epoch}')
            self.sparsifier._save_weights()
        
    def on_batch_begin(self, iteration, epoch, **kwargs):
        if epoch>=self.start_epoch:
            self.set_sparsity(iteration)
            self.sparsifier.prune(self.current_sparsity)

            if self.lth_reset and self.current_sparsity!=self.previous_sparsity: # If sparsity has changed, the network has been pruned
                    print(f'Resetting Weights to their epoch {self.rewind_epoch} values')
                    self.sparsifier._reset_weights()

        self.previous_sparsity = self.current_sparsity
        
    def set_sparsity(self, iteration):
        self.current_sparsity = self.sched_func(start=0., end=self.sparsity, pct=(iteration-self.start_iter)/(self.total_iters-self.start_iter))
    
    def on_train_end(self, **kwargs):
        print(f'Final Sparsity: {self.current_sparsity:.2f}')
        if self.reset_end:
            self.sparsifier._reset_weights()
        self.sparsifier._clean_buffers() # Remove buffers at the end of training