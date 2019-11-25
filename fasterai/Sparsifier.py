from fastai.callbacks import *
from fastai.vision import *

class Sparsifier():
    def __init__(self, meta):
        self.meta = meta
    
    def _compute_sparsity(self, meta):
        return meta['final_sparsity'] + (meta['initial_sparsity'] - meta['final_sparsity'])*(1 - (meta['current_step'] - meta['starting_step'])/((meta['ending_step'] - meta['starting_step'])))**3

    def prune(self, model, meta):

        binary_masks = {}

        sparsity = self._compute_sparsity(meta)
    
        for k, m in enumerate(model.children()):
        
            if self.meta['pruning_type'] == "filters" and isinstance(m, nn.Conv2d):
                
                weight = m.weight.data.abs().sum(dim=(1,2,3))
                y, i = torch.sort(weight)

                spars_index = int(weight.shape[0]*sparsity/100)
                threshold = y[spars_index]
                mask = weight.gt(threshold).float().cuda()

                binary_masks[k] = mask.view(-1, 1, 1, 1)
                m.weight.data.mul_(mask.view(-1, 1, 1, 1))
                m.bias.data.mul_(mask)
                
            if self.meta['pruning_type'] == 'weights' and isinstance(m, nn.Conv2d): 

                weight = m.weight.data.view(-1).clone().abs()
                y, i = torch.sort(weight)

                spars_index = int(weight.shape[0]*sparsity/100)
                threshold = y[spars_index]
                mask = weight.gt(threshold).float().cuda()
                
                binary_masks[k] = mask.view(m.weight.data.shape)
                m.weight.data.mul_(mask)
                
            if isinstance(m, nn.BatchNorm2d):
                
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                m.running_mean.mul_(mask)
                m.running_var.mul_(mask)    

        return binary_masks    
        

    def applyBinaryMasks(self, model, masks):

        for k, m in enumerate(model.children()):

            if isinstance(m, nn.Conv2d):
                mask = masks[k]
                m.weight.data.mul_(mask)   
                
        return model

#@dataclass
class SparsifyCallback(LearnerCallback):
        
    def __init__(self, learn:Learner, meta):
        super().__init__(learn)
        self.meta = meta
        self.sparsifier = Sparsifier(self.meta)
        self.binary_masks = None
    
    def on_train_begin(self, **kwargs):
        print(f'Pruning of {self.meta["pruning_type"]} until a sparsity of {self.meta["final_sparsity"]}%')
        
    def on_epoch_end(self, **kwargs):
        print(f'Sparsity: {self.sparsifier._compute_sparsity(self.meta):.2f}%')
        
    def on_batch_begin(self, **kwargs):

        if (self.meta['current_step'] - self.meta['starting_step']) % self.meta['span'] == 0 and self.meta['current_step'] > self.meta['starting_step'] and self.meta['current_step'] < self.meta['ending_step']:
            self.binary_masks = self.sparsifier.prune(self.learn.model, self.meta)
            
        if self.binary_masks:
            self.learn.model = self.sparsifier.applyBinaryMasks(self.learn.model, self.binary_masks)   
            
        if self.meta['current_step'] < self.meta['ending_step']:
            self.meta['current_step'] += 1