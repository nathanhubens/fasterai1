import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.core import *
from fastai.callbacks import *

def make_broadcastable(input, target):
    target_shape = target.shape
    output_shape = [*target.shape]
    
    for i in range(len(target_shape)):
        input_size = np.prod(input.shape)
        target_size = np.prod(np.array(target_shape[:i+1]))
        if input_size >= target_size:
            output_shape[i]=target_shape[i]
        else:
            output_shape[i]=1
        
    new_input = input.reshape(*output_shape)        
    return new_input

def annealing_gradual(start:Number, end:Number, pct:float)->Number:
    "Gradually anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return end + start - end * (1 - pct)**3