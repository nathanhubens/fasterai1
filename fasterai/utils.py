import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.core import *
from fastai.callbacks import *


def annealing_gradual(start:Number, end:Number, pct:float)->Number:
    "Gradually anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return end + start - end * (1 - pct)**3

def iterative(start:Number, end:Number, pct:float, n_steps=3)->Number:
    "Perform iterative pruning, and pruning in `n_steps` steps"
    return start + ((end-start)/n_steps)*(np.ceil((pct)*n_steps))
    
def one_shot(start:Number, end:Number, pct:float)->Number:
    "Perform one_shot pruning"
    return end