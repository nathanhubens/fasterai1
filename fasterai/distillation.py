import fastai
from fastai.vision import *
from fastai.callbacks import *


class KnowledgeDistillation(LearnerCallback):
    def __init__(self, learn:Learner, teacher:Learner, T:float=20., α:float=0.7):
        super().__init__(learn)
        self.teacher = teacher
        self.T, self.α = T, α
    
    def on_backward_begin(self, last_input, last_output, last_target, **kwargs):
        self.teacher.model.eval()
        teacher_output = self.teacher.model(last_input)
        new_loss = DistillationLoss(last_output, last_target, teacher_output, self.T, self.α)
        
        return {'last_loss': new_loss}

def DistillationLoss(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss(reduction='batchmean')(F.log_softmax(y/T, dim=-1), F.softmax(teacher_scores/T, dim=-1)) * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)