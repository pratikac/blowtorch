import torch as th
import torchvision as thv
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

import math, logging, pdb
from copy import deepcopy
import numpy as np

class ell2(_Loss):
    def __init__(self, opt, model):
        super().__init__()
        self.m = model
        self.l2 = opt['l2']

        self.wd = []
        self.l2s = []
        for m in model.modules():
            if not isinstance(m, nn.Sequential):
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    for n,p in m.named_parameters():
                        if not n in ['bias']:
                            self.wd.append(p)

    def forward(self, yh, y):
        f = 0
        for p in self.wd:
            f = f + self.l2/2.*p.norm()**2
        return f

class wrap(_Loss):
    def __init__(self, *losses):
        super().__init__()
        self.losses = losses

    def forward(self, yh, y, x=None):
        f = 0
        for l in self.losses:
            f += l(yh,y)
        return f