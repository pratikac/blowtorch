import torch as th
import torchvision as thv
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

import math, logging, pdb
from copy import deepcopy
import numpy as np

bn_track_running_stats = False

def get_num_classes(opt):
    d = dict(mnist=10, svhn=10, svhnx=10, cifar10=10,
            cifar100=100, imagenet=1000, halfmnist=10)
    if not opt['dataset'] in d:
        assert False, 'Unknown dataset: %s'%opt['dataset']
    return d[opt['dataset']]

class View(nn.Module):
    def __init__(self,o):
        super().__init__()
        self.o = o

    def forward(self,x):
        return x.view(-1, self.o)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def num_parameters(model):
    return sum([w.numel() for w in model.parameters()])

class mnistfc(nn.Module):
    name = 'mnistfc'
    def __init__(self, opt):
        super().__init__()

        c = 1024
        opt['d'] = 0.2
        opt['l2'] = 0.

        self.layers = [
            View(784),
            nn.Dropout(0.2),
            nn.Linear(784,c),
            nn.BatchNorm1d(c, track_running_stats=bn_track_running_stats),
            nn.ReLU(inplace=True),
            nn.Dropout(opt['d']),
            nn.Linear(c,c),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
            nn.Dropout(opt['d']),
            nn.Linear(c,10)]
        self.m = nn.Sequential(*self.layers)

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class lenet(nn.Module):
    name = 'lenet'
    def __init__(self, opt, c1=20, c2=50, c3=500):
        super().__init__()

        opt['l2'] = 0.
        opt['d'] = opt.get('d', 0.25)

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.BatchNorm2d(co, track_running_stats=bn_track_running_stats),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.Dropout(p))

        self.layers = [
            convbn(1,c1,5,3,opt['d']),
            convbn(c1,c2,5,2,opt['d']),
            View(c2*2*2),
            nn.Linear(c2*2*2, c3),
            nn.BatchNorm1d(c3, track_running_stats=bn_track_running_stats),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(c3,10)]
        self.m = nn.Sequential(*self.layers)

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class lenets(lenet):
    name = 'lenets'
    def __init__(self, opt, c1=8, c2=16, c3=128):
        if (not 'd' in opt) or opt['d'] < 0:
            opt['d'] = 0.0
        super().__init__(opt, c1, c2, c3)

class allcnn(nn.Module):
    name = 'allcnn'

    def __init__(self, opt, c1=96, c2=192):
        super().__init__()

        opt['l2'] = 1e-3
        opt['d'] = opt.get('d', 0.)

        num_classes = get_num_classes(opt)

        def convbn(ci,co,ksz,s=1,pz=0):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
                nn.BatchNorm2d(co, track_running_stats=bn_track_running_stats),
                nn.ReLU(True)
                )

        self.layers = [
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(opt['d']),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,num_classes,1,1),
            nn.AvgPool2d(8),
            View(num_classes)]

        self.m = nn.Sequential(*self.layers)

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class allcnntt(allcnn):
    name = 'allcnntt'
    def __init__(self, opt, c1=4, c2=8):
        super().__init__(opt, c1, c2)

class allcnnt(allcnn):
    name = 'allcnnt'
    def __init__(self, opt, c1=8, c2=16):
        super().__init__(opt, c1, c2)

class allcnns(allcnn):
    name = 'allcnns'
    def __init__(self, opt, c1=12, c2=24):
        opt['d'] = 0.
        super().__init__(opt, c1, c2)

class allcnnl(allcnn):
    name = 'allcnnl'
    def __init__(self, opt, c1=120, c2=240):
        super().__init__(opt, c1, c2)

class caddtable_t(nn.Module):
    def __init__(self, m1, m2):
        super(caddtable_t, self).__init__()
        self.m1, self.m2 = m1, m2

    def forward(self, x):
        return th.add(self.m1(x), self.m2(x))

class wrn(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.name = 'wrn%d%d'%(opt['depth'], opt['widen'])

        opt['d'] = opt.get('d', 0.25)
        opt['l2'] = 5e-4
        opt['depth'] = opt.get('depth', 28)
        opt['widen'] = opt.get('widen', 10)

        d, depth, widen = opt['d'], opt['depth'], opt['widen']

        num_classes = get_num_classes(opt)

        nc = [16, 16*widen, 32*widen, 64*widen]
        assert (depth-4)%6 == 0, 'Incorrect depth'
        n = (depth-4)//6

        def block(ci, co, s, p=0.):
            h = nn.Sequential(
                    nn.BatchNorm2d(ci, track_running_stats=bn_track_running_stats),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ci, co, kernel_size=3, stride=s, padding=1, bias=False),
                    nn.BatchNorm2d(co, track_running_stats=bn_track_running_stats),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p),
                    nn.Conv2d(co, co, kernel_size=3, stride=1, padding=1, bias=False))
            if ci == co:
                return caddtable_t(h, nn.Sequential())
            else:
                return caddtable_t(h,
                            nn.Conv2d(ci, co, kernel_size=1, stride=s, padding=0, bias=False))

        def netblock(nl, ci, co, blk, s, p=0.):
            ls = [blk((i==0 and ci or co), co, (i==0 and s or 1), p) for i in range(nl)]
            return nn.Sequential(*ls)

        self.m = nn.Sequential(
                nn.Conv2d(3, nc[0], kernel_size=3, stride=1, padding=1, bias=False),
                netblock(n, nc[0], nc[1], block, 1, d),
                netblock(n, nc[1], nc[2], block, 2, d),
                netblock(n, nc[2], nc[3], block, 2, d),
                nn.BatchNorm2d(nc[3], track_running_stats=bn_track_running_stats),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(8),
                View(nc[3]),
                nn.Linear(nc[3], num_classes))

        for m in self.m.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                #m.weight.data.normal_(0, math.sqrt(2./m.in_features))
                m.bias.data.zero_()

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        # print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class wrn101(wrn):
    name = 'wrn101'
    def __init__(self, opt):
        opt['depth'], opt['widen'] = 10,1
        super().__init__(opt)

class wrn164(wrn):
    name = 'wrn164'
    def __init__(self, opt):
        opt['depth'], opt['widen'] = 16,4
        super().__init__(opt)

class wrn168(wrn):
    name = 'wrn168'
    def __init__(self, opt):
        opt['depth'], opt['widen'] = 16,8
        super().__init__(opt)

class resnet18(nn.Module):
    name = 'resnet18'
    def __init__(self, opt):
        super().__init__()
        opt['l2'] = 1e-4
        self.m = thv.models.resnet18(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        # print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class resnet50(nn.Module):
    name = 'resnet50'
    def __init__(self, opt):
        super().__init__()
        opt['l2'] = 1e-4
        self.m = thv.models.resnet50(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        # print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class resnet101(nn.Module):
    name = 'resnet101'
    def __init__(self, opt):
        super().__init__()
        opt['l2'] = 1e-4
        self.m = thv.models.resnet101(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        # print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class resnet152(nn.Module):
    name = 'resnet152'
    def __init__(self, opt):
        super().__init__(num_classes=get_num_classes(opt))
        opt['l2'] = 1e-4
        self.m = thv.models.resnet152()

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        # print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class alexnet(nn.Module):
    name = 'alexnet'
    def __init__(self, opt):
        super().__init__()
        self.m = getattr(thv.models, opt['m'])()

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        # print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class squeezenet(nn.Module):
    name = 'squeezenet'
    def __init__(self, opt):
        super().__init__()

        self.m = getattr(thv.models, 'squeezenet1_1')(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        # print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class densenet121(nn.Module):
    name = 'densenet121'
    def __init__(self, opt):
        super().__init__()
        opt['l2'] = 1e-4

        self.m = thv.models.densenet121(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        # print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class densenet169(nn.Module):
    name = 'densenet169'
    def __init__(self, opt):
        super().__init__()
        opt['l2'] = 1e-4

        self.m = thv.models.densenet169(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        # print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)

class densenet201(nn.Module):
    name = 'densenet201'
    def __init__(self, opt):
        super().__init__()
        opt['l2'] = 1e-4

        self.m = thv.models.densenet201(num_classes=get_num_classes(opt))

        self.N = num_parameters(self.m)
        s = '[%s] Num parameters: %d'%(self.name, self.N)
        # print(s)
        logging.info(s)

    def forward(self, x):
        return self.m(x)