import os, pdb, sys, json, subprocess
import numpy as np
import time, logging, pprint
from scipy import interpolate

import torch as th
import torchnet as tnt
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import os

home = os.path.dirname(os.path.abspath(__file__))
home = os.path.split(home)[0]

colors = {  'red':['\033[1;31m','\033[0m'],
            'blue':['\033[1;34m','\033[0m']}

def color(c, s):
    return colors[c][0] + s + colors[c][1]

def add_args(args, name='main'):
    p = argparse.ArgumentParser(name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # [key, default, help, {action_store etc.}]
    for a in args:
        if len(a) == 2:
            a += ['', {}]
        elif len(a) == 3:
            a.append({})
        a[3]['help'] = a[2]

        if type(a[1]) == bool:
            if a[1]:
                a[3]['action'] = 'store_false'
            else:
                a[3]['action'] = 'store_true'
        else:
            a[3]['type'] = type(a[1])
            a[3]['default'] = a[1]

        p.add_argument(a[0], **a[3])
    return vars(p.parse_args())

def build_filename(opt, blacklist=[], marker=''):
    blacklist = blacklist + ['l','h','o','B','g','r',
                    'time','v','g','j','gs']
    o = json.loads(json.dumps(opt))
    for k in blacklist:
        o.pop(k,None)

    t = ''
    if not marker == '':
        t = marker + '_'
    t = t + time.strftime('%b_%d_%H_%M_%S')
    opt['time'] = t
    opt['fname'] = t + '_opt_' + json.dumps(o, sort_keys=True,
                separators=(',', ':'))

def opt_from_filename(s, ext='.log'):
    _s = s[s.find('_opt_')+5:-len(ext)]
    d = json.loads(_s)
    d['time'] = s[s.find('('):s.find(')')][1:-1]
    return d

def gitrev(opt):
    cmds = [['git', 'rev-parse', 'HEAD'],
            ['git', 'status'],
            ['git', 'diff']]
    rs = []
    for c in cmds:
        subp = subprocess.Popen(c,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
        r, _ = subp.communicate()
        rs.append(r)

    rs[0] = rs[0].strip()
    return dict(sha=rs[0], status=rs[1], diff=rs[2])

def accuracy(yh, y, topk=(1,)):
    maxk = max(topk)
    bsz = y.size(0)

    _, pred = yh.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/bsz).item())
    return res

def error(yh, y, topk=(1,)):
    r = [100.0 - a for a in accuracy(yh, y, topk)]
    if len(r) == 1:
        return r[0]
    return r

def one_hot(y, nc):
    yh = th.ByteTensor(y.shape[0], nc).to(y.device).zero_()
    return yh.scatter_(1, y.squeeze(1), 1)

def setup(opt):
    # seed
    s = opt.get('s', 42)
    opt['s'] = s
    np.random.seed(s)
    th.manual_seed(s)

    # gpu
    ngpus = th.cuda.device_count()
    if ngpus:
        g, j = opt.get('g', 0), opt.get('j', 1)
        assert g < ngpus, "opt['g']=%d, %d GPUs detected"%(opt['g'], ngpus)

        opt['g'], opt['j'] = g, j
        gs = opt.get('gs', list(range(g, g+j)))
        opt['gs'] = gs
        if j == 1:
            th.cuda.set_device(g)

        th.cuda.manual_seed_all(s)

def schedule(e, opt, k=None):
    ks = opt.get(k+'s', json.dumps([[opt['B'], opt[k]]]))
    rs = np.array(json.loads(ks))

    interp = interpolate.interp1d(rs[:,0], rs[:,1],
                        kind='zero', fill_value='extrapolate')
    lr = np.asscalar(interp(e))
    if e in rs[:,0]:
        print('[LR] ', lr)
    return lr

def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

class AverageMeters(object):
    def __init__(self, ks):
        self.m = {}
        for k in ks:
            self.m[k] = tnt.meter.AverageValueMeter()
    def add(self, v):
        for k in v:
            assert k in self.m, 'Key not found'
            self.m[k].add(v[k])
    def value(self):
        return {k:self.m[k].value()[0] for k in self.m}
    def reset(self):
        for k in ks:
            self.m[k].reset()

from line_profiler import LineProfiler
def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    return inner
