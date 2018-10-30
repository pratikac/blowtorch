import torch as th
import torch.optim as optim

import numpy as np
import scipy.io as sio
import torch.nn as nn
import torchnet as tnt
import torchvision.transforms as transforms
from torchvision import datasets

import blowtorch.cvtransforms as cv
import cv2, os, sys, pdb

DATA = '/data/'

def get_imagenet_loader(opt, is_train=True, loc=DATA):
    loc = loc + 'imagenet'
    nw = max(opt['j'], 4)

    input_transform = [transforms.Resize(256)]
    normalize = [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]

    if is_train:
        traindir = os.path.join(loc, 'train')
        train_folder = datasets.ImageFolder(traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip()] + normalize))
        return th.utils.data.DataLoader(
            train_folder,
            batch_size=opt['b'], shuffle=True,
            num_workers=nw, pin_memory=True)
    else:
        valdir = os.path.join(loc, 'val')
        val_folder = datasets.ImageFolder(valdir,
                transforms.Compose(
                    input_transform +
                    [transforms.CenterCrop(224)] + normalize))
        return th.utils.data.DataLoader(
            val_folder,
            batch_size=opt['b'], shuffle=False,
            num_workers=nw, pin_memory=True)

def get_loader(opt, is_train=True, loc=DATA):
    is_train_str = 'train' if is_train else 'test'
    dataset = opt['dataset']

    if not dataset == 'imagenet':
        if dataset == 'mnist':
            ds = np.load(loc+'/mnist/mnist-'+is_train_str+'-zero-one.npz')
        elif dataset in ['cifar10', 'cifar100', 'cifar20']:
            ds = np.load(loc+'cifar/' + dataset + '-'+is_train_str+'-zero-one.npz')
        elif dataset == 'svhn' or dataset == 'svhnx':
            if is_train:
                d1 = sio.loadmat(loc + 'svhn/train_32x32.mat')
                ds = dict(data=d1['X'], labels=d1['y'])
                if dataset == 'svhnx':
                    d2 = sio.loadmat(loc + 'svhn/extra_32x32.mat')
                    ds = dict(data=np.concatenate([d1['X'],d2['X']], axis=3),
                            labels=np.concatenate([d1['y'],d2['y']]))
            else:
                d3 = sio.loadmat(loc + 'svhn/test_32x32.mat')
                ds = dict(data=d3['X'], labels=d3['y'])

            ds['data'] = np.transpose(ds['data'], (3,2,0,1)).astype(np.float32)/255.
            ds['labels'] = (ds['labels']-1).astype(np.int).squeeze()
        else:
            assert False, 'Unknown dataset: %s'%dataset

        data, labels = ds['data'], ds['labels']
        nfrac = int(len(data)*opt['frac'])
        data, labels = data[:nfrac], labels[:nfrac]

        tds = tnt.dataset.TensorDataset([data, labels])

        augment = opt.get('augment', True)
        augment = augment and is_train
        if dataset == 'mnist':
            augment = False

        sz = data.shape[-1]
        transforms = tnt.transform.compose([
            lambda x: x.transpose(1,2,0),
            cv.RandomHorizontalFlip(),
            cv.Pad(4, borderType=cv2.BORDER_REFLECT),
            cv.RandomCrop(sz),
            lambda x: x.transpose(2,0,1),
            th.from_numpy
            ])

        if is_train and augment:
            tds = tnt.dataset.TransformDataset(tds, {0:transforms})
        return tds.parallel(batch_size=opt['b'],
                            num_workers=max(opt['j'], 2), shuffle=is_train,
                            pin_memory=True)
    else:
        return get_imagenet_loader(opt, is_train=is_train, loc=loc)
