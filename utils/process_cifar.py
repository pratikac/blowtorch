# Modified from From Taco Cohen's Github: https://github.com/tscohen/gconv_experiments

import os, argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser(description='Process CIFAR-10/100')
parser.add_argument('--loc',   help='Directory containing cifar-10-batches-py/cifar-100-python',
        type=str, required=True)
parser.add_argument('--dataset',   help='cifar10 | cifar100 | cifar20',
        type=str, required=True)
opt = parser.parse_args()

class PCA(object):
    def __init__(self, D, n_components):
        self.n_components = n_components
        self.U, self.S, self.m = self.fit(D, n_components)

    def fit(self, D, n_components):
        """
        The computation works as follows:
        The covariance is C = 1/(n-1) * D * D.T
        The eigendecomp of C is: C = V Sigma V.T
        Let Y = 1/sqrt(n-1) * D
        Let U S V = svd(Y),
        Then the columns of U are the eigenvectors of:
        Y * Y.T = C
        And the singular values S are the sqrts of the eigenvalues of C
        We can apply PCA by multiplying by U.T
        """

        # We require scaled, zero-mean data to SVD,
        # But we don't want to copy or modify user data
        m = np.mean(D, axis=1)[:, np.newaxis]
        D -= m
        D *= 1.0 / np.sqrt(D.shape[1] - 1)
        U, S, V = np.linalg.svd(D, full_matrices=False)
        D *= np.sqrt(D.shape[1] - 1)
        D += m
        return U[:, :n_components], S[:n_components], m

    def transform(self, D, whiten=False, ZCA=False,
                  regularizer=10 ** (-5)):
        """
        We want to whiten, which can be done by multiplying by Sigma^(-1/2) U.T
        Any orthogonal transformation of this is also white,
        and when ZCA=True we choose:
         U Sigma^(-1/2) U.T
        """
        if whiten:
            # Compute Sigma^(-1/2) = S^-1,
            # with smoothing for numerical stability
            Sinv = 1.0 / (self.S + regularizer)

            if ZCA:
                # The ZCA whitening matrix
                W = np.dot(self.U,
                           np.dot(np.diag(Sinv),
                                  self.U.T))
            else:
                # The whitening matrix
                W = np.dot(np.diag(Sinv), self.U.T)

        else:
            W = self.U.T

        # Transform
        return np.dot(W, D - self.m)


def proc_cifar(loc):
    def _load_batch(fn):
        fo = open(fn, 'rb')
        d = pickle.load(fo , encoding='latin1')
        fo.close()

        if opt.dataset == 'cifar10':
            l = 'labels'
        elif opt.dataset == 'cifar100':
            l = 'fine_labels'
        elif opt.dataset == 'cifar20':
            l = 'coarse_labels'
        else:
            assert False, 'Unknown dataset'
        return d['data'].reshape(-1, 3, 32, 32), d[l]

    def normalize(data, eps=1e-8):
        data -= data.mean(axis=(1, 2, 3), keepdims=True)
        std = np.sqrt(data.var(axis=(1, 2, 3), ddof=1, keepdims=True))
        std[std < eps] = 1.
        data /= std
        return data

    proc_loc = 'proc'
    if os.path.exists(proc_loc):
        print('Found existing proc, delete the proc folder if you want to run again')
        return
    os.mkdir(proc_loc)

    print('[Loading]')
    if opt.dataset == 'cifar10':
        train_fns = [os.path.join(loc, 'data_batch_' + str(i)) for i in range(1, 6)]
        train_batches = [_load_batch(fn) for fn in train_fns]
        test_batch = _load_batch(os.path.join(loc, 'test_batch'))

        tx = np.vstack([train_batches[i][0] for i in range(5)]).astype('float32')
        ty = np.vstack([train_batches[i][1] for i in range(5)]).flatten()
        vx, vy = test_batch[0].astype('float32'), test_batch[1]

    elif opt.dataset == 'cifar100' or opt.dataset == 'cifar20':
        train_batch = _load_batch(os.path.join(loc, 'train'))
        test_batch = _load_batch(os.path.join(loc, 'test'))
        tx, ty = train_batch[0].astype('float32'), train_batch[1]
        vx, vy = test_batch[0].astype('float32'), test_batch[1]
    else:
        assert False, 'Unknown dataset'

    tx, vx = tx/255.0, vx/255.0
    np.savez(os.path.join(proc_loc, opt.dataset+'-train-zero-one.npz'), data=tx, labels=ty)
    np.savez(os.path.join(proc_loc, opt.dataset+'-test-zero-one.npz'), data=vx, labels=vy)

    tx, vx = normalize(tx), normalize(vx)
    txf, vxf = tx.reshape(tx.shape[0], -1).T, vx.reshape(vx.shape[0], -1).T

    np.savez(os.path.join(proc_loc, opt.dataset+'-train.npz'), data=tx, labels=ty)
    np.savez(os.path.join(proc_loc, opt.dataset+'-test.npz'), data=vx, labels=vy)

    print('[Whitening]')
    pca = PCA(D=txf, n_components=txf.shape[1])
    tx = pca.transform(D=txf, whiten=True, ZCA=True).T.reshape(tx.shape)
    vx = pca.transform(D=vxf, whiten=True, ZCA=True).T.reshape(vx.shape)

    print('[Saving]')
    np.savez(os.path.join(proc_loc, opt.dataset+'-train-proc.npz'), data=tx, labels=ty)
    np.savez(os.path.join(proc_loc, opt.dataset+'-test-proc.npz'), data=vx, labels=vy)
    print ('[Finished]')

loc, dataset = opt.loc, opt.dataset
assert os.path.exists(loc), 'loc does not exist: %s' % loc
if dataset == 'cifar10':
    raw_loc = os.path.join(loc, 'cifar-10-batches-py')
elif dataset == 'cifar100' or dataset == 'cifar20':
    raw_loc = os.path.join(loc, 'cifar-100-python')
else:
    assert False, 'Unknown dataset'

assert os.path.exists(raw_loc), 'Could not find data, download the dataset first'

proc_cifar(raw_loc)
