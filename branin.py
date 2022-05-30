from pickle import NONE
import numpy as np
import math
import pdb


# https://www.sfu.ca/~ssurjano/branin.html 


lb = np.hstack((-5 * np.ones(500), 0 * np.ones(500)))   # lower bounds for input domain
ub = np.hstack((10 * np.ones(500), 15 * np.ones(500)))  # upper bounds for input domain
    

def branin(x):
    # assume x is in [0,1]x[0,1]
    pdb.set_trace()
    assert x.shape == (2,), 'input must have shape (2,)'
    x1, x2 = -5+15*x[0], 15*x[1]  # transform from [0,1] to original domain
    t1 = x2 - 5.1 / (4 * math.pi ** 2) * x1 ** 2 + 5 / math.pi * x1 - 6
    t2 = 10 * (1 - 1 / (8 * math.pi)) * np.cos(x1)
    return t1 ** 2 + t2 + 10

def branin100(x, embedding_idx = [17, 87]):
    if len(x.shape) > 1:
        res = []
        for i in range(x.shape[0]):
            res.append(branin(x[i, embedding_idx]))
        print(res)
        return np.array(res)
    else:
        return branin(x[embedding_idx])

def branin_1000(x, embedding_idx = [17, 876]):
    # assert (x <= ub).all() and (x >= lb).all()
    return branin(x[embedding_idx])


if __name__ == '__main__':
    print(((lb+ub)/2).shape)
    print(branin_1000((lb + ub)/2))