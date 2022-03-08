import sys
import agents
import sklearn.decomposition
import numpy as np
import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed



def run(T,d,gamma):

    sys.stdout = open(f'./outs/T{T}_d{d}_gamma{gamma}.out', 'w')

    C = np.load('C.npy')
    y = np.load('y.npy')
    
    if d is not None:
        pca = sklearn.decomposition.PCA(d)
        C = pca.fit_transform(C)
    scaler = sklearn.preprocessing.Normalizer()
    C = scaler.fit_transform(C)

    ucb = agents.UCB(C, y, T, gamma=gamma)
    ucb.run()

    fig, ax = plt.subplots()
    ax.plot(*np.array(ucb.R_log).T)
    plt.savefig(f'./figs/T{T}_d{d}_gamma{gamma}.png', bbox_inches='tight')

    sys.stdout.close()


def run_combos():
    gammas = [.1,.5,.9,1.0,1.1,2,10]
    ds = [10, 50, 200, 700, None]
    Ts = [5000]
    perms = itertools.product(Ts, ds, gammas)
    Parallel(n_jobs=10)(delayed(run)(*i) for i in perms)
    return

if __name__ == '__main__':
    run_combos()
    
    

