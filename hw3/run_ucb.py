import sys
import agents
import sklearn.decomposition
import numpy as np
import matplotlib.pyplot as plt
T, d, gamma = sys.argv[1:]
T=int(T); gamma=float(gamma); d=int(d)
sys.stdout = open(f'./outs/T{T}_d{d}_gamma{gamma}.out', 'w')

C = np.load('C.npy')
y = np.load('y.npy')

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