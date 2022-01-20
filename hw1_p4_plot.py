"""Empirical tests of the agents - plots"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv

df.to_csv('p4.1.csv')

fig, ax = plt.subplots()

ax.plot(t, r_ucb, c='tab:red', label='ucb')
ax.plot(t, r_etc1, c='tab:blue', label=f'etc, m={etc1.m}', alpha = .4)
ax.plot(t, r_etc2, c='tab:blue', label=f'etc, m={etc2.m}', alpha = .6)
ax.plot(t, r_etc3, c='tab:blue', label=f'etc, m={etc3.m}')
ax.plot(t, r_gt, c='tab:green', label=f'thompson')
ax.set_xlabel('timestep')
ax.set_ylabel('regret')
plt.legend()
plt.savefig('p4.1.png', bbox_inches='tight', dpi=600)

# Problem 4.2
T = 20000
mus = [1]
for i in range(2,41):
    mus.append(1-1/np.sqrt(i-1))
print(mus)
arms = [scipy.stats.norm(loc=mu) for mu in mus]
    
agents = [
    UCB(T, arms, log_regret_every_n=10),
    ETC(T, arms, m=5, log_regret_every_n=10),
    ETC(T, arms, m=10, log_regret_every_n=10),
    ETC(T, arms, m=100, log_regret_every_n=10),
    GaussianThompson(T, arms, prior_mean=0, prior_var=1, log_regret_every_n=10)
]

ucb, etc1, etc2, etc3, gt = Parallel(n_jobs=5)(delayed(run_agent)(agent) for agent in agents)

# create dataframe
t, r_ucb = np.array(ucb.regret_log).T
_, r_etc1 = np.array(etc1.regret_log).T
_, r_etc2 = np.array(etc2.regret_log).T
_, r_etc3 = np.array(etc3.regret_log).T
_, r_gt = np.array(gt.regret_log).T

df = pd.DataFrame({
    't': t,
    'UCB': r_ucb,
    f'ETC {etc1.m}': r_etc1,
    f'ETC {etc2.m}': r_etc2,
    f'ETC {etc3.m}': r_etc3,
    'GT': r_gt
})

df.to_csv('p4.2.csv')

fig, ax = plt.subplots()

ax.plot(t, r_ucb, c='tab:red', label='ucb')
ax.plot(t, r_etc1, c='tab:blue', label=f'etc, m={etc1.m}', alpha = .4)
ax.plot(t, r_etc2, c='tab:blue', label=f'etc, m={etc2.m}', alpha = .6)
ax.plot(t, r_etc3, c='tab:blue', label=f'etc, m={etc3.m}')
ax.plot(t, r_gt, c='tab:green', label=f'thompson')
ax.set_xlabel('timestep')
ax.set_ylabel('regret')
plt.legend()
plt.savefig('p4.2.png', bbox_inches='tight', dpi=600)
