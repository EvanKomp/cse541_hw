import numpy as np
import sklearn.linear_model

def randargmax(b,**kw):
    return np.argmax(np.random.random(b.shape) * (b==b.max()), **kw)

class Agent:
    
    def __init__(self, C, y, T, n=10, gamma=1.0, log_rate=10):
        self.C = C
        self.y = y
        self.T = T
        self.n = n
        self.gamma = gamma
        self.log_rate = log_rate
        self.R_log = []
        self.R = 0.0
        self.t = 0
        self.d = C.shape[1]
        self.phi_d = int(self.d*n)
        
        self.V = np.eye(self.phi_d) * gamma
        self.V_inv = np.linalg.inv(self.V)
        self.S = np.zeros((self.phi_d,1))
        
        self.startup()
        return
    
    @property
    def theta(self):
        return self.V_inv @ self.S
    
    def startup(self):
        raise NotImplemented()
        
    def phi(self, ind, a):
        """Featurize the action played and the context given.
        
        Parameters
        ----------
        ind : index of context raised
        a : index of action played
        """
        c = self.C[ind]
        out = np.zeros((self.n, c.size))
        out[a] = c
        assert out.size == self.phi_d
        return out.flatten()
    
    def r(self, ind, a):
        """Reward
        
        Parameters
        ----------
        ind : index of context raised
        a : index of action played
        """
        return int(self.y[ind] == a)
    
    def pull(self, ind, a):
        """Commit an agent action. Updates regret.
        
        The problem statement was fixed to have optimal policy always give reward of 1.
        So for a play is just 1 - reward
        
        Parameters
        ----------
        ind : index of context raised
        a : index of action played
        """
        r = self.r(ind, a)
        self.R += 1.0 - r
        
        self.t += 1
        if self.t % self.log_rate == 0:
            self.R_log.append((self.t, self.R))
            
        self.update(ind, a, r)
        # print(f'Pulled arm {a} for context {ind} and recieved reward {r}.')
        return r
    
    def run(self):
        """Run the algorithm until T."""
        
        while self.t < self.T:
            ind_t = np.random.choice(len(self.y))
            at = self.pick(ind_t)
            self.pull(ind_t, at)
        return
    
    def pick(self, ind):
        """Pick arm based on context vector
        
        parameters
        ind - index of the context that was raised
        """
        raise NotImplemented()
        
    def update(self, ind, a, r):
        phis = self.phi(ind,a).reshape((-1,1))
        self.S += r * phis
        self.V += phis @ (phis.T)
        self.V_inv = self.V_inv - (self.V_inv @ phis @ phis.T @ self.V_inv)/(1 + phis.T @ self.V_inv @ phis)
        return
            
        
class ETC_world(Agent):
    
    def __init__(self, C, y, T, tau, n=10, log_rate=10):
        self.tau = tau
        super().__init__(C, y, T, n=n, log_rate=log_rate)
        return
    
    def startup(self):
        
        for i in range(self.tau):
            
            ind_t = np.random.choice(len(self.y))
            at = np.random.choice(self.n)
            r = self.pull(ind_t, at)
        self.theta_lock = np.array(self.theta)
        return
            
    def pick(self, ind):
        phis = [self.phi(ind, a) for a in range(self.n)]
        r_hats = [self.theta_lock.reshape(1,-1) @ phi.reshape(-1,1) for phi in phis]
        return np.argmax(r_hats)
    
    
class ETC_bias(Agent):
    
    def __init__(self, C, y, T, tau, n=10, log_rate=10):
        self.tau = tau
        super().__init__(C, y, T, n=n, log_rate=log_rate)
        return
    
    def startup(self):
        Cs = []
        ys = []
        for i in range(self.tau):
            ind_t = np.random.choice(len(self.y))
            at = np.random.choice(self.n)
            r = self.pull(ind_t, at)
            if r == 1:
                Cs.append(self.C[ind_t])
                ys.append(self.y[ind_t])
        Cs = np.array(Cs)
        ys = np.array(ys).reshape(-1,1)
        
        model = sklearn.linear_model.SGDClassifier()
        model.fit(Cs, ys)
        self.model = model
        return
            
    def pick(self, ind):
        at = self.model.predict(self.C[ind].reshape(1,-1))
        return int(at)
    
    
class FTL(ETC_world):
    
    def pick(self, ind):
        phis = [self.phi(ind, a) for a in range(self.n)]
        r_hats = np.array([self.theta.reshape(1,-1) @ phi.reshape(-1,1) for phi in phis])
        return randargmax(r_hats)
    

class UCB(Agent):
    
    def startup(self):
        self.del_ = 1/self.T
        self.V0 = np.array(self.V)
        return
    
    def pick(self, ind):
        print(f'Context {ind} revealed')
        beta = np.sqrt(self.gamma) + np.sqrt(2*np.log(1/self.del_)+np.log(np.linalg.det(self.V)/np.linalg.det(self.V0)))
        theta = self.theta
        phis = np.array([self.phi(ind, a) for a in range(self.n)])
        r_hats = phis @ theta
        bound = []
        for phi in phis:
            bound.append(beta * np.sqrt(phi.reshape(1,-1) @ self.V_inv @ phi.reshape(-1,1)))
        bound = np.array(bound).reshape(-1,1) 
        print('Beta: ', beta)
        print('Predicted reward:confidence')
        print(np.concatenate([r_hats, bound], axis=1))
        a = randargmax(r_hats + bound)
        print(f'Chose arm {a}')
        return a
    