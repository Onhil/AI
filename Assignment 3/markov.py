#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

hidden_states = ['Happy', 'Sad']
observable_states = ['Cooking', 'Crying', 'Sleeping', 'Socializing', 'Watching TV']
x0 = [0.6, 0.4]
state_space = pd.Series(x0, index=hidden_states,name='states')
print(state_space)
#%%

theta = pd.DataFrame(columns=hidden_states, index=hidden_states)
theta.loc[hidden_states[0]] = [0.9, 0.1]
theta.loc[hidden_states[1]] = [0.2, 0.8]

print(theta)
t = theta.values
#%%
phi = pd.DataFrame(columns=observable_states, index=hidden_states)
phi.loc[hidden_states[0]] = [0.1, 0.2, 0.4, 0.0, 0.3]
phi.loc[hidden_states[1]] = [0.3, 0.0, 0.3, 0.3, 0.1]

print(phi)
p = phi.values
#%%

obs_map = {'Cooking':0, 'Crying':1, 'Sleeping':2, 'Socializing':3, 'Watching TV':4}

# Change as needed
T = 30


obs = np.random.randint(5, size = T)

obs = np.array(obs)
obs = np.array([3, 3, 0, 4, 2])
inv_obs_map = dict((v,k) for k, v in obs_map.items())
obs_seq = [inv_obs_map[v] for v in list(obs)]

#%%
def viterbi(pi, a, b, obs):
    
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]
    
    # init blank path
    path = np.zeros(T, dtype = np.int8)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
    
    # init delta and phi 
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    ##print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            ##print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    # find optimal path
    print('-'*50)
    ##print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    #p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        #p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        ##print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi

#%%

path, delta, phis = viterbi(x0,t,p,obs)

state_map = {0:'Happy', 1:'Sad'}
state_path = [state_map[v] for v in path]

print(pd.DataFrame()
 .assign(Observation=obs_seq)
 .assign(Best_Path=state_path))
#%%