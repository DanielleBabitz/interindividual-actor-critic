#imports
from itertools import count

from functions import *
import os.path

#create all result dirs

dirs = ['results', 'results/model', 'results/prosociality', 'results/ingroup_favoritism', 'results/loss_aversion',
     'results/stickiness', 'results/self_reinforcement', 'results/local_global']

for dir_path in dirs:
     if not os.path.exists(dir_path):
         os.makedirs(dir_path)
         print(f"Directory '{dir_path}' created.")
     else:
         print(f"Directory '{dir_path}' already exists.")

#RUN SIMULATIONS

#fig 1 - individual actor-critic

#set up hyperparams
HP = {'trials':1000, 'nt':150, 'n_actions':2, 'lrs':0.1,'lrp':0.1, 'n_agents': 1,
   'ratio':0.5, 'dr':1, 'n_states':1, 'la':0, 'actions': ['1','2'], 'rewards': [5,0]}

#run simulation
policy, V, advantage = run_all_trials(HP, model = 'individual_actor_critic')

#plot and save figures
# #mean results
plot_individual_mean_curves(HP['rewards'],HP['nt'], policy, V, advantage)

#fig 2 - interindividual actor-critic

#set up hyperparams
HP = {'trials':500, 'nt':600, 'n_actions':2, 'lrs':0.1,'lrp':0.1,
   'ratio':0.5, 'dr':1, 'n_states':2, 'la':0, 'n_agents':2,
   'actions':['selfish','coop'],
   'rewards':{'selfish':[5,-10],'coop':[0,0]}
   }

#run simulation - 2 agents with large and small phi

#create adjacency matrix
adj_mat = np.ones((HP['n_agents'],HP['n_agents'])) - np.eye(HP['n_agents'])  #all-to-all connected advacency matrix
#create reward matrix
r_mat = create_reward_mat(HP['rewards'], HP['actions'], adj_mat)

phi = [0.25,0.8]
policy = []
V = []
advantage = []

for p in phi:
   HP['ratio'] = p
   policy_all, V_all, advantage_all = run_all_trials(HP, adj_mat = adj_mat, reward_mat=r_mat)
   policy.append(policy_all)
   V.append(V_all)
   advantage.append(advantage_all)

#plot results
rewards = [HP['rewards']['selfish'][0], HP['rewards']['selfish'][1]]
t = int(HP['nt']/HP['n_agents'])
for i in range(len(phi)):
    filename = 'model/interindividual_actor_critic_phi_' + str(phi[i])
    plot_interindividual_mean_curves(t, rewards, V[i], advantage[i], policy[i], filename)

#fig 5 - loss aversion and established norms

#set up hyperparams
HP = {'trials':500, 'nt':600, 'n_actions':2, 'lrs':0.1,'lrp':0.1,
   'ratio':0.8, 'dr':1, 'n_states':2, 'la':0, 'n_agents':2,
   'actions':['selfish','coop'],
   'rewards':{'selfish':[5,-10],'coop':[0,0]}
   }

#run simulations
adj_mat = np.ones((HP['n_agents'],HP['n_agents'])) - np.eye(HP['n_agents'])
r_mat = create_reward_mat(HP['rewards'], HP['actions'], adj_mat)
policy = []
V = []
advantage = []

#B - no established norm
HP['la'] = 0.5
policy_no_norm, V_no_norm, advantage_no_norm = run_all_trials(HP, adj_mat = adj_mat, reward_mat=r_mat)
policy.append(policy_no_norm)
V.append(V_no_norm)
advantage.append(advantage_no_norm)

#C - established norm (play music)
V0 = [[-9,4.5],[-9,4.5]]
theta0 = [[1.09861232,-1.09861226],[1.09861232,-1.09861226]] #0.9, 0.1

#1 - with loss aversion
policy_norm_la, V_norm_la, advantage_norm_la = run_all_trials(HP, adj_mat = adj_mat, reward_mat=r_mat, thetas_init=theta0, V_init=V0)
policy.append(policy_norm_la)
V.append(V_norm_la)
advantage.append(advantage_norm_la)

#2 - no loss aversion
HP['la'] = 0
policy_norm_no_la, V_norm_no_la, advantage_norm_no_la = run_all_trials(HP, adj_mat = adj_mat, reward_mat=r_mat, thetas_init=theta0, V_init=V0)
policy.append(policy_norm_no_la)
V.append(V_norm_no_la)
advantage.append(advantage_norm_no_la)

#3 - reward feedback
HP['la'] = 0
policy_norm_la_r_feedback, V_norm_la_r_feedback, advantage_norm_la_r_feedback = run_all_trials(HP, adj_mat = adj_mat, reward_mat=r_mat, feedback='reward')
policy.append(policy_norm_la_r_feedback)
V.append(V_norm_la_r_feedback)
advantage.append(advantage_norm_la_r_feedback)

#plot results
rewards = [HP['rewards']['selfish'][0], HP['rewards']['selfish'][1]]
t = int(HP['nt']/HP['n_agents'])
filenames = ['no_norm','norm_la','norm_no_la','norm_la_r_feedback']
for i in range(len(filenames)):
 filename = 'loss_aversion/'+filenames[i]
 plot_interindividual_mean_curves(t, rewards, V[i], advantage[i], policy[i], filename)

#fig 3 - prosociality

#set up hyperparams

import networkx as nx

#Prosociality - single examples from the parameter space:
#parameters
HP = {'trials':100, 'nt':2000, 'n_actions':2, 'lrs':0.15,'lrp':0.15,
   'ratio':0.5, 'dr':1, 'n_states':2, 'la':0, 'n_agents': 20, 'd': 8,
   'actions':['selfish','coop'],
   'rewards':{'selfish':[5,-0.5],'coop':[0.5,0.5]},  #[self, other]
   }

degrees = [8,11]
n=20
ratio = [0.4,0.5,0.6]

#run simulation
policy_list = []

for d in degrees:
 HP['d'] = d
 for phi in ratio:
   HP['ratio'] = phi
   policy_all, V_all, advantage_all = run_all_trials(HP, connections='random')
   policy_list.append(policy_all)

#plot results

#histogram
num_bins=20
fig, ax = plt.subplots(figsize=(5,5))
labels = ['neighbors=8, $\phi$=0.4','neighbors=8, $\phi$=0.5','neighbors=8, $\phi$=0.6','neighbors=11, $\phi$=0.4','neighbors=11, $\phi$=0.5','neighbors=11, $\phi$=0.6']
colors = ['8cc5e3','336E94','003366','FF8CA0','D15C61','9B1B30']

for i in range(len(policy_list)):
 data = policy_list[i]
#Create histograms for each simulation
 histograms = [np.histogram(data[i,-1,:,1], bins=20, range=(0, 1), density=True)[0] for i in range(HP['trials'])]
#Calculate the average histogram
 average_histogram = np.mean(histograms, axis=0)
 plt.bar(np.linspace(0, 1, 20), average_histogram/n, width=0.05, color=colors[i],label = labels[i], alpha = 0.8)
 plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
 plt.tight_layout()
 plt.grid(False)

plt.ylabel('Proportion of agents')
plt.xlabel('$\pi_{\t{prosocial}}$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig('results/prosociality/hist'+'.svg', format='svg', bbox_inches='tight')
plt.show()

#policy curves

#Calculate average policy learning for the first action over all simulations and agents for each combination
fig, ax = plt.subplots(figsize=(5,5))

for i in range(len(policy_list)):
    average_policy_learning = np.mean(policy_list[i][ :, :, :, 1], axis=(0,2))
    std_policy_learning = np.std(policy_list[i][ :, :, :, 1], axis=(0,2))
    plt.plot(range(len(average_policy_learning)), average_policy_learning, label=labels[i], color=colors[i])

    plt.fill_between(range(len(average_policy_learning)),
              average_policy_learning - std_policy_learning,
              average_policy_learning + std_policy_learning,
              alpha=0.3, color=colors[i])

#Customize plot
plt.xlabel('Round')
plt.ylabel('$\pi_{\t{prosocial}}$')
n_rounds = HP['nt']/n
plt.xticks(ticks=np.linspace(0,len(average_policy_learning),6), labels=np.arange(0,n_rounds+1,20, dtype=int))
plt.legend()
plt.xlim(0,len(average_policy_learning)-1)
plt.ylim(0,1)
plt.grid(False)
plt.tight_layout()
plt.savefig('results/prosociality/policy'+'.svg', format='svg', bbox_inches='tight')
plt.show()

#run simulation - heatmaps

n=20
HP['trials'] = 500
degree = np.arange(0,n)
phi = np.arange(0,1.05,0.05)

res_array_mean_policy = np.zeros((len(degree),len(phi)))
res_array_std_policy_within = np.zeros((len(degree),len(phi)))  #take the mean of the std within agents in each simulation

for d in range(len(degree)):
 policy_list = []
 res_list = []
 for l in range(len(phi)):
   HP['ratio'] = phi[l]
   HP['nt'] = n*50
   HP['d'] = degree[d]
   policy = run_all_trials(HP, connections='random')[0]
   mean_policy = np.mean(policy[:,-1,:,1], axis=(1)) #mean policy of each simulation
   mean_policy_all = np.mean(mean_policy) #mean policy across all simulations
   std_policy_within = np.mean(np.std(policy[:,-1,:,1], axis=(1)))  #mean sd across all simulations
   res_array_mean_policy[d,l] = mean_policy_all
   res_array_std_policy_within[d,l] = std_policy_within

#save results
np.save('results/prosociality/res_array_mean_policy.npy', res_array_mean_policy)
np.save('results/prosociality/res_array_std_policy_within.npy', res_array_std_policy_within)

#plot results in a matrix with colorbar indicating the proportion of selfish agents

from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size': 12})   #Set the desired font size

phi = np.round(phi,decimals=2)

plt.imshow(res_array_mean_policy[:,:].T, cmap='Blues', vmin=0, vmax=1)
cbar = plt.colorbar()
cbar.set_label('$\pi_{\t{prosocial}}$ mean')

#Set x-axis ticks and labels
plt.xticks(np.arange(len(degree)), degree)
plt.xlabel('Neighbors')

#Set y-axis ticks and labels
plt.yticks(np.arange(len(phi)), phi)
plt.ylabel(r"Weight of others' evaluations $(\phi)$ ")

plt.grid(False)
plt.gca().invert_yaxis()

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))   #Max 5 ticks on x-axis
plt.yticks([0,4,8,12,16,20],['0','0.2','0.4','0.6','0.8','1'])

#set colorbar ticks to start from 0
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar.set_ticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])

plt.savefig('results/prosociality/mat_mean'+'.svg', format='svg', bbox_inches='tight')
plt.show()

plt.imshow(res_array_std_policy_within.T, cmap='Blues', vmin=0, vmax=0.1)
cbar = plt.colorbar()
cbar.set_label('$\pi_{\t{prosocial}}$ sd')
#Set x-axis ticks and labels
plt.xticks(np.arange(len(degree)), degree)
plt.xlabel('Neighbors')

#Set y-axis ticks and labels
plt.yticks(np.arange(len(phi)), phi)
plt.ylabel(r"Weight of others' evaluations $(\phi)$ ")

plt.grid(False)
plt.gca().invert_yaxis()

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))   #Max 5 ticks on x-axis
plt.yticks([0,4,8,12,16,20],['0','0.2','0.4','0.6','0.8','1'])

#set colorbar ticks
cbar.set_ticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
cbar.set_ticklabels([0, 0.02, 0.04, 0.06, 0.08, 0.1])

plt.savefig('results/prosociality/mat_sd'+'.svg', format='svg', bbox_inches='tight')

plt.show()


#plot results

#fig 4 - ingroup favoritism

#set up hyperparams
HP = {'trials':100, 'nt':3200, 'n_actions':2, 'lrs':0.15,'lrp':0.15,
  'ratio':0.5, 'dr':1, 'n_states':2, 'la':0, 'n_agents':0,
  'actions':['ingroup','universal'],
  'rewards':{'ingroup':[0.75,-0.25],'universal':[0.5,0.5]},
  }

nodes_per_group = 16
n_ingroup = [12,9]
n_outgroup = [2,5]
phi = [0.2,0.4,0.9]
n_groups = 2
HP['n_agents'] = nodes_per_group*n_groups

#run simulation
policy_all = []

#create partition list
partition = {}
for i in range(HP['n_agents']):
    group_id = i // nodes_per_group
    partition[i] = group_id

for i in range(len(n_ingroup)):
 n_in = n_ingroup[i]
 n_out = n_outgroup[i]
 adj_matrix = create_symmetric_adjacency_matrix_n_groups(nodes_per_group, n_in, n_out, n_groups)
 partition_list = list(partition.values())
 rewards = list(HP['rewards'].values())
 reward_matrix = create_reward_matrix_n_groups(adj_matrix,partition_list,rewards)
 for i in range(reward_matrix.shape[0]):
     np.fill_diagonal(reward_matrix[i], 0)
 reward_matrix = np.transpose(reward_matrix, (1, 2, 0))

 for j in range(len(phi)):
     HP['ratio'] = phi[j]
     policy, V, advantage = run_all_trials(HP,adj_mat=adj_matrix,reward_mat=reward_matrix)
     policy_all.append(policy)

#plot results

#final policy histogram

labels = ['83%, $\phi$=0.2','83%, $\phi$=0.4', '83%, $\phi$=0.9', '66%, $\phi$=0.2','66%, $\phi$=0.4', '66%, $\phi$=0.9']
plt.rcParams.update({"font.size": 16})
colors = ['8cc5e3','336E94','003366','FF8CA0','D15C61','9B1B30']

fig, ax = plt.subplots(figsize=(7,5))

for i in range(len(policy_all)):

    data = policy_all[i]
    #Create histograms for each simulation
    histograms = [np.histogram(data[i,-1,:,1], bins=20, range=(0, 1), density=True)[0] for i in range(HP['trials'])]

    #Calculate the average histogram
    average_histogram = np.mean(histograms, axis=0)
    plt.bar(np.linspace(0, 1, 20), average_histogram/HP['n_agents'], width=0.05, color=colors[i],label = labels[i])
    plt.ylabel('Proportion of agents')
    plt.xlabel('$\pi_{\t{universal}}$')
    #Plot the average histogram
    plt.grid(False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylim(0,1)
plt.xlim(0,1)
plt.tight_layout()
plt.savefig('results/ingroup_favoritism/hist.svg', format='svg', bbox_inches='tight')
plt.show()

#policy curves
#Calculate average policy learning for the first action over all simulations and agents for each combination
fig, ax = plt.subplots(figsize=(5,5))

for i in range(len(policy_all)):
 average_policy_learning = np.mean(policy_all[i][ :, :, :, 1], axis=(0,2))
 std_policy_learning = np.std(policy_all[i][ :, :, :, 1], axis=(0,2))
 plt.plot(range(len(average_policy_learning)), average_policy_learning, label=labels[i], color=colors[i])

 plt.fill_between(range(len(average_policy_learning)),
              average_policy_learning - std_policy_learning,
              average_policy_learning + std_policy_learning,
              alpha=0.3, color=colors[i])

plt.xlabel('Round')
plt.ylabel('$\pi_{\t{universal}}$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylim(0,1)
plt.xlim(0,100)
plt.tight_layout()
plt.grid(False)
plt.savefig('results/ingroup_favoritism/policy_curve.svg', format='svg', bbox_inches='tight')
plt.show()

#heatmaps

HP['trials'] = 100
HP['nt'] = 4800
n_ingroup = [14,13,12,11,10,9,8,7]
n_outgroup = [0,1,2,3,4,5,6,7]
phi = np.arange(0,1.05,0.05)

policy_all = []

res_array_mean_policy = np.zeros((len(n_ingroup),len(phi)))
res_array_std_policy_within = np.zeros((len(n_ingroup),len(phi)))

for i in range(len(n_ingroup)):
 n_in = n_ingroup[i]
 n_out = n_outgroup[i]
 adj_matrix = create_symmetric_adjacency_matrix_n_groups(nodes_per_group, n_in, n_out, n_groups)
 partition_list = list(partition.values())
 rewards = list(HP['rewards'].values())
 reward_matrix = create_reward_matrix_n_groups(adj_matrix,partition_list,rewards)
 for r in range(reward_matrix.shape[0]):
     np.fill_diagonal(reward_matrix[r], 0)
 reward_matrix = np.transpose(reward_matrix, (1, 2, 0))

 for j in range(len(phi)):
     HP['ratio'] = phi[j]
     policy = run_all_trials(HP,adj_mat=adj_matrix,reward_mat=reward_matrix)[0]
     policy_all.append(policy)
     mean_policy = np.mean(policy[:,-1,:,1], axis=1) #mean policy of each simulation
     mean_policy_all = np.mean(mean_policy) #mean policy across all simulations
     std_policy_within = np.mean(np.std(policy[:,-1,:,1], axis=1))  #mean sd across all simulations
     res_array_mean_policy[i,j] = mean_policy_all
     res_array_std_policy_within[i,j] = std_policy_within

#plot results in a matrix with colorbar indicating the proportion of selfish agents

phi = np.array([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
n_ingroup = [14,13,12,11,10,9,8,7]
ingroup_ratio = (np.array(n_ingroup) / np.max(n_ingroup)) * 100
ingroup_ratio = np.round(ingroup_ratio,decimals=0).astype(int)
outgroup_ratio = 100 - ingroup_ratio

fig, ax = plt.subplots(figsize=(6.4, 5))
plt.imshow(res_array_mean_policy.T,cmap='Blues', aspect='auto', vmin=0, vmax=1)
cbar = plt.colorbar()
cbar.set_label('$\pi_{\t{universal}}$ mean')

#Set x-axis ticks and labels
plt.xticks(np.arange(len(ingroup_ratio)), outgroup_ratio)
plt.xlabel('% Outgroup neighbors')
#Set y-axis ticks and labels
plt.yticks(np.arange(len(phi)), phi)
plt.ylabel(r"Weight of others' evaluations $(\phi)$ ")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yticks([0,4,8,12,16,20],['0','0.2','0.4','0.6','0.8','1'])
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()

plt.grid(False)
plt.savefig('results/ingroup_favoritism/heatmap_mean_left'+'.svg', format='svg', bbox_inches='tight')

plt.show()

fig, ax = plt.subplots(figsize=(6.4, 5))
plt.imshow(res_array_std_policy_within.T, cmap='Blues', aspect='auto')
cbar = plt.colorbar()
cbar.set_label('$\pi_{\t{universal}}$ sd')
plt.xticks(np.arange(len(ingroup_ratio)), outgroup_ratio)
plt.xlabel('% Outgroup neighbors')
plt.yticks(np.arange(len(phi)), phi)
plt.ylabel(r"Weight of others' evaluations $(\phi)$ ")

plt.gca().invert_yaxis()
plt.gca().invert_xaxis()

plt.grid(False)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.yticks([0,4,8,12,16,20],['0','0.2','0.4','0.6','0.8','1'])

plt.savefig('results/ingroup_favoritism/heatmap_sd_left'+'.svg', format='svg', bbox_inches='tight')
plt.show()


#fig 6 - stickiness

#set up hyperparams
#initialize thetas as average of p(smoking in piblic) ~0.9
thetas0 = np.array([[2.00964741, -0.83528764],[2.09931656, -0.87662711], [2.0289205,  -0.90056976],[1.90618188, -0.66281248],[1.86390116, -0.58256621]])
HP = {'trials':300, 'nt':3000, 'n_actions':2, 'lrs':0.2,'lrp':0.2,
   'ratio':0.5, 'dr':1, 'n_states':2, 'la':0, 'n_agents':5,
   'actions':['selfish','coop'],
   'rewards':{'selfish':[5,-1],'coop':[0,0]},
   }

new_rewards_other = [-2.5,-3.6,-4.5]
loss_aversion = [0,0.3,0.5]
n_rounds = int(HP['nt']/HP['n_agents'])
V_init = np.zeros((HP['n_agents'],HP['n_states']))
V_init[:,1] = HP['rewards']['selfish'][0] * 0.9
adj_mat = np.ones((HP['n_agents'],HP['n_agents'])) - np.eye(HP['n_agents'])  #all-to-all connected

#run simulation - multiple values of loss aversion

#advantage as feedback
policy = np.zeros((len(new_rewards_other),len(loss_aversion),HP['trials'],n_rounds,HP['n_agents'],HP['n_actions']))
V = np.zeros((len(new_rewards_other),len(loss_aversion),HP['trials'],HP['nt'],HP['n_agents'],HP['n_states']))
advantage = [[] for r in new_rewards_other]

#reward as feedback
policy_rfeedback = np.zeros((len(new_rewards_other),len(loss_aversion),HP['trials'],n_rounds,HP['n_agents'],HP['n_actions']))

for r in range(len(new_rewards_other)):
 for l in range(len(loss_aversion)):
     HP['rewards']['selfish'][1] = new_rewards_other[r]
     V_init[:,0] = HP['rewards']['selfish'][1]   #agents learn the new negative reward
     r_mat = create_reward_mat(HP['rewards'], HP['actions'], adj_mat)
     HP['la'] = loss_aversion[l]
     policy_all, V_all, adv_all = run_all_trials(HP, adj_mat=adj_mat, reward_mat=r_mat, thetas_init=thetas0, V_init=V_init, turns='random')
     policy_all_rfeedback, V_all_rfeedback, adv_rfeedback = run_all_trials(HP, adj_mat=adj_mat, reward_mat=r_mat, thetas_init=thetas0, V_init=V_init,
                                                                           turns='random', feedback='reward')
     policy[r,l] = policy_all
     V[r,l] = V_all
     advantage[r].append(adv_all)
     policy_rfeedback[r,l] = policy_all_rfeedback

#plot results
blue_colors = ['BDDEFF','79BCFF','005AB4']
rounds = int(HP['nt']/HP['n_agents'])-1
plt.rcParams.update({"font.size": 14})

#plot policy curves for all condtions (3 r_other, 3 loss aversion values)

p_list = [policy, policy_rfeedback]
labels = ['policy','policy_rfeedback']

for p in range(len(p_list)):
 fig, ax = plt.subplots(1,len(new_rewards_other), sharey=True, sharex=True, figsize=(10,3))
 for r in range(len(new_rewards_other)):
     for l in range(len(loss_aversion)):
       mean_policy = np.mean(p_list[p][r,l,:,:-1,:,0], axis=(0,2))
       ax[r].plot(np.arange(rounds),mean_policy,label='l = '+str(loss_aversion[l]),color=blue_colors[l], linewidth=2)
       for spine in ax[r].spines.values():
         spine.set_edgecolor('black')
       ax[r].grid(False)
 plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))   #Centered vertically, slightly to the right
 ax[0].set_ylabel('Policy',fontsize=14)
 ax[1].set_xlabel('Round',fontsize=14)
 plt.xlim(0,rounds+1)
 plt.ylim(0,1)
 plt.tight_layout()
 plt.savefig('results/stickiness/'+labels[p]+'.svg', format='svg')
 plt.show()

#plot advantage for all conditions

fig, axes = plt.subplots(1,len(new_rewards_other), sharey=True, sharex=True, figsize=(10, 3))
rounds = int(HP['nt']) - 5
plt.rcParams.update({"font.size": 16})
t=np.arange((HP['nt']-50)//HP['n_agents'])

for r in range(len(new_rewards_other)):
 adv = np.array(advantage[r])
 for l in range(len(loss_aversion)):
     mean_adv_ns = np.nanmean(adv[l, :, :-50, :, 1, 0], axis=(2, 0))[::5]*(HP['n_agents']-1)*HP['ratio'] #advantage of not smoking in public, non-smokers (other acting)
     axes[r].plot(t, mean_adv_ns, linestyle= (0, (5, 10)), label='l = '+str(loss_aversion[l]),color=blue_colors[l], linewidth=2, dashes=[5,5])
     axes[r].axhline(y = 5, color = 'black', linewidth=1, linestyle='--')
     for spine in axes[r].spines.values():
         spine.set_edgecolor('black')
     axes[r].grid(False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))   #Centered vertically, slightly to the right
axes[0].set_ylabel('A (no public smoking)')
fig.text(0.45, -0.06, 'Round')
fig.text(0.91, 0.53, '$r_{smokers}$')
plt.xlim(0,len(t)+10)
plt.tight_layout()

plt.savefig('results/stickiness/advantage.svg', format='svg')
plt.show()

#fig 7 - self-reinforcement

#set up hyperparams
#self reinforcement: sampling 10 subgroups of 10 out of a population of n=100 agents

HP = {'trials': 1, 'nt':30000, 'n_actions':2, 'lrs':0.15,'lrp':0.15,
   'ratio':1, 'dr':1, 'n_states':2, 'la':0, 'n_agents':100, 'd':10,
   'actions':['tipping','not tipping'],
   'rewards':{'group1':[1,0],'group2':[0,1]},
   }

#create reward mat - half prefer a1, half prefer a2, no self-rewards
reward_mat = np.zeros((HP['n_agents'],HP['n_agents'],HP['n_actions']))
group_size = int(HP['n_agents']/2)
reward_mat[:,0:group_size,0] = HP['rewards']['group1'][0]
reward_mat[:,0:group_size,1] = HP['rewards']['group1'][1]
reward_mat[:,group_size:,0] = HP['rewards']['group2'][0]
reward_mat[:,group_size:,1] = HP['rewards']['group2'][1]
np.fill_diagonal(reward_mat[:,:,0],0)
np.fill_diagonal(reward_mat[:,:,1],0)

#run simulation
policy_no_la = run_all_trials(HP, reward_mat=reward_mat,turns='random',connections='groups')[0]
HP['la'] = 0.5
policy_la = run_all_trials(HP, reward_mat=reward_mat,turns='random',connections='groups')[0]
policy_la_rfeedback = run_all_trials(HP, reward_mat=reward_mat,turns='random',connections='groups', feedback='reward')[0]

#plot results

#plot examples - same figure

p_list = [policy_no_la, policy_la, policy_la_rfeedback]
n_rounds = int(HP['nt']/HP['n_agents'])
titles = ['tipping','not tipping']
partition_list = list(reward_mat[0,:,0])
color_list = ['#9EAFE6' if partition_list[i] == 0 else '#E69EC9' for i in range(len(partition_list))]
labels = ['policy_no_la', 'policy_la', 'policy_la_rfeedback']

for p in range(len(p_list)):
 fig,ax = plt.subplots(1,figsize=(3.5,3.5))
 for j in range(0,HP['n_agents'],3):
     plt.plot(p_list[p][0,:-1,j,0],color = color_list[j])
 plt.xlabel('Round')
 plt.ylabel('$\pi_{\t{tipping}}$')
 plt.ylim(0,1)
 plt.xlim(-1,n_rounds)
 plt.xticks(np.arange(0, n_rounds+1, 100))
 plt.grid(False)
 plt.savefig('results/self_reinforcement/'+labels[p]+'.svg', format='svg', bbox_inches='tight')
 plt.show()

# fig 8 - local conformity/global diversity

#set up hyperparams

HP = {'trials':1, 'nt':30000, 'n_actions':2, 'lrs':0.15,'lrp':0.15,
  'ratio':0.5, 'dr':1, 'n_states':2, 'la':0.5, 'n_agents': 200,
  'actions':['tipping','not tipping'],
  'rewards':{'tipping':[1,0],'not tipping':[0,1]},
  'n_ingroup': 50, 'n_outgroup': 2, 'group_size': 100, 'n_groups':2}

partition_list = []
for group_number in range(HP['n_groups']):
 partition_list.extend([group_number] * HP['group_size'])

# #run simulation
policy_la = run_all_trials(HP, turns='random',connections='local_global')[0]
policy_la_rfeedback = run_all_trials(HP, turns='random',connections='local_global', feedback='reward')[0]
HP['la'] = 0
policy_no_la = run_all_trials(HP, turns='random',connections='local_global')[0]

#plot results
p_list = [policy_no_la, policy_la, policy_la_rfeedback]
n_rounds = int(HP['nt']/HP['n_agents'])
titles = ['tipping','not tipping']
color_list = ['9EAFE6' if partition_list[i] == 0 else 'E69EC9' for i in range(len(partition_list))]
labels = ['policy_no_la', 'policy_la', 'policy_la_rfeedback']

for p in range(len(p_list)):
    fig,ax = plt.subplots(1,figsize=(3.5,3.5))
for j in range(0,HP['n_agents'],3):
    plt.plot(p_list[p][0,:-1,j,0],color = color_list[j])
plt.xlabel('Round')
plt.ylabel('$\pi_{\t{tipping}}$')
plt.ylim(0,1)
plt.xlim(-1,n_rounds)
plt.xticks(np.arange(0, n_rounds+1, 100))
plt.grid(False)
plt.savefig('results/local_global/'+labels[p]+'.svg', format='svg', bbox_inches='tight')
plt.show()

#run simulation - group differences curve in 8B

import pickle

n_in = [52,51,50,49,48,47,46,45,44,43,42]
n_out = [0,1,2,3,4,5,6,7,8,9,10]

count_diff_group_results = np.zeros(len(n_in))
means_diff = np.zeros(len(n_in))

HP['la'] = 0.5
HP['trials'] = 100
n_repetitions = 50

#run simulations for each pair of (n_in, n_out), total of 400 trials in batches of 100.
for i in range(len(n_in)):
    #set parameters
    HP['n_ingroup'] = n_in[i]
    HP['n_outgroup'] = n_out[i]

    partition = []
    for group_number in range(HP['n_groups']):
        partition.extend([group_number] * HP['group_size'])
    partition = np.array(partition)

    #run batches of 100 trials and calculate the % of independent group results
    #save results to be able to extract count_diff_group_results from the data without running the simulation again
    for j in range(n_repetitions):
        filename = str(HP['n_ingroup'])+'_'+str(HP['n_outgroup'])+'_'+str(j)
        if not os.path.exists('results/local_global'+filename+'.pkl'):
            policy = run_all_trials(HP,connections='local_global')[0]
            mean_values = np.mean(policy[:, -1, :, 0], axis=1)
            #Create a boolean mask where the mean is within the range [0.2, 0.8]
            within_range_mask = (mean_values >= 0.25) & (mean_values <= 0.75)
            # Count the number of instances where the condition is True
            count_within_range = np.sum(within_range_mask)
            count_diff_group_results[i] += count_within_range
            with open('results/local_global'+filename+'.pkl', 'wb') as f:
                pickle.dump(policy, f)
            del policy
        else:
            # If file exists, open and load the file content
            with open('results/local_global' + filename + '.pkl', 'rb') as f:
                policy = pickle.load(f)
            mean_values = np.mean(policy[:, -1, :, 0], axis=1)
            # Create a boolean mask where the mean is within the range [0.25, 0.75]
            within_range_mask = (mean_values >= 0.3) & (mean_values <= 0.7)
            # Count the number of instances where the condition is True
            count_within_range = np.sum(within_range_mask)
            count_diff_group_results[i] += count_within_range

            # Split the partition into 2 groups
            group_idx = []
            for group_number in range(HP['n_groups']):
                group_idx.append(np.where(partition==int(group_number))[0])
            means_diff[i] += np.mean(np.abs(policy[:, -2, group_idx[0], 1]- policy[:, -2, group_idx[1], 1]), axis=(1,0))

            del policy

means_diff = means_diff / n_repetitions

#plot group differeneces in the policy

fig,ax = plt.subplots(figsize=(6,4))
plt.plot(percent_outgroup_neighbors,means_diff, linestyle='-', marker='o', color='#004182')
plt.xlabel('% Outgroup neighbors', fontsize=20)
xticks = np.array([0,3.85,7.69,11.54,15.38,19.23])
labels = list(map(str, np.round(xticks, decimals=0).astype(int)))
plt.xticks(xticks,labels, fontsize=20)
plt.ylabel('Group differences $(\Delta\pi)$', fontsize=20)
plt.yticks(fontsize=20)
plt.grid(False)
plt.ylim(0,means_diff.max()+0.02)
plt.xlim(percent_outgroup_neighbors.min()-0.3,20)

plt.savefig('results/local_global/curve_delta.svg', bbox_inches='tight', format='svg')
plt.show()
