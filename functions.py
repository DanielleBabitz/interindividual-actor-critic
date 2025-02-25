#imports
from os import error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
import networkx as nx

def run_all_trials(HP, model = 'interindividual_actor_critic', adj_mat=False, reward_mat = False, turns = 'ordered', thetas_init='uniform', V_init = 'default', connections='fixed', feedback = 'advantage'):
    """
    run multiple trials of individual or interindividual actor-critic
    :param HP: hyperparams (dict)
    :param model: individual or interindividual actor-critic (default is interindividual)
    :param adj_mat: adjacency matrix (numpy array)
    :param reward_mat: reward matrix (numpy array)
    :param turns: type of turns within each round (ordered or random, default is ordered by agent number)
    :param thetas_init: how to initialize the policy (uniform or biased, default is uniform)
    :param V_init: howw to initialize the values (0 for all states or other, default is 0)
    :param connections: type of connections in the adjacency matrix (fixed, random, groups, local_global, default is fixed (given as input))
    :param feedback: type of feedback between agents (advantage or reward, default is advantage)
    :return: policy, value, advantage values for all rounds (numpy arrays)
    :return:
    """
    valid_models = ['interindividual_actor_critic', 'individual_actor_critic']
    if model not in valid_models:
        raise ValueError(f"Invalid model '{model}' specified. Must be one of {valid_models}")

    n_rounds = int(HP['nt']/HP['n_agents'])
    policy_arr, V_arr, advantage_arr = initialize_arrays(HP['trials'], HP['nt'], n_rounds, HP['n_actions'],HP['n_agents'], HP['n_states'])

    for j in range(HP['trials']):

        if j % 100 == 0:
            print('done with trial ' + str(j) + ' out of ' + str(HP['trials']))

        if model == 'interindividual_actor_critic':
            policy, V, adv = interindividual_actor_critic(HP, adj_mat, reward_mat, turns, thetas_init, V_init, connections, feedback)
        if model == 'individual_actor_critic':
            policy, V, adv = individual_actor_critic(HP)
        policy_arr[j,:,:] = np.copy(policy)
        V_arr[j,:] = np.copy(V)
        advantage_arr[j,:,:] = np.copy(adv)

    return policy_arr, V_arr, advantage_arr

def individual_actor_critic(HP):
    """
    run one trial of actor-critic learning
    :param HP: hyperparameter dictionary
    :return: policy array (n_turms),
             Value array (n_trials x n_turns),
             Advantage array (n_trials x n_turns x n_actions)
    """

    V = np.zeros(HP['nt'])         #state value estimates
    pe = np.zeros(HP['nt'])                    #prediction error
    reward = np.zeros(HP['nt'])             #rewards
    a_arr = np.arange(HP['n_actions'])                                 #actions selected
    adv = np.zeros((HP['nt'],HP['n_actions']))                  #array to save advantage
    adv[0,:] = np.nan                                       #initial advantage is defined as nan
    c = np.zeros(HP['n_actions'])         #control
    thetas = np.zeros(HP['n_actions'])    #deafault is uniform initial policy
    policy = np.zeros([HP['nt'],HP['n_actions']]) #policy array

    for i in range(1,HP['nt']-1):
      #softmax policy update
      policy[i,:] = np.exp(HP['dr']*thetas)/np.sum(np.exp(HP['dr']*thetas))
      #sample action
      a = np.random.choice(a_arr, size=1, p=policy[i,:])[0]
      a = a.astype(int)
      #reward and prediction error
      reward[i] = HP['rewards'][a]
      pe[i] = reward[i] - V[i]
      #loss-aversion
      pe[i] = pe[i] * (1 - np.sign(pe[i]) * HP['la'])
      #update state value and advantage
      V[i+1] = V[i] + HP['lrs']*pe[i]

      adv[i,a] = pe[i]
      adv[i,~a] = adv[i-1,~a]

      #policy gradient update
      c[a] = HP['lrp']*HP['dr']*policy[i,a]*(1-policy[i,a])
      thetas[a] =thetas[a] + c[a]*pe[i]

    return policy, V, adv

def interindividual_actor_critic(HP, adj_mat, reward_mat, turns = 'ordered', thetas_init='uniform', V_init = 'default', connections='fixed', feedback = 'advantage'):
    """
    run the interindividual actor-critic algorithm on a social network
    :param HP: hyperparams (dict)
    :param adj_mat: adjacency matrix (numpy array)
    :param reward_mat: reward matrix (numpy array)
    :param turns: type of turns within each round (ordered or random, default is ordered by agent number)
    :param thetas_init: how to initialize the policy (uniform or biased, default is uniform)
    :param V_init: howw to initialize the values (0 for all states or other, default is 0)
    :param connections: type of connections in the adjacency matrix (fixed, random, groups, local_global, default is fixed (given as input))
    :param feedback: type of feedback between agents (advantage or reward, default is advantage)
    :return: policy, value, advantage values for all rounds (numpy arrays)
    """

    #calculate number of rounds
    n = HP['n_agents']
    n_rounds = int(HP['nt']/n) #number of rounds
    if HP['nt'] % n != 0:
        raise ValueError('number of trials must be divisible by number of agents')

    #initialize arrays
    action_arr = np.arange(HP['n_actions'])
    adv = np.zeros((HP['nt'],n,HP['n_actions'], HP['n_states']))                  #array to save advantage
    adv[0,:,:,:] = np.nan                                       #initial advantage is defined as nan
    c = np.zeros((n,HP['n_actions']))         #control
    thetas = np.zeros((n,HP['n_actions']))    #default is uniform initial policy
    policy = np.ones((n_rounds,n,HP['n_actions']))/HP['n_actions'] #initialize policy
    state_mat = np.eye(n).astype(int)   #state matrix, with 1 as the state of the acting agent (self acting), 0 as the state of the other agents (other acting)
    V = np.zeros((HP['nt'], n, HP['n_states']))  # state value estimates

    #if initial values are not 0:
    if V_init != 'default':
        V[0,:,:] = np.copy(V_init)

    #if initial policy is biased (not uniform):
    if thetas_init != 'uniform':
        thetas = np.copy(thetas_init)
        policy[0, :, :] = np.exp(HP['dr'] * thetas) / np.sum(np.exp(HP['dr'] * thetas),axis=1,keepdims=True)

    i=1 #start from the second time step for the advanatge calculation
    while i<HP['nt']-n:
        agents_idx = np.arange(n)
        round = int(i/n)+1

        #option for a random permutation of turns within each round
        if turns == 'random':
            turns_round = np.random.permutation(list(range(n)))
        else:
            turns_round = np.arange(n)

        #type of connections within the network: random regular, division to groups with or without inter-group connections
        if connections == 'random':
           G = nx.random_regular_graph(HP['d'], HP['n_agents'])
           adj_mat = nx.to_numpy_array(G)
           reward_mat = create_reward_mat(HP['rewards'], HP['actions'], adj_mat)

        if connections == 'groups':
            adj_mat = create_adj_matrix_subgroups(HP['n_agents'],HP['d'])

        if connections == 'local_global':
            adj_mat = adj_mat_local_global(HP['n_ingroup'], HP['group_size'], HP['n_outgroup'], HP['n_groups'])
            reward_mat = create_reward_matrix_local_global(HP,adj_mat)

        for j in turns_round:

            #sample action
            a= np.random.choice(action_arr, size=1, p=policy[round-1,j,:])
            a = a.astype(int)
            #reward and prediction error
            reward = np.squeeze(reward_mat[j,:,a])
            pe = reward - V[i-1,agents_idx,state_mat[j]]
            pe = adj_mat[j].T * pe
            #loss aversion - others
            pe = pe * (1 - np.sign(pe) * HP['la'])
            #pe of acting agent
            if feedback == 'reward':
                fb = adj_mat[j] * reward
                fb = fb * (1 - np.sign(fb) * HP['la'])
                pe[j] = reward[j] + HP['ratio'] * np.sum(fb) - V[i - 1, j, 1]
            else:
                pe[j] = reward[j] + HP['ratio'] * np.sum(pe) - V[i - 1, j, 1]
            pe[j] = pe[j] * (1 - np.sign(pe[j]) * HP['la'])
            #update state value and advantage
            V[i] = np.copy(V[i-1])
            V[i,np.arange(n),state_mat[j]] = V[i-1,np.arange(n),state_mat[j]] + HP['lrs']*pe
            #update advantage
            adv[i] = np.copy(adv[i-1])
            adv[i,agents_idx,a,state_mat[j]] = np.copy(pe)
            #policy gradient
            c[j,a] = HP['dr']*policy[round-1,j,a]*(1-policy[round-1,j,a])
            thetas[j,a] =thetas[j,a] + HP['lrp']*c[j,a]*pe[j]
            #softmax policy update - acting agent
            policy[round,j,:] = softmax(thetas[j])
            i += 1

    return policy, V, adv

#HELPER FUNCTIONS

def initialize_arrays(n_trials, n_turns, n_rounds, n_actions, n_agents, n_states):
    """
    initialize empty arrays for tracking simulation progress
    :param n_trials: number of trials in the simulation
    :param n_trials: number of turns in the simulation (total number of actions taken over all agents)
    :param n_rounds: number of rounds in the simulation (number of times each agent takes an action)
    :param n_actions: number of possible actions
    :return: policy, V, advantage, actions, reward, pe arrays
    """
    if n_agents ==1:
        policy_arr = np.zeros((n_trials, n_turns, n_actions))
        V_arr = np.zeros((n_trials, n_turns))
        advantage_arr = np.zeros((n_trials, n_rounds, n_actions))

    elif n_agents > 1:
        policy_arr = np.zeros((n_trials, n_rounds, n_agents, n_actions))
        V_arr = np.zeros((n_trials, n_turns, n_agents, n_states))
        advantage_arr = np.zeros((n_trials, n_turns, n_agents, n_actions, n_states))

    return policy_arr, V_arr, advantage_arr

def create_reward_mat(rewards, actions, adj_mat):
    """
    cretae a reward matrix, given an adjacency matrix, actions and rewards
    :param rewards: rewards for self and other, for each action (numpy array)
    :param actions: action names (list)
    :param adj_mat: adjacency matrix (numpy array)
    :return: reward matrix (numpy array)
    """
    n = len(adj_mat)
    reward_mat = np.zeros((n,n,len(actions)))
    for a in range(len(actions)):
        reward_mat[:,:,a] = adj_mat * rewards[actions[a]][1] #update rewards for connected others
        reward_mat[:,:,a] += np.eye(n)*rewards[actions[a]][0] #update rewards for self
    return reward_mat

def create_adj_matrix_subgroups(total_nodes, group_size):
    """
    divide a fully-connected network of size total_nodes to fully-connected subgroups of size group_size
    :param total_nodes: number of nodes in the network
    :param group_size: size of each subgroup
    :return: adjacency matrix (numpy array)
    """

    # Check if total_nodes is divisible by group_size with no remainder
    if total_nodes % group_size != 0:
        raise ValueError(f"total_nodes ({total_nodes}) must be divisible by group_size ({group_size}) with no remainder.")

    # Number of groups and size of each group
    num_groups = int(total_nodes/group_size)
    # Create an empty adjacency matrix
    adj_matrix = np.zeros((total_nodes, total_nodes))
    # Randomly assign groups of 10 out of 100 nodes
    nodes = np.arange(total_nodes)
    np.random.shuffle(nodes)
    # Select 10 groups of 10 nodes each
    groups = np.array_split(nodes, num_groups)
    # Interconnect nodes within each group to form complete subgraphs (fully connected)
    for group in groups:
        # Connect every node with every other node within the same group
        for i in group:
            for j in group:
                if i != j:
                    adj_matrix[i, j] = 1  # Connect the nodes

    # Ensure the adjacency matrix is symmetric
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    # Set diagonal to zero (no self-loops)
    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix

def adj_mat_local_global(n_ingroup, group_size, n_outgroup, n_groups):
    """
    create an adjacency matrix of n groups with symmetric preferences in connections - the number of groups has to be divisible by 2
    :param n_ingroup: number of ingroup connections (int, float)
    :param group_size: number of agents in each group (int, float)
    :param n_outgroup: number of outgroup connections (int, float)
    :param n_groups: number of groups (int, float)
    :return: adjacency matrix (numpy array)
    """

    #create a separate random-regular graph for each group
    adjacency_matrix = np.zeros((n_groups * group_size, n_groups * group_size), dtype=int)
    G = nx.random_regular_graph(n_ingroup, group_size)
    ingroup_connections = nx.to_numpy_array(G)
    for i in range(n_groups):
        adjacency_matrix[i * group_size:(i + 1) * group_size,
        i * group_size:(i + 1) * group_size] = ingroup_connections

    #connect each ingroup member with n outgroup members
    from scipy import linalg
    for group in range(n_groups-1): #iterate over all groups other than the first one
        for outgroup in range(group+1,n_groups):
            outgroup_idx = np.arange(outgroup*group_size,outgroup*group_size+group_size)
            group_idx = np.arange(group*group_size,(group+1)*group_size)
            outgroup_connections = linalg.circulant(outgroup_idx) #create a circulant matrix of outgroup connections, to draw from
            row = np.arange(group_size)
            for i in range(n_outgroup):
                adjacency_matrix[group_idx, outgroup_connections[:,row[i]]] = 1
                adjacency_matrix[outgroup_connections[:,row[i]], group_idx] = 1

    return adjacency_matrix

def create_symmetric_adjacency_matrix_n_groups(group_size, n_ingroup ,n_outgroup, n_groups):
    """
    Create a symmetric adjacency matrix for a network divided to two groups, whith equal n_outgroup and n_outgroup connections per node
    applies only WHEN BOTH GROUPS ARE OF EQUAL SIZE
    :param group_size: size of each group
    param n_ingroup: number of ingroup connections per node
    :param n_outgroup: number of outgroup connections per node
    :param n_groups: number of groups
    :return: adjacency matrix
    """

    #errors
    if n_groups!=2:
        raise ValueError("currently the function only works for two groups")
    if n_ingroup > group_size:
        raise ValueError("n_ingroup must be less than group_size (no self-connections)")
    if n_outgroup > group_size:
        raise ValueError("n_outgroup must be less than or equal to group_size")

    from scipy import linalg
    import networkx as nx

    # Create an empty adjacency matrix
    adjacency_matrix = np.zeros((n_groups * group_size, n_groups * group_size), dtype=int)
    # Connect nodes within the same group, according to n_ingroup, as a random-regular graph
    if n_groups == 2:
        G = nx.random_regular_graph(n_ingroup, group_size)
        ingroup_connections = nx.to_numpy_array(G)
        for i in range(n_groups):
            adjacency_matrix[i*group_size:(i+1)*group_size, i*group_size:(i+1)*group_size] = ingroup_connections

        #create symmetric outgroup connections, with n_outgroup connections per node
        for group in range(n_groups-1): #iterate over all groups other than the first one
            outgroup_idx = np.setdiff1d(np.arange(group_size*n_groups),np.arange(group*group_size,(group+1)*group_size)) #find the idx of outgroup connections with the next group
            group_idx = np.arange(group*group_size,(group+1)*group_size)
            outgroup_connections = linalg.circulant(outgroup_idx) #create a circulant matrix of outgroup connections, to draw from
            rng = np.random.default_rng()
            random_row = rng.permuted(range(group_size))
            for i in range(n_outgroup):
                adjacency_matrix[group_idx, outgroup_connections[:,random_row[i]]] = 1
                adjacency_matrix[outgroup_connections[:,random_row[i]], group_idx] = 1

    return adjacency_matrix

def create_reward_matrix_n_groups(adj_mat,partition_list, rewards):
    #the function gets a partition of the network and then creates a reward matrix based on the partition,
    #in which ingroup connections are rewarded with the first reward, outgroup connections with the second reward - for each action
    n_actions = len(rewards)
    partition = np.array(partition_list)
    reward_matrix = np.zeros((n_actions, len(partition), len(partition)))
    for i in range(n_actions):
        for j in range(len(partition)):
            ingroup = np.where(partition == partition[j])[0]
            edges = np.where(adj_mat[j] == 1)[0]
            outgroup = np.setdiff1d(edges,ingroup)
            reward_matrix[i, j, ingroup] = rewards[i][0]
            reward_matrix[i, j, outgroup] = rewards[i][1]
            reward_matrix[i, j, j] = rewards[i][0]
    return reward_matrix

def create_reward_matrix_local_global(HP, adj_mat):
    reward_mat = np.stack([adj_mat,adj_mat],axis=2)
    for a in range(HP['n_actions']):
         for group in range(HP['n_groups']):
             if group%2 == 0:
                 #start from first or second, every other agent gets reward:
                 start = group*HP['group_size']
                 end = group*HP['group_size'] + HP['group_size']
                 group_idx = list(range(start,end))
                 prefer_a1 = group_idx[::2]
                 prefer_a2 = group_idx[1::2]
                 reward_mat[:,prefer_a1,a] = HP['rewards'][HP['actions'][a]][0]
                 reward_mat[:,prefer_a2,a] = HP['rewards'][HP['actions'][a]][1]
                 np.fill_diagonal(reward_mat[:,:,a],0)
             else:
                 #start from first or second, every other agent gets reward:
                 start = group*HP['group_size']
                 end = group*HP['group_size'] + HP['group_size']
                 group_idx = list(range(start,end))
                 prefer_a1 = group_idx[1::2]
                 prefer_a2 = group_idx[::2]
                 reward_mat[:,prefer_a1,a] = HP['rewards'][HP['actions'][a]][0]
                 reward_mat[:,prefer_a2,a] = HP['rewards'][HP['actions'][a]][1]
                 np.fill_diagonal(reward_mat[:,:,a],0)

    return reward_mat

#PLOTTING FUNCTIONS

#plot params
policy_colors= ['#005AB4','gray']
value_color = '#DB0B73'
a_do_nothing_color = '#808080'
lw_self = 3
lw_other = 1.5
sns.set(style='white',  rc={"lines.linewidth": 2, 'font.size': 12, 'axes.edgecolor': 'black'})
sns.set_context("notebook", font_scale=1.5)

#plot one agent's learning

def plot_individual_mean_curves(rewards, n_rounds, policy, V, advantage):

    #plot V and A
    plt.figure(figsize=(5, 4))
    plt.axhline(y=rewards[0], color='red', linewidth=1, linestyle='--', label=r"$r_{a1}$")
    plt.axhline(y=rewards[1], color='black', linewidth=1, linestyle='--', label=r"$r_{a2}$")
    plt.plot(np.mean(V, axis=0), label=r"$\hat{V}$", color=value_color, linestyle='--', linewidth=lw_self)
    plt.plot(np.nanmean(np.array(advantage)[:, :-1, 0], axis=0), label=r"$\hat{A_{a1}}$", color=value_color,
             linewidth=lw_self)
    plt.plot(np.nanmean(np.array(advantage)[:, :-1, 1], axis=0), label=r"$\hat{A_{a2}}$", color=a_do_nothing_color,
             linewidth=lw_self)
    plt.xlim(0, n_rounds)
    plt.yticks([-rewards[1], 0, rewards[1]])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Rounds')

    plt.tight_layout()
    plt.savefig('results/model/individual_actor_critic_V_A.svg', bbox_inches='tight', format='svg')
    plt.show()

    #plot policy
    plt.figure(figsize=(5, 4))
    plt.plot(np.nanmean(np.array(policy)[:, 1:-1, 0], axis=0), label=r"$\pi_{a1}$", color=policy_colors[0],
             linewidth=lw_self)
    plt.plot(np.nanmean(np.array(policy)[:, 1:-1, 1], axis=0), label=r"$\pi_{a2}$", color=policy_colors[1],
             linewidth=lw_self)

    plt.xlim(0, n_rounds)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Rounds')
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('results/model/individual_actor_critic_policy.svg', bbox_inches='tight', format='svg')
    plt.show()

#plot learning curves (values, advantage) and policy for interindividual actor-critic

def plot_interindividual_mean_curves(t, rewards, V, advantage, policy, filename, titles=False):
    """
    plot the mean curves of V, advantage and policy curves over all agents, for a given action
    :param t: number of rounds (int)
    :param rewards: rewards for self, other for the plotted action (list)
    :param V: values (numpy array)
    :param advantage: advantages (numpy array)
    :param policy: policies (numpy array)
    :param titles: titles of the plots (optional)
    :param filename: path to save the files
    :return: nan
    """

    #mean - v and advantage
    plt.figure(figsize=(6,4))
    plt.plot(np.mean(V[:,:-1,0,1],axis=0)[::2], label=r"$\hat{V}_{self acting}$", color=value_color, linestyle = '--', linewidth = lw_self)
    plt.plot(np.mean(V[:,:-1,0,0],axis=0)[::2], label=r"$\hat{V}_{other acting}$", color=value_color, linestyle = '--', linewidth = lw_other)
    plt.plot(np.nanmean(advantage[:,:-1,0,0,1],axis=0)[::2], label=r"$\hat{A_{self acting}}$", color=value_color, linewidth = lw_self)
    plt.plot(np.nanmean(advantage[:,:-1,0,0,0],axis=0)[::2], label=r"$\hat{A_{other acting}}$", color=value_color, linewidth = lw_other)
    plt.axhline(y = rewards[0], color = 'red', linewidth=1, label=r"$r_{self}$", linestyle='--')
    plt.axhline(y = rewards[1], color = 'black', linewidth=1, linestyle='--', label=r"$r_{other}$")
    plt.axhline(y = 0, color = 'black', linewidth=1)
    plt.xlim(0,t)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Round')
    plt.tight_layout()
    plt.savefig('results/'+filename+'.svg', format='svg', bbox_inches='tight')
    plt.show()

    #mean policy
    plt.figure(figsize=(6,4))
    plt.plot(np.nanmean(policy[:,:-1,0,0],axis=0), label=r"$\pi_{play music}$", color=policy_colors[0], lw = lw_self)
    plt.plot(np.nanmean(policy[:,:-1,0,1],axis=0), label=r"$\pi_{do nothing}$", color=policy_colors[1], lw = lw_self)
    plt.xlim(0,t)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Round')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig('results/'+filename+'.svg', format='svg', bbox_inches='tight')
    plt.show()