# Simulation Code for “How social norms emerge: the inter-individual actor-critic”

## **Overview**

This repository contains the simulation code used in the paper titled “How social norms emerge: the inter-individual actor-critic”. The simulations, which are described in the paper, aim to demonstrate how the inter-individual actor-critic model explains the emergence of social norms with various properties: prosociality, ingroup bias, stickiness, self-reinforcement, and local conformity / global diversity.

The code is written in Python, and it implements the inter-individual actor-critic model in groups of interacting agents. Below you'll find instructions on how to set up and run the simulations, as well as descriptions of the code structure.

## **Requirements**

Install Python packages: numply, scipy, seaborn, matplotlib, networkx, pickle.

## **Code Structure**

- simulations.py: main simulation script that runs the model and outputs all results included in the paper.
  - To find a specific section, search its’ heading in the paper (e.g., “prosociality”, “stickiness”).
- functions.py: a file with all functions required to run the simulations.
- /results: a folder where simulation results and graphs will be saved, subdivided into the following folders (according to the paper’s structure):
  - /model
  - /prosociality
  - /ingroup_bias
  - /loss_aversion
  - /stickiness
  - /self_reinforcement
  - /local_global
- README.md: this file.

## **Key functions**

### **Interindividual_actor_critic**

Purpose: run the interindividual actor-critic algorithm. The functions assumes there are two states (self acting, other acting).

Inputs:

- HP: hyperparameter dictionary
- adj_mat: adjacency matrix of the network (n_agents x n_agents)
- reward_mat: reward matrix (n_agents x n_agents x n_actions)
- turns: ordered (agents act in order from 1 to n) / random (randomized order)
- thetas_init: initial theta values of all agents. Default is 0 (uniform policy).
- V_init: initial Values of all agents. Default is 0 for all states.
- Connections: an option to create a specific adjacency matrix
  - Default is “fixed” (adjacency matrix is given as input)
  - “random”: random-regular netowrk
  - “groups”: the network is divided to n groups of fully-connected networks.
  - “local_global”: the network is divided to 2 groups of random-regular networks with n_ingroup neighbors, and each agent has n_outgroup aditional neighbors from the other group.
- Feedback: whether the feedback agents communicate to their neighbors is the advnatage or the reward. Default is advantage.

Outputs: numpy arrays with the policy, V, and advantage values for all rounds, for all agents.

Usage: used in all the simulations except the first one, which focuses on the individual actor-critic. The function is called using **run_all_trials()** which takes the same input + a parameter “model” (‘interindividual_actor_critic' or 'individual_actor_critic').

## **Simulation parameters**

Here is a list of important simulation parameters that can be modified:

### **Hyperparameters**

- ‘trials’: the number of trials in the simulation (repetitions of the same game/task).
- ‘nt’: the number of turns each agent has to take an action, within each trial.
- ‘n_actions’: the number of possible actions to choose from.
- ‘lrs’: learning rate of .
- ‘lrp’: learning rate of .
- ‘ratio’: the weight given to others’ feedback (.
- ‘dr’: discount ratio (set to 1).
- ‘n_states’: the number of states (set to 2).
- ‘la’: loss aversion magnitude (0 to 1).
- ‘n_agents’: the total number of agents in the simulation.
- ‘actions’: action names (list).
- ‘rewards’: action rewards from self, other (dict of lists).

## **Expected outputs**

The simulations will generate all the plots included in the paper, in the following folders:

- - /model: figures 1C, 2C
    - /prosociality: figures 3B, 3C
    - /ingroup_bias: figures 4B, 4C
    - /loss_aversion: figures 5B, 5C
    - /stickiness: figures 6B, 6C
    - /self_reinforcement: figures 7B, 7C, 7D
    - /local_global: figures 8B, 8C, 8D

## **Troubleshooting**

The complex network package networkx may raise errors with outputing the adjacency matrix, depending on the version installed. Should this occur, troubleshoot via the networkx documentation (<https://networkx.org>) or contact us for help.

**Notes**

In the simulations of _local conformity / global diversity_ and _self reinforcement_, the first part outputs a single example for each condition. However, several examples may be needed to get the full range of possible results.
