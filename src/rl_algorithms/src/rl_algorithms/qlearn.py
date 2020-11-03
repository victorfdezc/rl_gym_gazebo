#!/usr/bin/env python

import numpy as np

class QLearn:
    '''
    Tabular Q-Learning Class
    '''
    def __init__(self, env, epsilon=0.1, lr=0.2, gamma=0.9):
        # Dictionaries in Python are the fastest object for
        # look up tables
        self.Q = {} # Q-table

        self.epsilon = epsilon  # Exploration constant (Epsilon-Greedy)
        self.learning_rate = lr # Learning rate
        self.gamma = gamma # Discount factor
        self.env = env

        self.actions = range(env.action_space.n)

    def getQ(self, state, action):
        '''
        Get the Q value for a given (state,action) tuple. If that 
        key doesn't exist, 0.0 is returned.
        '''
        return self.Q.get((state, action), 0.0)

    def chooseAction(self, state):
        '''
        Epsilon-Greedy
        '''
        if np.random.random() < self.epsilon:
            # Random action
            return self.env.action_space.sample()
        else:
            # Best action
            return np.argmax([self.getQ(state, a) for a in self.actions])

    def learn(self, current_state, current_action, next_reward, next_state):
        '''
        Q-Learning:
            Q(s, a) = Q(s, a) + lr * ((reward + gamma * max(Q(s'))) - Q(s,a))
        '''
        oldQ = self.Q.get((current_state, current_action), None)
        # If the key is not found, we initialize the value with the given reward
        if oldQ is None:
            oldQ = next_reward
        # Compute the estimation of the expected return
        G = next_reward + self.gamma * np.max([self.getQ(next_state, next_action) for next_action in self.actions])
        # Update Q(s,a)
        self.Q[(current_state, current_action)] = oldQ + self.learning_rate * (G - oldQ)