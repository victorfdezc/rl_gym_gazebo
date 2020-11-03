#!/usr/bin/env python

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class QLearnRBF:
    def __init__(self, env, epsilon=0.1, lr=0.1, gamma=0.99, reg_term=0.0, rbf_samplers = None, training_examples = None):
        self.env = env
        self.epsilon = epsilon # Exploration constant (Epsilon-Greedy)
        self.gamma = gamma # Discount factor
        self.actions = range(env.action_space.n)

        # RBF kernels implementation example with Scikit-learn (RBF feature extraction)
        if rbf_samplers == None:
            observation_examples = np.array([env.observation_space.sample() for x in range(10000)])

            scaler = StandardScaler()
            scaler.fit(observation_examples)

            # Concatenation of several RBF kernels with different variance
            featurizer = FeatureUnion([
                    ("rbf1", RBFSampler(gamma=5.0, n_components=500)),
                    ("rbf2", RBFSampler(gamma=2.0, n_components=500)),
                    ("rbf3", RBFSampler(gamma=1.0, n_components=500)),
                    ("rbf4", RBFSampler(gamma=0.5, n_components=500))
                    ])
            featurizer.fit(scaler.transform(observation_examples))
        else:
            scaler = StandardScaler()
            scaler.fit(training_examples)
            featurizer = FeatureUnion(rbf_samplers)
            featurizer.fit(scaler.transform(training_examples))

        self.scaler = scaler
        self.featurizer = featurizer

        self.models = []
        # One linear model per action (estimation of V(s) per action)
        for _ in self.actions:
            # SGDRegressor:
            # loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, 
            # fit_intercept=True, n_iter=5, shuffle=True, verbose=0, 
            # epsilon=0.1, random_state=None, learning_rate='invscaling',
            # eta0=0.01, power_t=0.25, warm_start=False, average=False
            model = SGDRegressor(eta0=lr, alpha=reg_term, learning_rate="constant")
            # Parameter initialization
            model.partial_fit(self.transform([env.reset()]), [0])
            self.models.append(model)

    def transform(self, observations):
        '''
        Applies to the given observations the RBF feature extractor.
        '''
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

    def getQ(self, state):
        '''
        Get the V(s) value for each action
        '''
        X = self.transform([state])
        return np.stack([m.predict(X) for m in self.models]).T

    def chooseAction(self, state):
        '''
        Epsilon-Greedy
        '''
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.getQ(state))

    def learn(self, current_state, current_action, next_reward, next_state):
        '''
        Train the linear model
        '''
        # Calculamos la estimación del retorno esperado:
        G = next_reward + self.gamma * np.max(self.getQ(next_state))
        self.models[current_action].partial_fit(self.transform([current_state]), [G])