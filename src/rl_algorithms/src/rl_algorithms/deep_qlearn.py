#!/usr/bin/env python

import numpy as np
import tensorflow as tf

class FCLayer:
  '''
  Fully connected layer
  '''
  def __init__(self,
              input_size,
              output_size, 
              activation_function=tf.math.tanh, 
              initializer=tf.random_normal_initializer, 
              use_bias=True, 
              batch_norm = False):
    
    self.initializer = initializer()
    self.f = activation_function
    self.use_bias = use_bias
    self.batch_norm = batch_norm

    # TF variable for weights
    self.W = tf.Variable(initial_value=self.initializer(shape=(input_size, output_size)), trainable = True)
    # Store net params for later use
    self.params = [self.W]

    # Add bias in the same way as weights
    if use_bias:
      self.b = tf.Variable(initial_value=np.zeros(output_size).astype(np.float32), trainable = True)
      self.params.append(self.b)

  def forward(self, X):
    '''
    Output of the fully connected layer
    '''
    if self.use_bias:
      a = tf.linalg.matmul(X, self.W) + self.b
    else:
      a = tf.linalg.matmul(X, self.W)
    z = self.f(a)

    return tf.keras.layers.BatchNormalization()(z) if self.batch_norm else z


class DQN:

  def __init__(self,
              input_size,
              output_size,
              hidden_layer_sizes,
              epsilon=0.1,
              lr=1e-2,
              gamma=0.99,
              max_experiences=10000,
              min_experiences=100,
              batch_size=32,
              activation_function=tf.math.tanh,
              batch_norm=False):
    # Model params:
    self.output_size = output_size # Output size
    self.max_experiences = max_experiences # Buffer max. experiences
    self.min_experiences = min_experiences # Buffer min. experiences
    self.batch_size = batch_size # Batch size
    self.gamma = gamma # Discount factor
    self.epsilon = epsilon # Exploration factor

    # Feedforward Network structure:
    self.layers = []
    in_size = input_size
    for l,out_size in enumerate(hidden_layer_sizes):
      # It is not recommended to add BN to the last fc layer
      if l<len(hidden_layer_sizes)-1:
        layer = FCLayer(in_size, out_size, activation_function=activation_function, batch_norm=batch_norm)
      else:
        layer = FCLayer(in_size, out_size, activation_function=activation_function, batch_norm=False)
      self.layers.append(layer)
      in_size = out_size
    # Output layer
    layer = FCLayer(in_size, output_size, activation_function=tf.keras.activations.linear, batch_norm=False)
    self.layers.append(layer)

    # Get all params
    self.params = []
    for layer in self.layers:
      self.params += layer.params

    # Create placeholders for the input, desired output and executed actions:
    self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, input_size), name='X')
    self.G = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='G')
    self.actions = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='actions')

    # Compute network output
    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = Z
    self.prediction = Y_hat

    # Select only the outputs corresponding to the executed actions
    selected_action_values = tf.math.reduce_sum(Y_hat * tf.one_hot(self.actions, output_size), axis=1)
    # And compute cost
    cost = tf.math.reduce_sum(tf.square(self.G - selected_action_values))
    self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(cost)

    # Experience replay
    self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

    # Create placeholder for parameters assignment (target network)
    self.assigned_value = tf.compat.v1.placeholder(tf.float32,shape=(None))
    # Store assign ops to run later
    self.assign_ops = []
    for p in self.params:
      self.assign_ops.append(p.assign(self.assigned_value))

  def set_session(self, session):
    '''
    Set TF session
    '''
    self.session = session

  def copy_from(self, main_model):
    '''
    Copy the values of the params of the given model to our network
    '''
    # Get params from the other model (main model)
    main_params = main_model.params
    for c, v in enumerate(main_params):
      # Get the value of the main model parameters and assign it
      # to our parameters
      value = self.session.run(v)
      self.session.run(self.assign_ops[c], feed_dict={self.assigned_value: value})

  def getQ(self, X):
    '''
    Execute the model and get a prediction
    '''
    # Make sure the input is a matrix
    X = np.atleast_2d(X)
    return self.session.run(self.prediction, feed_dict={self.X: X})

  def learn(self, target_network):
    '''
    Train the model given the target network and the experience buffer
    '''
    if len(self.experience['s']) < self.min_experiences:
      return

    # Get random samples from the experience buffer to create the mini-batch
    idx = np.random.choice(len(self.experience['s']), size=self.batch_size, replace=False)
    states = [self.experience['s'][i] for i in idx]
    actions = [self.experience['a'][i] for i in idx]
    rewards = [self.experience['r'][i] for i in idx]
    next_states = [self.experience['s2'][i] for i in idx]
    dones = [self.experience['done'][i] for i in idx]
    next_Q = np.max(target_network.getQ(next_states), axis=1)
    # If the next state is terminal, its return is 0, so the return of
    # the current state is the reward. If not, the return of the current
    # state is:
    #                 reward + gamma*max_a'(Q(s',a'))
    targets = [r + self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

    # Execute the optimizer to minimize the cost
    self.session.run(self.train_op, feed_dict={self.X: states, self.G: targets, self.actions: actions})

  def add_experience(self, s, a, r, s2, done):
    '''
    Add new samples to the experience buffer
    '''
    if len(self.experience['s']) >= self.max_experiences:
      self.experience['s'].pop(0)
      self.experience['a'].pop(0)
      self.experience['r'].pop(0)
      self.experience['s2'].pop(0)
      self.experience['done'].pop(0)
    self.experience['s'].append(s)
    self.experience['a'].append(a)
    self.experience['r'].append(r)
    self.experience['s2'].append(s2)
    self.experience['done'].append(done)

  def chooseAction(self, X):
    '''
    Epsilon-Greedy
    '''
    if np.random.random() < self.epsilon:
      return np.random.choice(self.output_size)
    else:
      return np.argmax(self.getQ(X)[0])