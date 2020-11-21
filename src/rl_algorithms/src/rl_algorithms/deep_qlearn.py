#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class FCLayer:
  '''
  Implementación de una capa fully connected
  '''
  def __init__(self,
              input_size,
              output_size, 
              activation_function=tf.math.tanh, 
              initializer=tf.random_normal_initializer, 
              use_bias=True, 
              batch_norm = False):
    # Instanciamos el objeto del inicializador directamente para luego 
    # poder usar el método __call__ y obtener los valores iniciales:
    self.initializer = initializer()
    self.f = activation_function
    self.use_bias = use_bias
    self.batch_norm = batch_norm

    # Primero creamos una variable TF para los pesos de la capa, los cuales son
    # inicializados mediante el inicializador indicado en los argumentos
    self.W = tf.Variable(initial_value=self.initializer(shape=(input_size, output_size)), trainable = True)
    # Almacenamos los pesos de la capa en una lista de la clase para 
    # hacerlos accesibles más adelante y poder obtener sus valores
    # numéricos (esencialmente para copiar los parámetros de la
    # red neuronal principal a la target network):
    self.params = [self.W]

    # En el caso de usar bias, creamos otra variable TF entrenable
    # y la añadimos a la lista de parámetros
    if use_bias:
      # Los bias se suelen inicializar con valor 0
      self.b = tf.Variable(initial_value=np.zeros(output_size).astype(np.float32), trainable = True)
      self.params.append(self.b)

  def forward(self, X):
    '''
    Función que calcula la salida de una capa fully connected. Para ello,
    primero multiplicamos el vector de entrada por la matriz de pesos, y 
    luego sumamos el bias, para al final aplicar la función de activación.
    '''
    if self.use_bias:
      a = tf.linalg.matmul(X, self.W) + self.b
    else:
      a = tf.linalg.matmul(X, self.W)
    z = self.f(a)
    # Aunque sea un debate, usualmente BN se usa después de la función de
    # activación:
    return tf.keras.layers.BatchNormalization()(z) if self.batch_norm else z


class DQN:
  '''
  Clase que define una Deep Q-Network
  '''
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
    # Almacenamos los atributos del modelo
    self.output_size = output_size # Tamaño del vector de salida
    self.max_experiences = max_experiences # Máximo número de muestras a almacenar en buffer
    self.min_experiences = min_experiences # Mínimo número de muestras necesarias en buffer
    self.batch_size = batch_size # Tamaño del mini-batch
    self.gamma = gamma # Factor de descuento
    self.epsilon = epsilon # Factor de exploración

    # Creamos la estructura de la red neuronal feedforward
    self.layers = []
    in_size = input_size
    for l,out_size in enumerate(hidden_layer_sizes):
      # Parece ser que BN no debe usarse en la última capa (sin contar
      # con la capa de salida)
      if l<len(hidden_layer_sizes)-1:
        layer = FCLayer(in_size, out_size, activation_function=activation_function, batch_norm=batch_norm)
      else:
        layer = FCLayer(in_size, out_size, activation_function=activation_function, batch_norm=False)
      self.layers.append(layer)
      in_size = out_size
    # Y añadimos la capa final (función de activación lineal ya
    # que es un problema de regresión)
    layer = FCLayer(in_size, output_size, activation_function=tf.keras.activations.linear, batch_norm=False)
    self.layers.append(layer)

    # Recogemos todos los parámetros de las capas (weights y biases) 
    # y los almacenamos para poder, más tarde, copiarlos de una red 
    # neuronal a otra:
    self.params = []
    for layer in self.layers:
      self.params += layer.params

    # Creamos placeholders para las entradas de la red
    # neuronal y de la función de coste (debemos tener 
    # en cuenta que la primera dimensión es None porque es 
    # la dimensión del batch, la cual no hace falta definir)
    # Estado de entrada a la red neuronal:
    self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, input_size), name='X')
    # Estimación del retorno (target para calcular el error):
    self.G = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='G')
    # Acciones ejecutadas (para calcular el error):
    self.actions = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='actions')

    # Calculamos la salida de la red neuronal:
    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = Z
    self.prediction = Y_hat # salida de la red neuronal

    # Como la salida de la red neuronal tendrá tantos nodos como acciones,
    # para calcular el error deberemos usar únicamente la predicción realizada
    # para la acción que se ha ejecutado:
    selected_action_values = tf.math.reduce_sum(Y_hat * tf.one_hot(self.actions, output_size), axis=1)
    # Sabiendo la predicción del modelo correspondiente a la acción ejecutada
    # en cada muestra del mini-batch, podremos calcular el error total:
    cost = tf.math.reduce_sum(tf.square(self.G - selected_action_values))
    self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(cost)

    # Inicializamos el diccionario el cual usaremos como buffer para
    # el proceso de repetición de experiencia (experience replay)
    self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

  def set_session(self, session):
    '''
    Método para estabelcer la sesión TF
    '''
    self.session = session

  def copy_from(self, main_model):
    '''
    Método para copiar el valor de los parámetros del modelo dado en los
    argumentos a los parámetros de este modelo
    '''
    ops = []
    # Obtenemos los parámetros de este modelo
    my_params = self.params
    # Y los parámetros del otro modelo
    main_params = main_model.params
    # E iteramos por cada uno de ellos (cada uno de los conjuntos
    # de pesos y bias de cada capa del modelo)
    for p, q in zip(my_params, main_params):
      # Obtenemos el valor numérico de los parámetros del otro modelo
      actual = self.session.run(q)
      # Y le asignamos a nuestros parámetros el valor de los
      # otros parámetros
      op = p.assign(actual)
      # Como en TF siempre trabajamos con operaciones, almacenamos
      # las operaciones de asignación en una lista
      ops.append(op)
    # Para ejecutarlas todas simultáneamente:
    self.session.run(ops)

  def getQ(self, X):
    '''
    Método para ejecutar el modelo y dar una predicción dado el estado
    de entrada al modelo
    '''
    # Primero nos aseguramos de que el vector de entrada tiene dos dimensiones
    # (recordemos que la primera dimensión es la del tamaño del batch)
    X = np.atleast_2d(X)
    # Y ejecutamos el modelo
    return self.session.run(self.prediction, feed_dict={self.X: X})

  def learn(self, target_network):
    '''
    Método para entrenar la red neuronal. Para ello debemos crear un 
    mini-batch teniendo en cuenta la experiencia almacenada en el buffer,
    además de que necesitaremos una target network para crear los targets 
    o salidas deseadas y evitar inestabilidad en el aprendizaje
    '''
    # Si el buffer no contiene el mínimo de experiencia requerido, no 
    # entrenamos el modelo y seguimos ejecutando acciones aleatorias
    if len(self.experience['s']) < self.min_experiences:
      return

    # En caso contrario, seleccionamos un batch aleatorio:
    idx = np.random.choice(len(self.experience['s']), size=self.batch_size, replace=False)
    # Obtenemos los estados, acciones, recompensas y estados siguientes de las
    # muestras obtenidas aleatoriamente del mini-batch:
    states = [self.experience['s'][i] for i in idx]
    actions = [self.experience['a'][i] for i in idx]
    rewards = [self.experience['r'][i] for i in idx]
    next_states = [self.experience['s2'][i] for i in idx]
    dones = [self.experience['done'][i] for i in idx]
    # Calculamos la predicción del modelo para el estado siguiente
    # teniendo en cuenta la mejor acción mediante la target network
    next_Q = np.max(target_network.getQ(next_states), axis=1)
    # Si el estado siguiente es terminal, eso implica que su retorno 
    # es 0, por lo que el retorno estimado del estado actual es simplemente 
    # la recompensa. En caso contrario, el retorno estimado del estado 
    # actual es: 
    #                 reward + gamma*max_a'(Q(s',a'))
    targets = [r + self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

    # Ejecutamos el optimizador para entrenar el modelo dado el mini-batch 
    # obtenido:
    self.session.run(self.train_op, feed_dict={self.X: states, self.G: targets, self.actions: actions})

  def add_experience(self, s, a, r, s2, done):
    '''
    Método para añadir muestras al buffer
    '''
    # Si el buffer ha llegado a almacenar el máximo de muestras permitidas,
    # antes de añadir una nueva muestra deberemos eliminar la muestra más
    # antigua almacenada:
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