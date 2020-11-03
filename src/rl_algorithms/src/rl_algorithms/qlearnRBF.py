#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class QLearnRBF:
    def __init__(self, env, epsilon=0.1, lr=0.1, gamma=0.99, reg_term=0.0, rbf_samplers = None, training_examples = None):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        # Obtenemos las posibles acciones por medio del atributo de 
        # espacio de acciones del entorno: 
        self.actions = range(env.action_space.n)

        # Ahora implementaremos la técnica de extracción de características 
        # mediante funciones RBF. Para ello estandarizaremos la entrada al 
        # modelo y generaremos las características a partir de múltiples RBF 
        # (recordemos que el extractor de características mediante funciones 
        # RBF es equivalente a la capa oculta de la red neuronal de base 
        # radial).
        if rbf_samplers == None:
            # Se genera un conjunto de muestras del espacio de estados para 
            # poder generar un conjunto de funciones RBF óptimas por medio 
            # de RBFSampler. Para ello vamos a hacer uso del método sample() 
            # del espacio de observaciones del entorno de Gym:
            observation_examples = np.array([env.observation_space.sample() for x in range(10000)])

            # Como la entrada al modelo es un estado, podremos crear y ajustar 
            # el objeto que estandarizará la entrada de la red neuronal mediante 
            # el conjunto de muestras previamente generado. La estandarización
            # consistirá en normalizar los datos de entrada de forma que tengan 
            # media igual a 0 y varianza igual a 1:
            scaler = StandardScaler()
            scaler.fit(observation_examples)

            # Usamos FeatureUnion para usar múltiples extractores de 
            # características, en este caso, funciones RBF con diferentes 
            # varianzas lo cual no permitirá cubrir diferentes partes del 
            # espacio de estados. En concreto estamos usando 4 RBFSampler 
            # cada uno con varianzas distintas. Para construir cada RBFSampler,
            # le indicamos la varianza mediante gamma y el número de exemplars 
            # a crear mediante n_components. Mediante este objeto podremos 
            # transformar un estado a un conjunto de características útiles para
            # el modelo lineal. Esto es equivalente a la capa oculta de la red 
            # neuronal de base radial. En este caso, nuestra capa oculta estará 
            # formada por n_components*4 neuronas
            featurizer = FeatureUnion([
                    ("rbf1", RBFSampler(gamma=5.0, n_components=500)),
                    ("rbf2", RBFSampler(gamma=2.0, n_components=500)),
                    ("rbf3", RBFSampler(gamma=1.0, n_components=500)),
                    ("rbf4", RBFSampler(gamma=0.5, n_components=500))
                    ])
            # Generamos las funciones RBF óptimas (es decir, generamos los 
            # exemplars óptimos) mediante las muestras del espacio de estados 
            # previamente recogidas (recordemos que antes deberemos estandarizar 
            # estas muestras). Cada estado de entrada a este extractor de 
            # características será transformado a un conjunto de características  
            # formado por n_components*4 características
            featurizer.fit(scaler.transform(observation_examples))
                    
        else:
            scaler = StandardScaler()
            scaler.fit(training_examples)
            featurizer = FeatureUnion(rbf_samplers)
            featurizer.fit(scaler.transform(training_examples))

        self.scaler = scaler
        self.featurizer = featurizer

        self.models = []
        # Recordemos que vamos a usar un modelo lineal por cada acción
        for _ in self.actions:
            # Parámetros por defecto de SGDRegressor:
            # loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, 
            # fit_intercept=True, n_iter=5, shuffle=True, verbose=0, 
            # epsilon=0.1, random_state=None, learning_rate='invscaling',
            # eta0=0.01, power_t=0.25, warm_start=False, average=False
            model = SGDRegressor(eta0=lr, alpha=reg_term, learning_rate="constant")
            # Debemos hacer un partial_fit con target igual a 0 para inicializar 
            # los parámetros del modelo,  en caso contrario no podremos hacer 
            # ninguna predicción con el modelo. Recordemos que en algunos 
            # entornos esto nos puede servir como método de valores iniciales 
            # optimistas para explorar nuevas acciones:
            model.partial_fit(self.transform([env.reset()]), [0])
            self.models.append(model)

    def transform(self, observations):
        '''
        Mediante este método transformamos las observaciones dadas como
        argumento. Para ello, primero estandarizamos estas observaciones
        para luego introducirlas en el extractor de características
        previamente creado mediante múltiples funciones RBF.
        '''
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

    def getQ(self, state):
        '''
        Para predecir el valor de V(s) dado el estado para cada una de 
        las acciones, primero, transformamos el estado (observación) 
        mediante el extractor de características previamente creado 
        (capa oculta de la red neuronal), y luego introducimos el vector 
        de características obtenido en cada uno de los modelos lineales 
        creados para cada acción, obteniendo una lista con los retornos
        esperados estimados para cada acción para el estado dado.
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
        En este método primero transformaremos el estado s en un vector 
        de características mediante las funciones RBF. Luego, llamamos 
        a partial_fit para actualizar únicamente el modelo correspondiente 
        a la acción que ejecutamos previamente.
        '''
        # Calculamos la estimación del retorno esperado:
        G = next_reward + self.gamma * np.max(self.getQ(next_state))
        self.models[current_action].partial_fit(self.transform([current_state]), [G])