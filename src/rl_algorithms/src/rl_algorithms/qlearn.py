#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class QLearn:
    '''
    Clase que implementa el algoritmo de Q-Learning tabular
    '''
    def __init__(self, env, epsilon=0.1, lr=0.2, gamma=0.9):
        # Iniciamos Q(s,a) como un diccionario vacío. Los diccionarios están 
        # muy bien optimizados en Python, pudiendo buscar en ellos con 
        # complejidad computacional O(1) (los diccionarios son tablas hash). 
        # Los diccionarios son mejores que las listas (estos tienen complejidad
        # O(n)). Por tanto, la solución más rápida en Python es por medio del 
        # uso de diccionarios para almacenar los estados y acciones con su 
        # retorno esperado correspondiente:
        self.Q = {} # Q-table

        self.epsilon = epsilon  # Constante de exploración
        self.learning_rate = lr # Tasa de aprendizaje
        self.gamma = gamma # Tasa de descuento (recompensas futuras)
        self.env = env

        # Obtenemos las posibles acciones por medio del atributo de 
        # espacio de acciones del entorno: 
        self.actions = range(env.action_space.n)

    def getQ(self, state, action):
        '''
        Función para obtener el valor de Q(s,a) dado s y a. En caso
        de que todavía no tengamos un valor de Q(s,a) inicializado para
        esa tupla, devolveremos un valor de 0.0.
        '''
        return self.Q.get((state, action), 0.0)

    def chooseAction(self, state):
        '''
        Epsilon-Greedy
        '''
        if np.random.random() < self.epsilon:
            # Acción aleatoria
            return self.env.action_space.sample()
        else:
            # Mejor acción posible a partir de la estimación actual de Q(s,a)
            return np.argmax([self.getQ(state, a) for a in self.actions])

    def learn(self, current_state, current_action, next_reward, next_state):
        '''
        Aplicamos la ecuación de actualización de Q(s,a) basada en descenso 
        del gradiente:
            Q(s, a) = Q(s, a) + lr * ((reward + gamma * max(Q(s'))) - Q(s,a))
        '''
        oldQ = self.Q.get((current_state, current_action), None)
        # En caso de no encontrar la tupla (current_state, current_action), 
        # inicializamos su valor de Q como la recompensa obtenida:
        if oldQ is None:
            oldQ = next_reward
        # Calculamos la estimación del retorno esperado:
        G = next_reward + self.gamma * np.max([self.getQ(next_state, next_action) for next_action in self.actions])
        # Actualizamos Q(s,a) con la ecuación de descenso del gradiente:
        self.Q[(current_state, current_action)] = oldQ + self.learning_rate * (G - oldQ)